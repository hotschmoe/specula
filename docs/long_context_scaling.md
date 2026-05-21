# long_context_scaling.md — scaling specula's pathb pipeline to 32k / 64k / 128k

Structural design doc. This is **not** a low-ctx sweep — Qualcomm
already ships `cl{512,1024,2048,3072,4096}`, and `docs/e2e_optimizations.md`
task 11 covers re-running our parametric pipeline across that range. The
question here is what it *structurally* takes to push **ctx 32k / 64k /
128k** on the Snapdragon X2 Elite (48 GB unified LPDDR5X), first for
Qwen3-4B and then for the real target, Qwen3.6-27B.

Status: analysis/design (Session 30). Companion to `docs/e2e_optimizations.md`
(quality plan), `docs/qai_hub_recipe.md` (Qualcomm recipe), `docs/roadmap.md`
B5 (long-context compile tiers).

## 0. TL;DR — the three structural conclusions

1. **KV-cache memory is the single dominant constraint, and it is the
   only thing that grows with ctx.** Weights are fixed; the attention
   mask is trivial; activations are ~ctx-independent at ar1. For
   Qwen3-4B the KV cache is **144 KiB/token at int16, 72 KiB/token at
   int8** (`2 · 36 layers · 8 KV heads · 128 head_dim · dtype_bytes`).
   At 128k that is **18 GiB (int16) / 9 GiB (int8)** of KV — large, but
   *within* a 48 GB device.

2. **128k for Qwen3-4B (dense, global attention) is memory-feasible on
   48 GB but impractical on per-step latency and compile cost.** int8
   KV at 128k is 9 GiB, ~14 GiB total with weights — it fits. What does
   *not* scale is per-decode-step cost: every ar1 step streams the
   whole KV cache, an ~80 ms/token floor at 128k. The structural fix is
   *changing the attention structure* — sliding-window / streaming
   attention caps the KV at a fixed window so both memory *and* per-step
   cost stop scaling with ctx. With a 4k window + attention sinks,
   "128k" becomes a 4k-resident-KV problem (0.28 GiB int8) served by a
   single binary. **The biggest blocker is dense-global-attention's
   per-step KV streaming + per-ctx compile ladder; the biggest lever is
   sliding-window attention.**

3. **Qwen3.6-27B is the harder case but a *different* case — it is an
   SSM+attention hybrid (Mamba2-style), not a dense transformer.** Only
   its handful of attention layers carry an O(ctx) KV cache; the SSM
   layers carry an O(1) recurrent state. That makes long context
   *structurally cheaper* for 27B than a naïve "27B is bigger so KV is
   bigger" projection — but it also means the pathb split, the KV
   tensor layout, and `genie_config.json`'s `kv-dim` knob all need a
   hybrid-aware rewrite that the current Qwen3-dense pipeline does not
   have.

Headline numbers (Qwen3-4B, dense, global attention, full KV resident;
weights w4a16 ≈ 2.5 GB resident; total ≈ KV + weights + ~1-2 GB runtime):

| ctx | KV int16 | KV int8 | total int16 / int8 | fits 44 GB usable? |
|----:|---------:|--------:|-------------------:|:-----------:|
| 4k   | 0.56 GiB | 0.28 GiB | ~5 / ~5 GiB    | trivial |
| 32k  | 4.5 GiB  | 2.25 GiB | ~9 / ~7 GiB    | comfortable |
| 64k  | 9.0 GiB  | 4.5 GiB  | ~14 / ~9 GiB   | fits |
| 128k | 18.0 GiB | 9.0 GiB  | ~23 / ~14 GiB  | **fits** (latency, not memory, is the limit) |

---

## 1. KV-cache memory scaling — the dominant constraint

### 1.1 The formula

KV cache holds, per token, the K and V projections for every layer and
every KV head:

```
bytes_per_token = 2 (K and V)
                * num_layers
                * num_kv_heads
                * head_dim
                * dtype_bytes
```

This is confirmed by the split layout in `end-to-end/lib/split.py`
(`decode_inputs`/`decode_outputs`): each layer contributes a
`past_key_values.{L}.key` and `.value` tensor of shape
`[1, num_kv_heads, ctx, head_dim]`, and by the Qualcomm reference
bundle's `metadata.json` — the decode (`ar1`) parts take
`past_key_0_in [8,1,128,511]` (KV heads × batch × head_dim × past) and
emit `past_key_0_out [8,1,128,1]` (just the **one new token's** slice).
That `_out` shape is the structural proof that **the KV cache is a
runtime tensor, not part of the compiled weights** — the graph only
ever produces the incremental slice; the genie runtime owns the full
`[…, ctx, …]` buffer (see §3, §4).

### 1.2 Qwen3-4B (dense)

From `models/Qwen3-4B/config.json`: `num_hidden_layers = 36`,
`num_key_value_heads = 8`, `head_dim = 128`.

```
per-token = 2 * 36 * 8 * 128 * dtype_bytes
          = 73,728 * dtype_bytes
```

- **int16 KV** (dtype_bytes = 2): `73,728 × 2 = 147,456 B/token = 144 KiB/token`
- **int8 KV** (dtype_bytes = 1): `73,728 × 1 = 73,728 B/token = 72 KiB/token`

| ctx | tokens | KV int16 | KV int8 | KV int4 (notional) |
|----:|-------:|---------:|--------:|-------------------:|
| 4,096   | 4 K   | 0.5625 GiB | 0.281 GiB | 0.141 GiB |
| 32,768  | 32 K  | 4.5 GiB    | 2.25 GiB  | 1.125 GiB |
| 65,536  | 64 K  | 9.0 GiB    | 4.5 GiB   | 2.25 GiB  |
| 131,072 | 128 K | 18.0 GiB   | 9.0 GiB   | 4.5 GiB   |

Sanity-check the arithmetic trap: `num_kv_heads` (8) belongs in the
per-token term **exactly once**. `2 · 36 · 8 · 128 · 2 = 147,456 B`
per token; `× 131,072 tokens = 19.3 GB = 18.0 GiB` int16 at 128k. It
is easy to over-count KV heads (e.g. once in the per-token term and
again from a `[8,1,…]`-leading tensor shape) and land at a spurious
144 GiB — the figures above are the correct ones.

### 1.3 Qwen3-4B — does it fit 48 GB?

Budget: 48 GB unified LPDDR5X, of which the BIOS exposes ~44 GB to the
accelerators (`docs/2026-05-13_qwen3_6_27b_mtp.md` confirms the ~44 GB
ceiling). Weights + KV + activations + the genie/HTP runtime working
set must co-reside.

- Qwen3-4B w4a16 weights ≈ **2.4 GB** (4-part bundle; the reference is
  3.0–3.6 GB *on disk* including int16 lm_head, but the resident weight
  footprint memory-mapped is the int4 payload — call it ~2.5 GB).
- Activations at ar1 are negligible (a few MB — single-token hidden
  states); at ar128 prefill they grow but stay sub-GB.
- genie/HTP spill-fill + scratch: order ~1–2 GB.

| ctx | KV int16 | total int16 | KV int8 | total int8 | verdict |
|----:|---------:|------------:|--------:|-----------:|---------|
| 4k   | 0.56 GiB | ~5 GiB   | 0.28 GiB | ~5 GiB   | trivial |
| 32k  | 4.5 GiB  | ~9 GiB   | 2.25 GiB | ~7 GiB   | comfortable |
| 64k  | 9.0 GiB  | ~14 GiB  | 4.5 GiB  | ~9 GiB   | fits |
| 128k | 18.0 GiB | ~23 GiB  | 9.0 GiB  | ~14 GiB  | **fits** |

**Conclusion for 4B:** dense global attention at 128k *does fit* a
48 GB device — KV is 18 GiB int16 / 9 GiB int8, leaving comfortable
headroom. The constraint at 4B scale is not memory capacity; it is
**compute and bandwidth per decode step** (every step does attention
over the full ctx — see §1.5) and **per-ctx compile cost** (§3, §6).

### 1.4 Qwen3.6-27B — the hybrid changes everything

`docs/2026-05-13_qwen3_6_27b_mtp.md` is explicit: Qwen3.6-27B is **not a
dense transformer**. It has **65 blocks**: blocks 0–62 are
SSM+attention hybrid (Mamba2-style), block 63 is attention-only, block
64 is the MTP head. A hybrid block typically interleaves a Mamba2/SSM
mixer with a periodic full-attention layer.

This is the key structural fact for long context:

- **SSM/Mamba2 layers carry an O(1) recurrent state**, not an O(ctx)
  KV cache. Their memory footprint is *independent of context length*.
- **Only the full-attention layers carry an O(ctx) KV cache.**

So the naïve projection "27B is ~7× the params of 4B, so KV is ~7×
bigger" is **wrong**. KV scales with the *number of attention layers*,
not total layers. Hybrids like this typically run attention on a small
fraction of layers (commonly 1-in-4 to 1-in-8).

Worked projection (assumptions flagged — the exact 27B config is not on
disk; `models/` holds only the GGUFs). Assume 27B-class attention
layers have `head_dim = 128`, `num_kv_heads ≈ 8` (GQA), and that
**~1/6 of the 64 transformer blocks are full-attention** → ~10–12
attention layers:

```
per-token (attn layers only) = 2 * N_attn * num_kv_heads * head_dim * dtype_bytes
                             ≈ 2 * 11 * 8 * 128 * dtype_bytes
                             = 22,528 * dtype_bytes
```

- int16: ~44 KiB/token  · int8: ~22 KiB/token

| ctx | 27B KV int16 (attn-only) | 27B KV int8 |
|----:|-------------------------:|------------:|
| 4k   | 0.17 GiB | 0.086 GiB |
| 32k  | 1.4 GiB  | 0.69 GiB  |
| 64k  | 2.75 GiB | 1.4 GiB   |
| 128k | 5.5 GiB  | 2.75 GiB  |

Plus the SSM recurrent state — a fixed (ctx-independent) per-layer
state of order a few MB total. **The hybrid's KV cache at 128k is
~5.5 GiB int16 — smaller than dense Qwen3-4B's.**

But the *weights* dominate 27B's budget: Q8_0 is 27 GiB, Q4_0 ~15 GiB
(`docs/2026-05-13_qwen3_6_27b_mtp.md`). A w4a16 NPU bundle would land
near ~14–16 GB resident.

| 27B config | weights | + KV 128k int16 | total | fits 44 GB? |
|---|---:|---:|---:|:--:|
| w4a16-equiv | ~15 GiB | 5.5 GiB | ~22 GiB + runtime | **yes** |
| w8a16-equiv | ~27 GiB | 5.5 GiB | ~34 GiB + runtime | yes, tight |

**Conclusion for 27B:** *because* it is a hybrid, 128k context is
memory-feasible on a 48 GB device at w4a16 — the SSM layers make long
context cheap. The blocker for 27B is **not memory**; it is that the
entire pathb pipeline (`rewrite_qwen3_pathb.py`, `lib/split.py`,
`lib/model_config.py`) is **Qwen3-dense-only** and has no concept of an
SSM mixer, a recurrent state tensor, or a hybrid block. See §7.

### 1.5 The compute side of long ctx (not just memory)

Even when KV *fits*, every ar1 decode step computes attention scores
over the full resident ctx: the QK^T and the softmax·V are O(ctx) per
step. At 128k that is 32× the attention work of a 4k step. On a
bandwidth-bound NPU the dominant cost is *streaming the whole KV cache
through HVX every token* — at 18 GiB int16 KV and 228 GB/s that is a
~80 ms/token floor *just to read KV*, before any matmul. This is the
second reason (after compile cost, §6) that dense global attention at
128k is impractical even where it is memory-feasible, and the second
argument for sliding-window attention (§5.2): a fixed 4k window caps
per-step KV streaming at ~2.25 GiB regardless of nominal ctx.

---

## 2. RoPE / position scaling

### 2.1 The trained-context ceiling

`models/Qwen3-4B/config.json`: `max_position_embeddings = 40960`,
`rope_theta = 1e6`, `rope_scaling = null`.

- **32k (32768) is *within* the 40960 trained window** — no RoPE
  scaling needed. A ctx-32k bundle just needs a longer cos/sin table.
- **64k and 128k *exceed* 40960** — the model was never trained at
  those positions. Feeding raw RoPE positions ≥ 40960 produces
  out-of-distribution rotary phases and quality collapses. Position
  scaling is mandatory.

### 2.2 How RoPE enters the pathb graph

This is the pipeline's biggest structural advantage for long context.
`docs/qai_hub_recipe.md` and `rewrite_qwen3_pathb.py` confirm: **RoPE
is hoisted out of the graph.** `Qwen3RotaryEmbedding.forward` is
bypassed; `cos`/`sin` become **graph inputs** `position_ids_cos` /
`position_ids_sin`. `end-to-end/lib/rope.py::build_rope_cache` computes
the table host-side:

```python
inv_freq = 1.0 / (rope_theta ** (arange(0, half) / half))
freqs    = outer(positions, inv_freq)
cos, sin = cos(concat[freqs,freqs]), sin(...)
```

The compiled graph never sees `rope_theta` or a position index — it
only consumes a `[1,1,head_dim]` (decode) cos/sin slice. The reference
bundle's `position_ids_cos/sin` are `[1,1,1,64]` uint16 graph inputs.

**Consequence:** RoPE *position scaling* is a **host-side change to
`lib/rope.py` only**. The pathb graph, the split, and the compiled
`.bin`s do not change at all for RoPE. This is unusually clean —
on a model with RoPE *inside* the graph, long-ctx scaling would force a
recompile.

### 2.3 Which scaling, and what it costs

Three standard options, in order of quality:

1. **Linear (position interpolation, PI).** Divide every position by a
   factor `s = target_ctx / 40960`. Trivial host-side change. Costs the
   most quality (uniformly compresses *all* frequencies, hurting local
   resolution); needs fine-tuning to be good. Cheapest to ship as a
   zero-training stopgap.
2. **NTK-aware scaling.** Scale `rope_theta` itself (`theta' = theta *
   s^(dim/(dim-2))`) instead of the positions. One-line change in the
   `build_rope_cache` call (`rope_theta` argument). Better than linear
   without fine-tuning because it preserves high-frequency (local)
   resolution and stretches only low frequencies. **The cheapest
   credible 64k/128k path** — note `genie_config.json` carries a
   `rope-theta: 1000000` knob, so genie itself can be told the new
   theta if its internal RoPE is used; in our pathb pipeline the table
   is host-built so even that is moot.
3. **YaRN.** Per-frequency interpolation (ramp between NTK and linear
   across the frequency bands) + an attention-temperature factor.
   Best quality at 4×–8× extension, and the de-facto standard
   (Qwen2.5/Qwen3 long-ctx variants ship YaRN `rope_scaling` configs).
   Still a **host-side-only** change: YaRN only changes how `inv_freq`
   and a logit-scale constant are computed; it does not add graph ops.
   The attention-temperature factor folds into the existing attention
   scale (already folded into K per `docs/qai_hub_recipe.md` §1).

**Recommendation:** ship **YaRN** for 64k/128k (it is what the Qwen
team themselves use for long-ctx Qwen3 checkpoints), **NTK-aware** as
the no-config-change fallback, and **nothing** for 32k (in-window).
Implementation is entirely in `lib/rope.py` — add a `rope_scaling`
parameter (the `ModelInfo.rope_scaling` field already exists in
`model_config.py`, and `FamilyConfig.rope_scaling_supported` is the
gate flag — currently `False` for qwen3, which the
`load_model_info` guard turns into a hard `NotImplementedError`). The
single concrete code task is: implement YaRN in `build_rope_cache`,
flip `rope_scaling_supported=True` for the qwen3 family, and verify the
pathb rotary hoist's identity-`attention_scaling` assertion still holds
(YaRN's logit scale is applied to attention, not to cos/sin, so the
hoist assertion is unaffected — but this must be checked, per the TODO
already in `model_config.py`).

### 2.4 cos/sin table size — negligible

A cos/sin table for 128k is `131072 * 128 * 4 B * 2 (cos+sin)` =
**128 MiB** host-side, and only a `[1,1,128]` slice is fed per decode
step. Immaterial against the KV budget. Long ctx does **not** need a
"different graph" for RoPE — just a longer host table (and, for
>40960, a scaled `inv_freq`).

---

## 3. What changes in the compiled pipeline

### 3.1 Per-part `.bin` size is independent of ctx — confirmed

The pathb graph pins ctx (`pin_shapes_qwen3_4b.py`: `--ctx` sets
`past_sequence_length = ctx-1`, `seq_k = ctx`, and the
`Concatpresent.*` KV dims). But the compiled `.bin` contains **weights
+ graph structure**, not the KV cache. The KV cache is a runtime I/O
tensor (§1.1: the graph's `present.*` outputs are the *incremental*
`[…,1,…]` slice; genie owns the full buffer). Therefore:

- **Weights are fixed** across ctx. A ctx-128k part `.bin` has the same
  weight payload as a ctx-512 part `.bin`.
- What *does* change with ctx in the `.bin`: the **shape constants**
  baked into the graph (attention mask `[1,1,1,ctx]`, the Slice ranges,
  the KV input tensor declared shapes `[1,kv_heads,ctx-1,head_dim]`),
  the **HTP scratch/tiling decisions** the compiler makes for those
  shapes, and the **graph-prep'd tiling of the attention matmuls**. The
  payload delta is small (kilobytes of shape metadata); the *compiler's
  tiling choices* differ materially.

This matches the on-disk reference: the four part `.bin`s
(778 / 669 / 669 / 1070 MB) are the *cl512* set; the bundle ships
**40 binaries total** = 4 parts × 5 ctx tiers × 2 ar modes (`ar1`,
`ar128`) — confirmed by `metadata.json` having 40 `model_files`
entries. The bytes are dominated by weights (shared via
`weight_sharing_enabled` in `htp_backend_ext_config.json`), so the
5 ctx tiers do **not** cost 5× the disk — but each is still a
*separately compiled context binary*.

### 3.2 Why each ctx needs its own compiled `.bin`

HTP context binaries are **statically shaped**. `qairt-converter` /
`qnn-context-binary-generator` lower a graph with fully concrete dims
(that is the entire reason `pin_shapes_qwen3_4b.py` exists). The HTP
graph-prep stage then makes ctx-dependent decisions: VTCM tiling of the
`[ctx, head_dim]` attention operands, the spill/fill schedule, the
loop structure of the `QK^T` and `softmax·V` matmuls. There is no
"dynamic ctx" — a binary compiled for `seq_k=512` physically cannot run
`seq_k=4096`.

So a 32k/64k/128k deployment needs **its own compiled `.bin` set per
ctx tier**, exactly as Qualcomm ships 5 tiers. With
`weight_sharing_enabled` the weight payload is shared across tiers in
the bundle, so the marginal cost of an extra tier is the per-tier
graph metadata + a recompile, not a full weight copy. The roadmap
already names this: **B5 "long-context compile tiers"** — a tiered
loader that picks the smallest tier ≥ current prompt length.

### 3.3 The attention mask is trivial at any ctx

The pathb mask is `[1,1,1,ctx]` (see `split.py` `mask_shape`, and
`metadata.json`'s `attention_mask [1,1,1,512] uint16`). At 128k it is a
single `[1,1,1,131072]` uint16 tensor = **256 KiB**. It is threaded
cross-part by `detect_shared_attn_mask` in `split.py`. Long ctx adds
nothing structural here — the mask is a rounding error against KV.

### 3.4 Split: ctx flows through but the layout does not change

`lib/split.py::build_part_specs` already takes `ctx` and stamps it into
every `past_key_values.{L}` / `present.{L}` shape. Scaling ctx is a
**parameter change, not a code change** — the split is already
ctx-parametric. (The one caveat for 27B: the split assumes every
decoder block has the same KV layout; a hybrid block has none — see §7.)

---

## 4. Genie / HTP runtime — KV management at long ctx

From `genie_config.json` and `htp_backend_ext_config.json`:

```json
"QnnHtp": {
  "use-mmap": true, "spill-fill-bufsize": 0, "mmap-budget": 0,
  "kv-dim": 128, "pos-id-dim": 64, "rope-theta": 1000000
},
"context": {"weight_sharing_enabled": true},
"memory": {"mem_type": "shared_buffer"}
```

- **`use-mmap: true`** — weights are memory-mapped from the `.bin`
  files. The OS pages them in on demand; they are *not* all resident,
  and they are clean/backed pages the kernel can evict under pressure.
  This is what lets the KV cache (dirty, non-evictable, anonymous
  memory) have first claim on RAM. For long ctx this matters: the KV
  cache is the *resident* cost; weights are *mmap-budgeted*.
- **`mmap-budget: 0`** — "no cap": let the OS manage the weight
  mapping. At long ctx, if KV pressure is high, capping `mmap-budget`
  (e.g. to the weight size) bounds how much RAM the weight mapping is
  allowed to hold resident, deliberately ceding RAM to KV. **This is a
  long-ctx tuning knob:** on a 128k run the policy should be "weights
  mmap-budget-capped, KV fully resident."
- **`spill-fill-bufsize: 0`** — the HTP spill/fill buffer (the
  scratch DRAM region the HTP controller uses to spill VTCM tiles
  during a graph execution) is auto-sized. For long-ctx attention the
  per-step working set (the `[ctx, head_dim]` tiles) is larger, so the
  spill/fill buffer grows; at 128k it may need to be explicitly sized
  rather than left at 0. It is *per-execution scratch*, not the KV
  cache — it does not scale with ctx the way KV does, but it does grow
  with the attention tile footprint.
- **`kv-dim: 128`** — the head_dim of the KV cache (= `head_dim`).
  genie uses this to allocate and stride the KV buffer. It is **fixed
  by the model** (128 for Qwen3-4B), *not* a long-ctx knob. The ctx
  length itself is `dialog.context.size` (4096 in the reference).
- **`pos-id-dim: 64`** — the RoPE position dimension (`head_dim/2`).
  Model-fixed; not a long-ctx knob.
- **`dialog.context.size`** — *this* is the ctx knob genie reads. It
  sets the KV buffer length. It must match the compiled tier's `seq_k`.

**KV residency model.** genie allocates one contiguous KV buffer of
`context.size` tokens up front (`shared_buffer` mem type → it is in the
unified LPDDR5X pool visible to the HTP). It is dirty, anonymous,
non-evictable memory — it is the long-ctx footprint of §1. Weights, by
contrast, are mmap'd and evictable. So the runtime memory equation at
long ctx is:

```
resident = KV_buffer(ctx)  +  HTP_spill_fill  +  activations  +  (paged-in fraction of mmap'd weights)
```

and the lever is: **cap the weight mapping (`mmap-budget`), let KV
own RAM.** genie has no paged-KV / KV-eviction mechanism today — the
KV buffer is monolithic and fully resident. That is exactly why §5's
feasibility levers are needed for the cases where §1's table says
"NO."

---

## 5. Feasibility levers for 64k / 128k

For Qwen3-4B dense at 128k the §1.3 table says int8 KV *fits* 48 GB
(~14 GiB total) — so for 4B the levers are about **per-step compute /
bandwidth** (§1.5) and **compile cost** (§6), not capacity. For a
larger *dense* model, or to drive per-step latency down, the levers
below are what make 128k *practical* rather than merely *resident*.
Ordered by structural impact.

### 5.1 KV-cache quantization (int8 / int4) — already half-done

- **int8 KV.** `docs/qai_hub_recipe.md` P1 already calls for **int8
  symmetric, in/out-tied** KV cache — the Qualcomm reference ships
  `past_key/value` as **uint8** (`metadata.json`). So int8 KV is not a
  new lever; it is the *recipe we are already adopting for quality
  parity*. It halves the §1.2 numbers for free. **This is the baseline
  for any long-ctx build** — int16 KV at long ctx is simply wasteful.
- **int4 KV.** Halves again (4.5 GiB at 128k for 4B). Structurally it
  needs: a 4-bit encoding on the `past_key/value` graph
  inputs/outputs in `lib/aimet.py`, and HTP support for int4 KV
  tensors (the weight path already uses int4; the KV-tensor path is
  less trodden). Quality cost: KV is more sensitive to quantization
  than weights (it is the attention *state*); int4 KV typically costs
  a few points of long-context retrieval accuracy. Worth it only when
  capacity-bound.
- **Structural change:** `lib/aimet.py` KV-tensor bitwidth + the
  `split.py` KV tensor dtype; the pathb graph itself is unchanged
  (KV dtype is an encoding, not a graph op).

### 5.2 Sliding-window attention (SWA) — the real lever

This is the one structural change that makes KV memory **stop scaling
with ctx**. With a window `W`, each layer attends only to the last `W`
tokens, so the resident KV is capped at `W` tokens *regardless of
nominal ctx*:

```
KV_resident = bytes_per_token * W      (not * ctx)
```

For Qwen3-4B int16 with `W = 4096`: KV is **fixed at 0.5625 GiB**
whether the conversation is 8k or 128k. Per-step attention compute is
also capped at `W` (fixes §1.5's bandwidth floor).

- **Structural changes:**
  - The attention mask becomes a *band* mask instead of a full causal
    triangle — still `[1,1,1,ctx_window]`, trivially shaped.
  - The pathb graph's KV input shape pins to `W`, not `ctx` — so the
    compiled `.bin` is a *fixed* `seq_k=W` binary; **you no longer need
    a per-ctx tier ladder.** One `W=4096` binary serves any
    conversation length. This *removes* the §3.2 per-ctx-compile
    problem.
  - genie's KV buffer is `W`-sized and used as a **ring buffer** — the
    runtime evicts the oldest token's KV slot as each new token's KV is
    written. genie does not do this today (its KV buffer is monolithic,
    §4); a ring-buffer KV manager is the runtime work item.
- **Quality cost:** pure SWA *loses* information beyond `W` tokens — a
  fact at position 0 is gone once the window slides past it. For many
  workloads (chat, code-in-context) acceptable; for long-document QA /
  retrieval it is not. That is what attention sinks fix (§5.3).
- **Note:** Qwen3-4B was *not* trained with SWA (`sliding_window: null`
  in config.json) — `docs/roadmap.md` W6.b notes Gemma4 likely *is*.
  Applying SWA to a globally-trained model without fine-tuning costs
  quality; SWA is cleanest when the model was trained for it. For
  Qwen3-4B, SWA-at-inference is a *capacity* fallback, not a
  quality-free win.

### 5.3 Attention sinks / StreamingLLM

Add a small fixed set of **sink tokens** (typically the first 4)
that *every* window always attends to, alongside the sliding window.
Empirically this recovers most of the quality SWA loses, because the
softmax "parking" mass that the model dumps on early tokens stays
available. KV resident = `(4 + W)` tokens — essentially still `W`.

- **Structural change:** the band mask gets 4 always-on columns at the
  front; the KV ring buffer pins the first 4 slots and rings only the
  rest. Small, local changes to the mask construction and the runtime
  KV manager. No graph-op change.
- **Quality cost:** much smaller than pure SWA; StreamingLLM's result
  is "near-baseline perplexity to effectively unbounded length" — but
  note it gives *fluency* at unbounded length, **not** *retrieval* of
  content that has scrolled out of `W`. For true 128k *recall* you
  still need the content inside the window.

### 5.4 Paged KV (vLLM-style)

Allocate the KV cache in fixed-size **blocks** (e.g. 256 tokens) and
keep a block table, instead of one monolithic `[ctx]` buffer. Benefits
at long ctx: no need to pre-allocate the full 128k buffer (allocate
blocks as the conversation grows), and blocks of evicted/inactive
sequences can be reclaimed — essential for the multi-session case
(`docs/roadmap.md` B6).

- **Structural change:** this is a **runtime/genie change**, not a
  pathb-graph change. genie's KV manager would need a block table and
  block-indexed gather into the attention op. genie has no such
  facility today; it is the largest runtime engineering item of the
  five levers. The compiled graph still sees a logically-contiguous
  `[ctx, head_dim]` KV tensor — paging is below that abstraction only
  if the HTP attention kernel can take a block table (it currently
  cannot). Realistically paged-KV on HTP means either gather-into-a-
  contiguous-staging-buffer (defeats some of the benefit) or a new
  attention kernel.
- **Quality cost:** none (paging is exact). Cost is pure engineering.

### 5.5 Chunked prefill (ar128+)

The reference bundle ships `ar128` prefill graphs. Long *prompts* (a
128k prompt) cannot be prefilled token-by-token (ar1) in acceptable
time; they must be processed in chunks of 128 (or larger ar). This is
*orthogonal* to decode-side KV scaling — it is about getting the prompt
*into* the KV cache fast. `docs/e2e_optimizations.md` already scopes
ar128 as a follow-on ("Real deployment needs prefill").

- **Structural change:** a second compiled graph variant per ctx tier
  with `seq_q = 128` (the bundle's `ar128` set). `pin_shapes` already
  takes `--seq-q`. For very long prompts the prefill chunk also needs
  a *growing* causal mask per chunk. No new graph ops; another
  pin+compile.
- **Quality cost:** none (prefill chunking is exact). It is a
  latency/throughput feature, mandatory for usable long-prompt UX.

### 5.6 Lever summary

| lever | KV memory effect | graph change | runtime change | quality cost |
|---|---|---|---|---|
| int8 KV | ½ | encoding only | none | ~none (already the recipe) |
| int4 KV | ¼ | encoding only | none | few pts long-ctx recall |
| SWA (W=4k) | **flat in ctx** | band mask, pin `seq_k=W` | KV ring buffer | loses >W recall (unless trained for SWA) |
| + attention sinks | flat | +4 sink columns | pinned-sink ring | small fluency loss; no recall fix |
| paged KV | exact, on-demand alloc | none (or new attn kernel) | **block-table KV mgr** | none |
| chunked prefill (ar128) | n/a (prefill speed) | +ar128 graph variant | chunk scheduler | none |

The decisive one is **SWA + sinks**: it converts "128k" from an
ever-growing-KV problem into a fixed-`W`-resident problem, and as a
bonus collapses the per-ctx compile ladder (§3.2) into a single
`seq_k=W` binary. Everything else is either free (int8 KV, chunked
prefill) or a capacity refinement (int4, paging).

---

## 6. Why Qualcomm caps at 4096

Qualcomm's reference ships `cl{512,1024,2048,3072,4096}` and stops. The
likely reasons, in order of weight:

1. **Per-ctx compile + validation cost, not memory.** §1.3 shows even
   int16 KV at 32k is only 4.5 GiB for 4B — Qualcomm could ship 32k on
   memory grounds. But every ctx tier is a **separately compiled,
   separately validated context binary** (§3.2): 4 parts × N tiers × 2
   ar modes. They already ship **40 binaries**. Doubling the tier
   ladder to cover 8k–128k means doubling the compile matrix, the
   on-device numerical validation, and the bundle QA surface. 4096 is a
   *product scoping* line, not a hardware limit.
2. **It is the trained-window-comfortable region.** 4096 is well
   inside Qwen3-4B's 40960 trained context, so Qualcomm ships tiers
   that need **zero RoPE scaling** — no YaRN, no quality caveat, no
   "long-context accuracy" disclaimer. Shipping 64k/128k would force
   them to ship and validate a RoPE-scaling story (§2) and own its
   accuracy regression.
3. **Per-step latency.** §1.5: a dense ar1 decode step at 128k streams
   the whole KV cache every token. At 4096 the KV-streaming cost is
   ~0.5 GiB/token int16 → sub-millisecond at 228 GB/s and negligible
   vs the weight read; at 128k it is 18 GiB → ~80 ms/token *floor*.
   A 4096 cap keeps every shipped tier in the latency regime where the
   NPU's per-step dispatch cost (the Phase 5 finding,
   `docs/npu_results.md`) dominates and the product feels responsive.
4. **Reference-bundle scope.** The bundle is a *demonstrator* of the
   qai-hub-models recipe, not a production long-context product.
   4096 covers chat / code / RAG-shortlist — the workloads the demo
   targets. Long context is left to whoever needs it (us).

So the 4096 cap is **compile/validation/latency scoping plus
staying inside the trained window** — not a statement that 32k+ is
infeasible. The hardware (44 GB usable) has ample room for 4B KV well
past 4096.

---

## 7. Recommendation + roadmap

### 7.1 Is 128k feasible on a 48 GB device?

- **Qwen3-4B, dense, global attention:** **Yes, on memory** — int8 KV
  at 128k is 9 GiB, ~14 GiB total with weights. **But impractical on
  per-step latency** (§1.5: ~80 ms/token KV-streaming floor) and
  expensive on compile (a 128k tier). 128k *resident* ≠ 128k *usable*.
- **Qwen3-4B with SWA+sinks:** **Yes, cleanly** — KV pinned at a 4k
  window = 0.28 GiB int8, one binary, fast per step. The catch is
  quality: Qwen3-4B was not trained for SWA, so this trades exact
  long-range recall for capacity+speed.
- **Qwen3.6-27B (hybrid):** **Yes** — the SSM layers make 128k KV only
  ~5.5 GiB int16, and w4a16 weights ~15 GiB fit comfortably in 44 GB.
  The blocker is **pipeline support for the hybrid architecture**, not
  memory.

**The single biggest blocker is not the 48 GB budget.** For 4B it is
the **per-step cost and per-ctx compile cost of dense global
attention**; the fix is **SWA + attention sinks**, which caps KV at a
window and collapses the compile ladder. For 27B it is that **the
entire pathb pipeline is dense-Qwen3-only and has no hybrid/SSM
support**.

### 7.2 Concrete structural path: ctx-512 today → 128k

**Stage A — parametric ctx, in-window (32k), no new techniques.**
Everything needed already exists. Build Qwen3-4B bundles at ctx
8k/16k/32k by passing `--ctx` to `pin_shapes_qwen3_4b.py`; `split.py`
is already ctx-parametric; `lib/rope.py` just needs a longer table.
32k is inside the 40960 trained window → no RoPE scaling. Adopt **int8
KV** (already the `qai_hub_recipe.md` P1 recipe). Deliverable: a
ctx-tier ladder (roadmap **B5**) and a loader that picks the smallest
tier ≥ prompt. This validates that the pipeline scales parametrically
and is the honest "we ship 32k" milestone — co-equal with Qualcomm's
4096 but 8× longer, at zero quality risk.

**Stage B — RoPE scaling for 64k/128k (host-side only).** Implement
**YaRN** in `lib/rope.py::build_rope_cache` (add a `rope_scaling`
parameter; the `ModelInfo.rope_scaling` field and the
`FamilyConfig.rope_scaling_supported` gate already exist in
`model_config.py` — flip qwen3 to `True` after the work lands).
**No graph, split, or `.bin` change** — RoPE is hoisted to graph
inputs (§2.2). Compile 64k/128k tiers (still dense KV, int8). Validate
long-context retrieval accuracy (needle-in-haystack) on-device. This is
the cheapest credible 64k/128k *quality* path; YaRN is what the Qwen
team themselves use.

**Stage C — SWA + attention sinks (the structural unlock).** Add a
band+sink mask construction to the pathb mask rewrite and a **ring
buffer KV manager** to the genie/runtime layer. Pin one `seq_k=W`
(W≈4096) binary — this **replaces the per-ctx tier ladder with a single
binary**. Now "128k" is a 4k-resident-KV problem: fast per step, small
memory, one compile. Quality caveat: lossy beyond `W` for a
globally-trained model — acceptable for chat/code, document this. This
is the lever that makes 128k *practical* rather than merely *resident*.

**Stage D — paged KV + chunked prefill (production hardening).**
Chunked `ar128` prefill (the bundle's `ar128` graphs;
`pin_shapes` already has `--seq-q`) so long *prompts* load fast.
Paged KV (block-table KV manager in genie) for multi-session
(`roadmap.md` B6) and on-demand allocation. Both are runtime
engineering, not graph changes.

### 7.3 How it must generalize to Qwen3.6-27B

The 27B target is the reason the pipeline must stay parametric — but
27B is a **hybrid (SSM + attention)**, and the current pathb pipeline
(`rewrite_qwen3_pathb.py`, `split.py`, `model_config.py`) assumes a
**uniform dense decoder block**. The required generalizations:

1. **`model_config.py`** — add a `qwen3_6` `FamilyConfig` and a
   `ModelInfo` that records *which* blocks are attention vs SSM (and
   the SSM state dimensions). The current struct has no field for
   "block type per layer." This is the prerequisite for everything
   else.
2. **`split.py`** — `build_part_specs` currently stamps a
   `past_key_values.{L}` KV tensor for *every* layer. For a hybrid it
   must emit a KV tensor only for **attention** layers and a
   **recurrent-state** tensor (O(1), ctx-independent) for **SSM**
   layers. The cross-part seam logic is unaffected (still the residual
   stream); the per-layer I/O list becomes block-type-aware.
3. **The pathb rewrite** — needs an SSM-mixer variant of the rotary/
   attention hoist, or to leave SSM blocks untouched. The MTP head
   (block 64) is a third block type again.
4. **`genie_config.json`** — `kv-dim` describes one KV geometry; a
   hybrid has KV only on some layers plus SSM state on others. The
   genie KV allocator must be told the *attention-layer count*, not
   the *total layer count*, or it will over-allocate KV ~6× (§1.4).
5. **The good news:** because §1.4 shows the hybrid's 128k KV is only
   ~5.5 GiB, **once the pipeline is hybrid-aware, 27B at 128k is
   *easier* than dense-4B at 128k** — the SSM layers give long context
   for free. SWA (Stage C) is then optional for 27B rather than
   load-bearing. The roadmap's W6 ("graduation") and the
   `docs/2026-05-13_qwen3_6_27b_mtp.md` arch audit are the natural
   home for items 1–4; they should be treated as a **prerequisite for
   any 27B NPU bundle**, long-context or not.

### 7.4 Bottom line

| target | 128k on 48 GB? | gating issue | fix |
|---|---|---|---|
| Qwen3-4B dense, global | resident yes, practical no | per-step KV-stream + per-ctx compile | SWA+sinks (Stage C) |
| Qwen3-4B + SWA+sinks | yes, clean | recall loss >W (not SWA-trained) | accept for chat/code; document |
| Qwen3.6-27B hybrid | yes (~22 GiB total) | pipeline has no hybrid support | hybrid-aware split/config (§7.3) |

Recommended order: **Stage A (32k, ships now, zero risk) →
Stage B (YaRN for 64k/128k) → Stage C (SWA+sinks, the unlock) →
Stage D (paging/prefill hardening)**, with the **hybrid-pipeline work
(§7.3) front-loaded** because it gates the 27B target the whole project
is aimed at — and because, once done, it makes 27B the *easiest* long-
context case rather than the hardest.
