# Qwen3.6-27B NPU bundle — kickoff brief (RunPod team)

Created: 2026-05-22. Workstream charter. Companion to
`docs/long_context_scaling.md` §1.4/§7.3/§8.8 (the analysis this
decision rests on), `docs/2026-05-13_qwen3_6_27b_mtp.md` (the CPU/GPU
arch audit), and `current_status.md` (session 31).

## The decision

**Stop trying to push dense Qwen3-4B past 4k on the NPU. Pivot the
RunPod conversion pipeline directly to Qwen3.6-27B as the long-context
target, then Qwen3.6-35B-A3B.** It will be messy — the pathb pipeline
is dense-Qwen3-only today and has no concept of an SSM mixer — but the
mess is on the real target instead of on scaffold we'd throw away.

### Why (the short version)

Session 31 (`long_context_scaling.md` §8.8) hit a hard, non-negotiable
wall: a per-decoder-layer KV `InputSlice` of `[1, 8, ctx, 128]` cannot
tile into the HTP's ~8 MiB VTCM once `ctx` passes ~4–8k. It is a
per-op HTP limit, not a build flag — it is *why Qualcomm caps their
reference at cl4096*. Dense 32k/64k global attention is **not
HTP-compilable** with the current graph.

The only architectural fix for a dense model is sliding-window
attention — but Qwen3-4B was **not trained with SWA** (`sliding_window:
null`), so SWA on it is a quality-lossy capacity hack, and the geometry
shows the wall is *identical* on the real target anyway (see table
below). Building SWA on Qwen3-4B means building a degraded feature we
then discard. A 4k context ceiling is not useful for the work this
project targets — so we go straight to the model that makes long
context structurally cheap.

## What Qwen3.6-27B actually is (measured from the GGUF)

Pulled `2026-05-22` from `models/Qwen3.6-27B-MTP-Q4_0.gguf` metadata
(`gguf_dump.py`). Architecture id: `qwen35`.

| key | value |
|---|---|
| `block_count` | 65 — 64 transformer blocks + block 64 = MTP head |
| `context_length` | **262144** — natively trained to 256k, no RoPE scaling needed |
| `full_attention_interval` | **4** — every 4th block is full attention |
| → attention layers | **~16** (blocks 3, 7, 11, … 63); the other ~48 are SSM/Mamba2 |
| `attention.head_count` / `head_count_kv` | 24 / **4** (GQA) |
| `attention.key_length` / `value_length` | **256** / 256 |
| `ssm.{conv_kernel,state_size,group_count,time_step_rank,inner_size}` | 4 / 128 / 16 / 48 / 6144 |
| `embedding_length` / `feed_forward_length` | 5120 / 17408 |
| `rope.freq_base` / `dimension_count` | 1e7 / 64 |
| `nextn_predict_layers` | 1 (the MTP head, block 64) |

Two facts drive everything:

1. **The ~48 SSM/Mamba2 layers carry an O(1) recurrent state — no KV
   cache, no `ctx`-scaling `InputSlice`, no TCM wall.** They scale to
   256k for free. This is why 27B long-context is *memory-cheap*: KV
   lives on only ~16 layers. Total KV ≈ 32 KiB/token int8 → ~4 GiB at
   128k, ~8 GiB at 256k — trivial on a 48 GB device.
2. **The ~16 full-attention layers still hit the exact same TCM
   wall.** Per-layer KV slice = `4 kv_heads × 256 head_dim` = 1024
   elem/token — *identical* to Qwen3-4B's `8 × 128`. At ctx-32768
   uint8 that is still a 32 MiB slice that will not tile. There is no
   native `sliding_window` key — these layers are global attention.

**So SWA is still required — but only on ~16 layers, and it is no
longer a quality hack.** In this hybrid the SSM layers *are* the
long-range path; attention is for local precision. Windowing the
attention layers is "with the grain" of the architecture, not against
it. That is the whole reason the pivot is worth it.

## Known pipeline snags (expect more)

The pathb pipeline (`end-to-end/lib/`, `scripts/rewrite_qwen3_pathb.py`)
assumes a uniform dense decoder block. Hybrid-awareness is the work.
In rough dependency order:

1. **`lib/model_config.py`** — add a `qwen3_6` `FamilyConfig` (the
   `FAMILY_CONFIGS` TODO already flags this) **and** extend `ModelInfo`
   with a per-block type map (attention / SSM / MTP) + the SSM state
   dims. Nothing downstream can branch correctly without this. The
   `full_attention_interval = 4` rule generates the map.
2. **`scripts/rewrite_qwen3_pathb.py`** (the rotary hoist, called from
   `lib/stages.py::stage_pathb_chain`) — must skip SSM blocks (no
   rotary to hoist) and handle the MTP head as a third block type. The
   hoist currently asserts identity attention-scaling; re-check
   against `qwen35`.
3. **`lib/split.py`** — `build_part_specs` stamps a `past_key_values.{L}`
   KV tensor for *every* layer. For the hybrid it must emit KV I/O
   only for the ~16 attention layers and an O(1) recurrent-state
   tensor for SSM layers. Cross-part seam logic (the residual stream)
   is unaffected.
4. **SWA on the 16 attention layers** — band mask + KV input pinned to
   window `W` (not `ctx`); pairs with the `lib/aimet.py::_apply_uint8_kv`
   uint8-KV path and `lib/qairt.py` `preserve_io`. With `W ≈ 4096` the
   per-layer slice is 4 MiB → tiles. Attention sinks (first 4 tokens)
   are a cheap follow-on.
5. **Runtime / genie KV manager** — must be told the *attention-layer
   count*, not the *total* layer count, or it over-allocates KV ~4×;
   plus a ring-buffer KV buffer for the windowed layers and SSM-state
   handling. `npu_engine/bench_pathb_ortqnn.py` is our engine, not
   genie — scope the KV manager there.
6. **MTP head (block 64)** — out of scope for the first bundle; decode
   it as a plain attention block or drop it. Self-draft via the MTP
   head is a later optimization (`docs/2026-05-13_qwen3_6_27b_mtp.md`).

## First milestone for the RunPod team

Goal: **the first loadable, on-device-correct Qwen3.6-27B NPU bundle
at a real context (target ctx 32768)** — not fast, not optimal, just
loadable and coherent. Suggested path:

1. Land snags 1–3 (hybrid-aware `model_config` / rewrite / split).
   Verify on a *short* ctx (512) first — a plain hybrid pathb export
   that compiles and decodes coherent text proves the topology before
   touching SWA.
2. Add SWA (snag 4) on the 16 attention layers at `W = 4096`; rebuild
   at ctx 32768; split into ≤8 parts; load via
   `npu_engine/bench_pathb_ortqnn.py`.
3. On-device sanity check: coherent decode at ctx 32768, logit cos vs
   the CPU/GGUF reference. Then sweep ctx → 65536 / 131072 / 262144.

Commit at each landed snag; append findings to `long_context_scaling.md`
§8 and bump `current_status.md`. Expect undocumented snags — log them.

## Out of scope (deferred, do not block on)

- Qwen3.6-35B-A3B (MoE) — same hybrid pipeline plus MoE expert
  dispatch; starts once 27B is loadable.
- MTP self-draft speculative decode.
- Perf optimization (ar128 prefill, throughput tuning) — correctness
  and loadability first.
- Genie loadability (`attention_mask`-as-input) — we use our own
  ORT-QNN engine; dropped per `long_context_scaling.md` §8.7.

## References

- `docs/long_context_scaling.md` §1.4 (hybrid KV math), §7.3 (the
  hybrid-pipeline generalization), §8.8 (the TCM wall)
- `docs/2026-05-13_qwen3_6_27b_mtp.md` — `qwen35` arch audit, CPU/GPU
  MTP numbers
- `current_status.md` — session 31 checkpoint
- `end-to-end/lib/{model_config,split,stages,aimet,qairt}.py`,
  `scripts/rewrite_qwen3_pathb.py` — the files that change

## Update 2026-06-15 — snag 1 landed + real config.json findings

**Snag 1 done** (`lib/model_config.py`, commits `9899603` + real-config
pin). `qwen3_6` FamilyConfig + `ModelInfo` now emit the per-block map
(`block_types` / `attention_layer_indices` / `num_attention_layers`),
MTP count, partial-rotary, mrope params, and linear-attention dims.
Dense 4B path provably unchanged. Verified against the on-disk
`Qwen/Qwen3.6-27B` config.json.

Pulled the **real HF config.json** (was GGUF-only before). It corrects
several assumptions the original charter (GGUF-derived) couldn't see:

1. **It is a vision-language model** — `architectures:
   [Qwen3_5ForConditionalGeneration]`, `model_type: qwen3_5`, with a
   `vision_config` and the LLM dims nested under `text_config`. For the
   NPU bundle we **target `text_config` and drop the vision tower**
   (`language_model_only`-style export). `model_config` reads
   `text_config` when present.
2. **`layer_types` is enumerated in-config** — 16 `full_attention` at
   blocks [3,7,…,63], 48 `linear_attention` (gated-delta-net). We read
   the list rather than deriving from `full_attention_interval`.
3. **Snag 2 is bigger than "skip SSM blocks".** The attention layers
   use **partial rotary** (`partial_rotary_factor: 0.25` → only 64 of
   256 head dims get RoPE), **mRoPE** (`mrope_interleaved: true`,
   `mrope_section: [11,11,10]`), and a **gated attention output**
   (`attn_output_gate: true`, `output_gate_type: swish`). The dense
   pathb hoist assumes full rotary on a 128-dim head, identity
   attention-scaling, and no mrope/gate. The rotary hoist must be
   re-derived for partial+mrope, or this layer family punted to AI Hub.
4. **Gated-delta-net dims** (real keys): `linear_conv_kernel_dim=4`,
   `linear_{key,value}_head_dim=128`, `linear_num_key_heads=16`,
   `linear_num_value_heads=48` (inner=6144). These drive snag 3's SSM
   recurrent-state I/O.
5. New tokenizer: `vocab_size=248320`, bos/eos=248044 (matches the
   [[reference_qwen_tokenizer_generations]] incompatibility note).

**Strategy (per user):** do as much as possible on the X2E (export /
rewrite / split / qairt-convert+quantize all run on-device via Prism
x86), and punt the physically-impossible-on-device pieces (the AIMET
GPU calibration, and likely the partial+mrope+gated attention export)
to **Qualcomm AI Hub** (free, API key on device). The non-standard
attention (snag 2) and the gated-delta-net ops are the prime AI Hub
candidates — verify what AI Hub's compile/quantize jobs accept for the
`qwen3_5` arch before hand-rolling the ONNX surgery.
