# NPU long-context research brief — exotic execution strategies

Companion to `docs/2026-05-21_specula_bundle_npu_testing.md`. That doc
reports what broke; this one is the research-grounded brainstorm for
what to try next. Prompted by: "can we do something exotic — rolling
loaded partitions, a custom Zig harness, on-the-fly quant? search
arxiv for what others (incl. Qualcomm) have tried."

**Read §0 first** — it pins the brainstorm to the real research goal
(Qwen3.6-27B @ 132k) and says which of the §1–§6 ideas that target
actually needs. §1 onward analyzes the ctx32768 4B bundle, which
remains a useful fallback toolbox / proving ground.

Hardware we're playing with: Snapdragon X2 Elite Extreme — **48 GB
LPDDR5X unified memory @ 228 GB/s**, shared across 12 Oryon CPU cores,
Adreno X2 GPU, Hexagon NPU. Three compute islands, one memory pool,
no copy cost to hand data between them. That last fact is the lever.

---

## 0. The actual target: Qwen3.6-27B @ 132k context

The research goal is not the 4B/32k bundle — it is **Qwen3.6-27B at
132k context** on this NPU. That target changes which ideas below
matter, so anchor on it first.

### 0.1 Qwen3.6-27B is a hybrid model, built for long context

Qwen3.6-27B is **dense 27B, 64 layers = 16 blocks of [3× Gated
DeltaNet + 1× Gated Attention]** ([Qwen blog][qwen36]). Only **1 layer
in 4 has a quadratic KV cache**, and those use just **4 KV heads** at
128 dim. The other 48 layers are **Gated DeltaNet — linear attention,
O(n)**, with a fixed-size recurrent state that does **not** grow with
context.

Recompute the 132k KV with the real architecture:

- 16 attention layers × 4 KV heads × 128 dim × 2 (K,V) = 16,384
  elem/token → ×132,000 = **~8.6 GB FP32 / ~2.2 GB int8 / ~1.1 GB int4**.
- DeltaNet recurrent state: **context-independent**, tens of MB total.

A naïve dense transformer at 27B/132k would need ~70 GB of FP32 KV —
impossible in 48 GB. **Qwen already solved that.** 132k context is
*not* the wall, and most of the "exotic KV offload / rolling to fit"
machinery in §3–§5 is **moot for this target**.

### 0.2 So the wall moves — to dense-27B weight bandwidth

Decode on a dense 27B reads **~14 GB of w4 weights every token** →
bandwidth-bound at **~4–8 t/s** on this NPU regardless of host-side
cleverness. KV no longer dominates; weight traffic does. That reframes
the whole menu (§5):

**Mandatory — won't run / won't be usable otherwise:**

1. **w4 weights** (~14 GB; fits with room). w8 (~27 GB) also fits now
   that KV is small, but w4 wins on decode bandwidth.
2. **Linked / weight-shared context binaries** (idea 0) — **still
   required, and orthogonal to the hybrid architecture.** A 27B w4 is
   ~14 GB, far over the HTP ~3.67 GB single-`.bin` serializer ceiling,
   so it *must* split into multiple parts. Those parts must be linked
   into one (or ≤5) weight-shared context(s) or they hit the same
   QNN-1002 session ceiling the 19-part 4B bundle did. Qwen3.6's
   hybrid attention fixes KV *size*, not context *count* — it does
   nothing for this. Note this *replaces* the need for rolling
   partitions (§3.1): link properly and all parts co-reside, so
   there is nothing to roll.
3. **AR128 batched-prefill graph** (idea 2) — prefilling 132k tokens
   AR1 is *hours*.
4. **DeltaNet linear-attention pipeline support — NEW, not elsewhere
   in this brief, and the biggest unknown.** The pathb pipeline
   (`rewrite_qwen3_htp` / `rewrite_qwen3_pathb`) is Qwen3
   *standard-attention* specific. Qwen3.6-27B's 48 Gated DeltaNet
   layers are a recurrence; they must be emitted in **chunkwise-
   parallel form** (matmuls) or the HTP chokes on the sequential scan.
   This is make-or-break and must be validated early (§0.4).

**The multiplier — how it becomes usable, not just runnable:**

5. **Speculative decoding** (idea 6) — the only way past the ~4–8 t/s
   weight-bandwidth wall, and the target ships its own draft:
   **`Qwen3.6-27B-MTP`** has a multi-token-prediction head, i.e. a
   built-in self-speculative draft. Session 27 already measured that
   MTP head at **+45–56 % TG** (`current_status.md`). Lean on it;
   optionally stack a separate 4B draft for more acceptance.

**Not on the critical path for this target:**

- ❌ Per-token **rolling partitions** (§3.1 / idea 3) — linking
  contexts (idea 0) solves the fit; rolling was a workaround for the
  broken 19-context split, not needed once parts share a context.
- ❌ Aggressive **KV offload / disk swap** (§4 long-context papers) —
  KV is ~2 GB int8, fits in RAM trivially.
- ❌ **On-the-fly weight quant** (§3.3) — quantize once in the pipeline.
- 🔸 int8 KV + `QnnMem` in-place KV (idea 4): good harness hygiene,
  no longer load-bearing.
- 🔸 Heterogeneous islands (idea 5): helps the 132k prefill, optional
  for decode.

### 0.3 The combination, in one line

**w4-quantized + DeltaNet-aware pipeline + linked weight-shared
contexts + AR128 prefill graph + MTP-based speculative decode.**
Items 1–3 are correct build settings; item 4 is the real research
risk; item 5 is what makes it usable. The exotic KV/rolling work in
§3–§5 is *not* required for Qwen3.6-27B @ 132k — the model's hybrid
linear/quadratic architecture already did that job.

### 0.4 Derisk these first, cheaply (before any 27B on-device work)

1. **Does Gated DeltaNet survive export → pathb rewrite → QNN at
   all?** Try one DeltaNet block through `optimum` export +
   `qairt-converter`. If the recurrence won't lower, nothing else
   matters.
2. **Does the HTP run DeltaNet's chunkwise form at matmul speed**, or
   does it fall back to a slow scan? Microbench one block.
3. **MTP-head acceptance rate at long context** — measure offline on
   the cloud GPU before porting (cf. SpecPV: self-spec degrades if the
   draft isn't long-context-aware).

Only if 1–2 pass does the 27B/132k target reduce to "known-tractable
engineering" (items 1–3 above). The rest of this brief stays relevant
as the fallback toolbox if the hybrid arch fights the NPU.

---

## 1. The problem, in three facts (the 4B/32k bundle)

1. **HTP session ceiling.** The ctx32768 bundle is 19 separate QNN
   *context binaries*. ORT-QNN loads ~4 then fails part 5 with QNN
   error 1002 — HTP context memory exhausted. ~4–5 co-resident
   contexts is the hard ceiling. Not a weight-size problem (the 19
   `.bin`s total only 3.9 GB); it's HTP per-context working-set
   (VTCM/TCM reservation + DDR scratch).
2. **Context load is slow.** Creating one HTP context from a prebuilt
   `.bin` measured 1.6–7 s. Per-token partition swapping is therefore
   a non-starter for decode (~15 swaps/token ⇒ tens of seconds/token).
3. **FP32 KV is the decode bottleneck.** Per `..._npu_testing.md` §5,
   ~165 of ~175 ms/decode-step is on-device, dominated by moving FP32
   KV (~100 MB per 12-layer part). At 32k context the KV is ~9.6 GB.

---

## 2. The fork: prebuilt bundle vs pipeline rebuild

Be honest about which lever each idea pulls.

- **Prebuilt bundle as-is.** The compiled context binary's graph I/O
  is frozen — FP32 KV, FP32 seams, AR1-only, 19 contexts. Host-side
  cleverness (scheduling, swapping, heterogeneous placement) is all
  that's available. Real, but bounded.
- **Pipeline rebuild on RunPod.** The split count, KV dtype, I/O
  dtype, AR128 graph, and **whether parts share one context** are all
  pipeline parameters. The single highest-value fix lives here (§3.0).

Most "exotic" ideas below are host-side and work on the bundle as-is.
The one that isn't exotic at all — and is the actual fix — is a
pipeline parameter. Lead with that so we don't out-clever ourselves.

---

## 3. Triage of the three proposed ideas

### 3.0 (not proposed, but first) — link the parts into ONE context

Qualcomm's *own* shipping Qwen3-4B bundle is 4 `.bin`s with **10
graphs each, weights shared** (5 ctx tiers × ar1/ar128). They do not
hit a session ceiling because graphs inside one context share the
HTP context allocation. QNN calls this a **Link job** —
"combine multiple models into a single context binary so weights can
be shared between graphs … exclusive to QNN context binaries for the
HTP" ([AI Hub docs][aihub-link]).

Our pathb pipeline emitted **19 independent contexts**. That is the
bug. `qnn-context-binary-generator` can weight-share graphs into one
context; `compile_split_bundle.py` runs it per-part instead. Fix:
re-split into ≤5 parts **or** link all parts into one weight-shared
context with N graphs. Then ctx32768 loads like the ctx512 bundle and
none of §3.1–3.3 is needed. **Do this first.** Everything below is for
"what if we want to be clever anyway / the rebuild isn't ready."

### 3.1 Rolling loaded partitions — *partly viable, with a twist*

Naive per-token rolling (load window, run, evict, repeat, every
token): **dead** — fact 2, ~15 swaps/token.

But reorder the computation and it works for **prefill**:

> Keep all 19 `.bin`s mmap'd in DDR (3.9 GB — trivial in 48 GB). Run
> the whole prompt **breadth-first per partition-window**: push *all*
> prefill chunks through resident parts 1–4, stash their output
> activations + KV in DDR, swap to parts 5–8, push all chunks again,
> … Total context swaps for a full 32k prefill ≈ 5, not 5 per chunk.

Activation stash cost: 32k tokens × 2560 hidden × 4 B ≈ 335 MB per
seam — nothing in 48 GB. This turns "19 contexts won't fit" into "5
sequential context-loads", ~10 s of swap overhead amortized over the
*entire* prefill. This is classic **pipeline-parallel / activation-
stashing** execution, and prefill is exactly where a 32k-context model
spends its time.

Decode is the hard part — autoregressive, 1 token needs all 19 parts.
Rolling can't hide swap cost there… **unless** decode is batched, which
is what §6 (self-speculative decoding) buys us. So: rolling works for
prefill today; rolling works for decode *if* paired with speculation.

Prior art: Apple's **LLM-in-a-flash** streams weights from storage on
demand with windowing + double-buffering ([arXiv 2312.11514][flash]);
**Active-weight swapping** runs models 2× DRAM by swapping hot weights
DRAM↔flash ([arXiv 2504.08378][aws]); **Demand Layering** keeps only
~1–2 layers resident via on-demand load pipelined against compute.
Our version is the same idea one level up — at the HTP-*context*
granularity, and we have it easy because the working set fits in RAM;
we're swapping *HTP residency*, not *memory residency*.

### 3.2 Custom Zig harness — *yes, but the win is "direct QNN C API"*

The value isn't Zig; it's **dropping ORT-QNN and calling `libQnnHtp`
directly**. ORT's QNN EP hides exactly the controls we need:

- **Explicit context lifecycle** — `QnnContext_createFromBinary` /
  `_free` on our schedule, for the §3.1 rolling pipeline. ORT ties
  context lifetime to `InferenceSession`.
- **`QnnMem` shared buffers** — register one DDR buffer the HTP reads
  *in place*. The decode KV could live in a single persistent buffer;
  each step writes only the new slot. Kills the per-step FP32-KV
  marshalling that IOBinding could not (`..._npu_testing.md` §5).
- **Multiple graphs per context** — the §3.0 fix, driven from our own
  runtime.
- **Async execute + multi-HTP-core**, and QNN's native profiler.

Zig is a *fine* language for it — first-class C interop (`@cImport`
straight onto the QNN headers), manual allocators (ideal for a fixed
ring of context slots), `comptime` to generate the per-part I/O
binding code from `bin_info/part_*.json`, clean ARM64-Windows cross-
compile. C or Rust would do equally; pick Zig if the team enjoys it.
This becomes "`npu_engine` v2" — a real engine, not an ORT wrapper.
Note `npu_engine/sidecar.py` already prototypes the long-lived-process
idea; v2 is that, in Zig, on the QNN C API.

### 3.3 On-the-fly quantization — *limited as-is; real only post-rebuild*

The compiled graph's I/O dtypes are frozen. You cannot feed it int8 KV
— it demands FP32. So "quantize on the fly to make pieces fit" does
**not** speed up the existing bundle's NPU compute.

Two honest sub-cases:
- *Host-side KV in int8, dequant-on-feed.* Cuts host KV RAM 4× (9.6 GB
  → 2.4 GB at 32k). The NPU still gets FP32, so **no compute speedup**.
  Marginal for us — 48 GB makes capacity a non-issue. Skip.
- *Quantized KV / I/O in the graph.* This is the real 4× win
  (`..._npu_testing.md` §5/§6) but it needs the bundle **recompiled**
  with int8 KV tensors — a pipeline change, not on-the-fly.

Where on-the-fly quant *is* genuine: a **mixed-precision draft**. Run
a 4-bit (or layer-skipped) version of the model as the speculative
draft and the full bundle as verifier — see §6. The "quantize to make
it fit" instinct is right; it just belongs in the draft model, not in
re-feeding a frozen graph.

---

## 4. What the literature is doing (grouped)

**NPU prefill restructuring.** `llm.npu` / mllm-NPU (ASPLOS'25, *on
Hexagon*) is the closest prior art ([arXiv 2407.05858][mllmnpu]):
chunked prefill (chunk = 256 tokens), and **chunk-sharing graphs** —
split ops into static (Linear/LayerNorm, prompt-size-independent) vs
dynamic (Attention), share the static subgraphs across chunks: "120 of
144 subgraphs" shared, **up to 4× / 75 % memory reduction**. Also:
extract activation outliers to run on CPU/GPU in parallel, and
**out-of-order block scheduling** across NPU/CPU/GPU by hardware
affinity. 22.4× faster prefill vs baselines.

**Heterogeneous CPU+GPU+NPU on unified memory.** HeteroInfer (GPU-NPU,
1.34× over single-engine, tensor partition + fast unified-memory sync);
Agent.xpu (per-op kernel callbacks across XPUs); APEX (async CPU/GPU
overlap). Key measured fact: **one engine saturates only ~40–45 GB/s
of memory bandwidth; two concurrent engines reach ~60 GB/s**
([arXiv 2501.14794][hetero], [2506.24045][agent], [2506.03296][apex]).
On a 228 GB/s part with three idle islands, single-island execution is
leaving bandwidth on the table.

**Long-context / KV.** KVSwap (disk-aware KV offload), KVNAND (KV in
flash), AccLLM, "Packing-Prefetch scheduler + large on-chip memory",
context parallelism for million-token inference, and **SpecPV** —
self-speculative decoding *specifically for long-context* via partial
verification ([2511.11907][kvswap], [2505.03745][accllm],
[2411.01783][ctxpar], [2512.02337][specpv]). For us KV *capacity* is
fine (48 GB); KV *bandwidth/dtype* is the issue.

**Self-speculative decoding (the on-brand one).** LayerSkip (Meta) —
"exits at early layers, verifies and corrects with remaining layers …
less memory footprint … shared compute and activations of draft and
verification" ([arXiv 2404.16710][layerskip]). CLaSp — skip
intermediate layers as the draft ([2505.24196][clasp]). ConfLayers,
PPSD (pipeline-parallel self-spec). 1.4–3.8× speedups, **no second
model**.

---

## 5. Ranked menu of ideas

| # | idea | lever | effort | payoff |
|--:|---|---|---|---|
| 0 | Re-split ≤5 parts / **link into 1 weight-shared context** | rebuild | low | unblocks ctx32768 entirely |
| 1 | Recompile with **int8 KV + quantized seams** | rebuild | med | ~4× decode (the real fix) |
| 2 | Emit an **AR128 prefill graph** | rebuild | med | ~100× PP |
| 3 | **Breadth-first rolling prefill** (§3.1) | host | med | makes 19-part prefill viable as-is |
| 4 | **Direct-QNN-C-API engine** + `QnnMem` in-place KV (§3.2) | host | high | kills per-step KV marshalling; enables 3 & 6 |
| 5 | **Heterogeneous layer split** — FFN on Hexagon, attention on Adreno | host | high | uses 2nd island; ~60 vs 45 GB/s |
| 6 | **Self-speculative decode** — early-exit draft, full bundle verifies | both | high | makes rolling viable for *decode*; on-brand |

0–2 are "correct, do them". 3–6 are the research.

---

## 6. The headline: self-speculative decoding *is* the unlock

This is a **speculative-decoding research repo**. The 19-partition
problem and speculative decoding fit together exactly:

- Autoregressive decode's curse is "every token needs all 19 parts" —
  which forces all-resident or fatal per-token swapping.
- Speculative decoding changes the unit of work from *one token* to
  *one accepted chunk of k tokens*. The full 19-part stack runs once
  per **verification**, in a **batched** AR-k pass — and a batched
  pass amortizes a partition-roll over k tokens. Rolling partitions,
  dead for token-by-token decode, becomes viable for *batched
  verification*.
- The **draft** is the model's own early layers (LayerSkip / CLaSp
  style) — only the first few HTP contexts, which **do fit** under the
  ceiling. Or the draft is the existing **ctx512** bundle, or a
  GGUF on Adreno/CPU (the repo already runs those).
- SpecPV shows self-spec + partial verification is *built for
  long-context* generation specifically.

So the crazy-but-grounded plan: a fast draft that fits the HTP
(early-exit head, or ctx512 bundle, or GPU draft) proposes k tokens;
the full ctx32768 stack verifies them in one batched roll across the
partition windows; acceptance rate sets the speedup. The 19-partition
cost is paid once per accepted chunk, not once per token — and that is
the only thing that makes a 19-context model decode-viable without a
rebuild. It also happens to be the project's whole thesis.

---

## 7. Concrete next experiments (cheap → ambitious)

1. **Re-split ctx32768 to 4–5 parts** on RunPod (`--num-parts`), or
   link the 19 into one weight-shared context. Re-run
   `bench_pathb_ortqnn.py`. Likely just *works*. (idea 0)
2. **Breadth-first prefill probe** — extend the harness: mmap all 19
   `.bin`s, load a 4-context window, prefill a long prompt window-by-
   window stashing seam activations in DDR. Measures whether ~5
   context-loads amortize. (idea 3)
3. **`QnnMem` KV spike** — minimal C/Zig program: one QNN context,
   one persistent KV buffer registered via `QnnMem`, time a decode
   step vs the ORT-QNN 165 ms. Proves out idea 4 before committing to
   the engine.
4. **Self-spec feasibility** — does a Qwen3-4B early-exit at layer ~18
   produce a usable draft distribution? Measure draft/verify agreement
   offline (cloud GPU) before any on-device work. (idea 6)
5. **Layer-island split** — prototype attention-on-Adreno /
   FFN-on-Hexagon for one layer; measure the unified-memory handoff
   cost. (idea 5)

Experiments 2–3 are the ones that need new on-device tooling and are
the natural seed for the Zig/C `npu_engine` v2.

---

## Sources

- [llm.npu / mllm-NPU — Fast On-device LLM Inference with NPUs (Hexagon), ASPLOS'25][mllmnpu]
- [LayerSkip — Early Exit Inference and Self-Speculative Decoding][layerskip]
- [CLaSp — In-Context Layer Skip for Self-Speculative Decoding][clasp]
- [SpecPV — Self-Speculative Decoding for Long-Context Generation][specpv]
- [LLM in a Flash — Inference with Limited Memory][flash]
- [Active-Weight Swapping Between DRAM and Flash][aws]
- [Characterizing Mobile SoC for Heterogeneous LLM Inference (HeteroInfer)][hetero]
- [Agent.xpu — Agentic LLM Workloads on Heterogeneous SoC][agent]
- [APEX — Asynchronous Parallel CPU-GPU Execution][apex]
- [KVSwap — Disk-aware KV Cache Offloading][kvswap]
- [AccLLM — Long-Context LLM Inference via Algorithm-Hardware Co-Design][accllm]
- [Context Parallelism for Scalable Million-Token Inference][ctxpar]
- [Qualcomm AI Hub — Linking / weight-shared context binaries][aihub-link]
- [Qwen3.6-27B — hybrid Gated DeltaNet + Gated Attention architecture][qwen36]

[mllmnpu]: https://arxiv.org/abs/2407.05858
[layerskip]: https://arxiv.org/abs/2404.16710
[clasp]: https://arxiv.org/pdf/2505.24196
[specpv]: https://arxiv.org/html/2512.02337v1
[flash]: https://arxiv.org/html/2312.11514v2
[aws]: https://arxiv.org/pdf/2504.08378
[hetero]: https://arxiv.org/abs/2501.14794
[agent]: https://arxiv.org/html/2506.24045v1/
[apex]: https://arxiv.org/pdf/2506.03296
[kvswap]: https://arxiv.org/abs/2511.11907
[accllm]: https://arxiv.org/pdf/2505.03745
[ctxpar]: https://arxiv.org/pdf/2411.01783
[aihub-link]: https://app.aihub.qualcomm.com/docs/hub/link_examples.html
[qwen36]: https://qwen.ai/blog?id=qwen3.6-27b
