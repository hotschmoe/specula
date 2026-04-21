# Phase 5 step 6 — x86 export report (two-artifact cycle)

**Produced:** 2026-04-21 on x86_64 Windows 11 Pro (Intel B50 Pro host,
no CUDA). Python 3.12.13 in `.venv-x86-export/`. Operator session log:
`status_x86.md` at repo root.

**Supersedes:** the session 7-9 `qwen3-0.6b-nomask/` pipeline
(quarantined per `docs/phase5_export_on_x86.md` §DEPRECATED — compile
succeeds, decode produces anti-correlated output).

## TL;DR for the aarch64 operator

Two rewritten Qwen3-0.6B ONNX variants land at
`Z:\exposed\junk\` for you to pick up. Both are CPU-ORT-verified
**cos = 1.0 and max_abs_diff = 0.0** against the optimum source
(not merely "close" — bit-identical on the probe regime). Both are
HTP-compile candidates for Phase 5 step 4 (retry).

| Artifact | Rewrite strategy | Cast→BOOL | Range | BOOL tensors | IO change from session 9 |
|---|---|---:|---:|---|---|
| `qwen3-0.6b-patha/` | `ConstantOfShape(..., BOOL)` replaces `Gather_5` + drop 2 BOOL-identity casts | 0 | 3 | many (LessOrEqual, And, Expand, 28×Slice_4, 28×Where_2) | `attention_mask` removed from inputs (promoted to initializer) |
| `qwen3-0.6b-pathbmask/` | Splice new FP32 `attention_bias` input directly into 28 `Add_2`; delete entire BOOL mask subgraph | 0 | **0** | **0** | `attention_mask` removed; `attention_bias` added as FP32 input |

**Recommendation:** compile both through AI Hub. The two compile
outcomes form a 2×2 table that tells us how strict HTP's BOOL
rejection really is — that's information we've never had.

## What was built, in one paragraph

`optimum.exporters.onnx --no-post-process --dtype fp16` produced a
clean graph (`qwen3-0.6b-optimum/`) with zero `com.microsoft` ops —
that part of the phase5 plan is solid. A single Python script
(`scripts/rewrite_qwen3_htp.py --mode {stage,fold-patha,fold-pathbmask}`)
then applies three transforms: (stage) the two safe rewrites from
doc §2a+2b (attention_mask to initializer, IsNaN/Where-guard elision),
then the two forked folds. Every stage is CPU-ORT-probed against the
optimum source via `scripts/probe_cos_vs_source.py`; all three probes
returned cos=1.0, argmax match, top-5 full overlap.

## Artifact detail

### `qwen3-0.6b-patha/`

Strategy: doc's Path A taken to its literal end.

Specific edits relative to the staged graph (7,611 nodes post 2a+2b):

1. `/model/Cast_2` (INT64 → BOOL) folded away. Its only consumers
   were `/model/Flatten` (dropped as Gather_5 collapses) and
   `/model/Shape_4` (rewired to read `attention_mask` directly —
   `Shape` doesn't care about dtype).
2. `/model/Flatten` and `/model/Gather_5` replaced with a two-node
   sequence: `Shape(Gather_5.indices)` → `ConstantOfShape(shape,
   value=True, dtype=BOOL)`. Output tensor name preserved
   (`/model/Gather_5_output_0`) so downstream needs no rewiring.
3. `/model/Cast_5` (BOOL → BOOL identity) removed; consumers read
   `/model/LessOrEqual_output_0` directly.
4. `/model/Cast_6` (BOOL → BOOL identity) removed; consumers read
   `/model/Reshape_1_output_0` directly.

Post-edit stats: 7,580 nodes, 0 Cast→BOOL, 0 com.microsoft, 0 IsNaN,
3 Range (kept — causal-mask builder still uses it).

**What HTP still sees:** BOOL tensors flowing through the graph via
`LessOrEqual → And → And_1 → Expand → 28×Slice_4 → 28×Where_2`. The
Cast-to-BOOL ops are gone but the BOOL data itself persists.

**Hypothesis tested:** HTP rejects `Cast(..., to=BOOL)` specifically,
not BOOL tensors in general. Compile result directly answers this —
if qai-hub's OPTIMIZING_MODEL phase accepts, the hypothesis is
supported and Path A is cheap; if it rejects on a BOOL-typed op,
we know the rejection is on BOOL tensors and Path B-mask is the way.

### `qwen3-0.6b-pathbmask/`

Strategy: the mask half of doc's Path B. Skip RoPE externalization
(deferred to a future cycle if this doesn't ship).

Specific edits:

1. New graph input added: `attention_bias` dtype FLOAT (FP32, matching
   `Where_2`'s output dtype — optimum's fp16 export keeps attention
   *math* in FP32 even with FP16 weights), shape
   `[batch_size, 1, seq_q, seq_k]` (fully dynamic on q/k sequence
   dims so compile-time shape-pinning can choose any).
2. For each of the 28 `/model/layers.N/self_attn/Add_2` nodes, input
   index 1 (`.../Where_2_output_0`) rewired to `attention_bias`.
3. Dead-code elimination: 445 now-orphaned nodes removed — the
   entire BOOL mask subgraph (Cast_2, Flatten, Gather_5, Reshape,
   Reshape_1, Cast_5, Cast_6, And, And_1, Expand, 28×Slice_4,
   28×Where_2, 28×mask-scalar constants, LessOrEqual, Range_2, the
   position-math Unsqueezes/Ranges that fed LessOrEqual, and the
   `ConstantOfShape` that fed the And's BOOL initializer).
4. `attention_mask` initializer itself pruned (no remaining consumer).

Post-edit stats: 7,166 nodes, **0 Cast→BOOL, 0 Range, 0 LessOrEqual,
0 IsNaN, 0 And, 0 BOOL tensors anywhere**. Matches the "zero BOOL
tensors" observation from the Qualcomm Qwen3-4B reference bundle.

### Runtime cost of Path B-mask: compute `attention_bias` each step

The X2E host-side `NPUSession` / `npu_draft_sidecar.py` has to feed
`attention_bias` per decode step. For the compile-target regime
(past=511, seq_q=1, seq_k=512) with *all 512 positions valid* (our
decode case — no padding, current position is always the tail of a
full window, no future to mask since we're at the end):

```python
attention_bias = np.zeros((1, 1, 1, 512), dtype=np.float32)
```

That's it — all zeros. If we ever need to support a partially-filled
window (prefill with padding, causal mask active), the feed would be
a lower-triangular `0.0 / -65504.0` (FP16 min cast to FP32) matrix.
For the current sd.npu draft path we don't need that case.

Path A needs no runtime feed change — identical signature to the
session 9 binary minus the removed `attention_mask` input.

## Correctness evidence

All three probes on CPU-ORT `CPUExecutionProvider` with
`ORT_DISABLE_ALL` graph optimizations. Probe regime: past=511 zeros +
BOS (151643) at position 511.

```
probe              cos      max_abs_diff  argmax_match  top5_overlap
---------------------------------------------------------------------
staged (2a+2b)     1.0      0.0           yes           5/5
patha              1.0      0.0           yes           5/5
pathbmask          1.0      0.0           yes           5/5
```

JSON evidence in `results/phase5_step6_probe_{staged,patha,pathbmask}.json`.
Op histograms and inventories in:
- `results/phase5_step6_bool_region.json`
- `results/phase5_step6_bool_region_dump.txt`

## IO signature summary (what `NPUSession` sees)

```
session 9 binary (broken):    59 inputs (input_ids, attention_mask, position_ids, 56 past_kv) / 57 outputs
qwen3-0.6b-patha:             58 inputs (input_ids, position_ids, 56 past_kv) / 57 outputs
qwen3-0.6b-pathbmask:         59 inputs (input_ids, position_ids, attention_bias, 56 past_kv) / 57 outputs
```

Path A removes one input (`attention_mask`) relative to session 9.
Path B-mask removes `attention_mask` AND adds `attention_bias` —
net same count, different shape.

`scripts/npu_draft_sidecar.py` needs a small tweak either way:

- **Path A:** stop feeding `attention_mask` at all (the graph no
  longer requests it).
- **Path B-mask:** stop feeding `attention_mask`; start feeding
  `attention_bias = np.zeros((1, 1, 1, 512), np.float32)` each step.

Both changes are one-line edits to whatever builds the feed dict. The
`bin_inspect.json` signature check documented in `status_x86.md` gives
you a rapid sanity check that the .bin's declared IO matches the ONNX
it was compiled from.

## Decision tree for aarch64 compile + validate

**Before compile:** read `status_x86.md` on x86 box to confirm the
artifacts you have match what's described here. Files to sanity-check:

- `qwen3-0.6b-patha/model.onnx` + `model.onnx_data` + sidecars
- `qwen3-0.6b-pathbmask/model.onnx` + `model.onnx_data` + sidecars
- `bin_inspect.json` on the old broken binary for IO-signature diff

**Compile order:** probably Path B-mask first. Reason: it's the
higher-confidence path (matches Qualcomm's production pattern in the
X2E zoo). If it compiles and decodes correctly, step 4 closes without
needing to resolve the "how strict is HTP's BOOL rule" question.

**Outcome matrix:**

| A compiles? | B-mask compiles? | Interpretation |
|---|---|---|
| ✓ | ✓ | HTP accepts BOOL tensors; Cast-to-BOOL was the only hard stop. Path A is cheapest; ship it. |
| ✗ | ✓ | HTP rejects BOOL tensors, not just casts. Ship Path B-mask. Confirms zoo pattern is load-bearing. |
| ✓ | ✗ | Unexpected. Path B-mask broke elsewhere (likely the additive-bias dtype or broadcast). Investigate Add_2 dtype promotion on HTP. |
| ✗ | ✗ | Both rejected. Escalate to Path B-full (RoPE externalization) or reconsider approach. |

**Validation gate (both artifacts, in order):**

1. `scripts/compile_qwen3_ai_hub.py --check --submit` per the existing
   step-4 playbook. Compile time: 15-25 min per artifact.
2. `bin_inspect.json` should show the expected IO (see table above).
3. `scripts/npu_vs_cpu_correctness.py` with the NPU-loaded binary —
   the four gates from session 9:
   - zero-KV + BOS cos ≥ 0.99 (was -0.18 broken).
   - Single-step prefilled cos ≥ 0.95 (was 0.55).
   - 16-step greedy match rate ≥ 50% (was 0%).
   - NPU stream produces recognizable English.

If any gate fails on a candidate that compiled cleanly, **do not ship
it** — the session 9 precedent showed AI Hub's op-lowering can accept
a graph that decodes wrong. The x86 cos=1.0 probe rules out rewrite
error on the ONNX side, so any gate failure is an HTP numerical
issue, which is a very different triage.

## Known deviations from `docs/phase5_export_on_x86.md`

Captured so the next session doesn't get confused reading the doc
next to the produced artifacts.

1. **Doc §"After 2a+2b"** predicted node count ~2,129 and 2 Cast→BOOL.
   Actual on a fresh optimum 2.1.0 export: 7,611 nodes and 3 Cast→BOOL.
   The doc's 2,129 is almost certainly a confused carry-over from the
   quarantined onnxsim output. Doc is under-counting Cast→BOOL because
   it missed the two BOOL-identity casts (Cast_5, Cast_6) that
   optimum inserts.
2. **Doc §3 "Where(bool, scores, -inf)"** is wrong about the mask
   application shape. Optimum produces a separate additive-mask
   subgraph: `Where(Slice_4_bool, 0.0, -inf)` → `Add(scores, .)` at
   each layer's `Add_2`. The graph already has an additive-bias
   structure; the BOOL subgraph is just *computing* the bias. This
   dramatically simplifies Path B-mask relative to the doc (no need
   to invent the additive pattern — just skip the BOOL-side
   computation and feed the bias directly).
3. **Recon showed 58 `Equal` nodes forming a separate BOOL surface**
   (shape-compute scaffolding: `Where(Equal, ConstantOfShape_zero,
   Reshape_int64)` patterns). These are unrelated to attention mask
   and sink at INT64. Doc didn't inventory them. Path A leaves them
   intact; Path B-mask is unchanged w.r.t. them. If HTP rejects these
   too, we'll learn it on compile and need a separate fix.
4. **Probe regime used here is `past=511, position=511`** rather than
   doc's "zero-KV + position 0" framing. Reason: staged/folded graphs
   pin attention_mask length to 512 internally (Path A) or expect
   attention_bias shape that matches 512 (Path B-mask). Source and
   candidate only produce identical numerics when both see the same
   total-length regime. Past=511+seq=1=total_len=512 matches the HTP
   decode compile target, which is what we actually care about.

## Unresolved (open for aarch64 team)

1. **HTP strictness on BOOL tensors**. See decision tree above. This
   cycle's two artifacts directly probe this.
2. **The 58 shape-compute `Equal → Where` patterns**. These produce
   BOOL but sink at INT64 (Where's then/else branches are INT64
   Reshape shapes). Whether HTP accepts BOOL-condition Where is a
   second question. If it doesn't, a future cycle may need to rewrite
   dynamic reshapes to static ones.
3. **w4a16 quantization pass**. Both artifacts ship as FP16-weights/
   FP32-activations. Per `results/ai_hub_model_zoo_check.md`, every
   Qualcomm X2E-shipped LLM is w4a16 and the gap is ~3× on HMX. Not
   a compile-correctness issue; a perf follow-up for post-step-4
   closure.
4. **`scripts/npu_draft_sidecar.py` runtime update**. One-line edit
   to the feed dict per the IO-signature table above. Do this before
   running `npu_vs_cpu_correctness.py` on the new binary.

## Reproducibility on x86

```bash
# From the specula repo root on an x86_64 host with uv installed:

uv venv --python 3.12 .venv-x86-export

VIRTUAL_ENV=$(pwd)/.venv-x86-export uv pip install --python .venv-x86-export/Scripts/python.exe \
    "optimum==2.1.0" "optimum-onnx==0.1.0" "torch==2.11.0" \
    "transformers==4.57.6" "huggingface_hub==0.36.2" \
    "onnx==1.21.0" "onnx-graphsurgeon" "onnxruntime" \
    "tokenizers" "numpy" "sentencepiece"

.venv-x86-export/Scripts/python.exe -m optimum.exporters.onnx \
    --model Qwen/Qwen3-0.6B \
    --task text-generation-with-past \
    --no-post-process \
    --dtype fp16 \
    --cache_dir models/.hf_cache \
    models/qwen3-0.6b-optimum

.venv-x86-export/Scripts/python.exe scripts/rewrite_qwen3_htp.py --mode stage
.venv-x86-export/Scripts/python.exe scripts/rewrite_qwen3_htp.py --mode fold-patha
.venv-x86-export/Scripts/python.exe scripts/rewrite_qwen3_htp.py --mode fold-pathbmask

# Verify: expect cos=1.0 verdict PASS on all three probes.
.venv-x86-export/Scripts/python.exe scripts/probe_cos_vs_source.py \
    --candidate models/qwen3-0.6b-staged/model.onnx
.venv-x86-export/Scripts/python.exe scripts/probe_cos_vs_source.py \
    --candidate models/qwen3-0.6b-patha/model.onnx
.venv-x86-export/Scripts/python.exe scripts/probe_cos_vs_source.py \
    --candidate models/qwen3-0.6b-pathbmask/model.onnx
```

Total wall-clock: ~12 minutes on a modest x86 box. Most of that is
the initial optimum export.

## Files delivered

In the transfer drop at `Z:\exposed\junk\`:

- `qwen3-0.6b-patha/` — Path A artifact (sidecars + model.onnx +
  model.onnx_data).
- `qwen3-0.6b-pathbmask/` — Path B-mask artifact (sidecars +
  model.onnx + model.onnx_data).
- This report as `phase5_step6_export_report.md`.

In the repo (tracked):

- `scripts/rewrite_qwen3_htp.py`
- `scripts/probe_cos_vs_source.py`
- `scripts/survey_bool_region.py`
- `status_x86.md`
- `docs/phase5_step6_export_report.md` (this file)
- `results/phase5_step6_*.json` + `.txt`
