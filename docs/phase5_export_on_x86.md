# Phase 5 — ONNX export on an x86_64 machine

Revised 2026-04-23 — **the X2E can now do this export natively.**
PyTorch 2.7+ ships `cp312-cp312-win_arm64` CPU wheels via
`download.pytorch.org/whl/cpu/`. The "Local export on the X2E" attempt
listed below as ruled out (`torch has no win_arm64 wheel`) is no longer
true — that constraint held when this doc was written but lifted in
April 2025 with the Microsoft + Arm collaboration. Smoke-tested
2026-04-23: optimum 2.1.0 export on torch 2.10.0+cpu produces a graph
bit-identical to the x86 baseline (7,667 nodes, same input/output
schema, same file size; logits cos=0.9999999995 vs the x86 ONNX on
CPU-ORT). See `docs/exporting_on_arm.md` for the verified install +
run recipe. This x86 doc is retained because (a) the in-repo evidence
trail and `status_x86.md` log all live here, and (b) x86 is still a
useful fallback if you want `torch==2.11.0` (no win_arm64 wheel for
2.11 yet).

Revised 2026-04-22 session 12 — **w4a16 compile requires Path B
(RoPE externalization).** Our existing `pathbmask` artifact (additive
FP16 mask, inline rotary_emb) compiles and runs at FP16 cleanly, but
fails AI Hub's w4a16 PTQ at the QNN backend op-validation step with
*"rotary_emb/MatMul: incorrect Value 0, expected equal to -32768"*.
Qualcomm's shipping Qwen3-4B w4a16 bundle on the same QAIRT 2.42 /
X2 Elite target works because they hoist rotary out of the compiled
graph and feed `position_ids_cos` / `position_ids_sin` as top-level
inputs. Path B was written up session 10 as an "alternative if Path
A fails"; session 12 evidence makes it **required** for any w4a16
compile. See the new section "Path B is required for w4a16 —
2026-04-22 finding" below, and the Lever C entry in
`docs/qwen3_perf_levers_investigation.md` for the full failure
analysis. **If you're here to produce the w4a16-capable ONNX: jump
straight to "Path B implementation contract (2026-04-22 revision)"
below** — skip the Path A fp16-only sections unless you're also
regenerating the fp16 baseline.

Revised 2026-04-21 session 10 — **major revision after session 10
diagnosis**: the onnxsim-based simplification from sessions 7-9
corrupts the graph. Compile succeeds but produces a binary that
anti-correlates with the CPU reference (cosine -0.18 on zero-KV +
BOS control). The "Simplify the graph with onnxsim" section below
has been replaced with a targeted surgical-fold approach that
preserves computational correctness.

Read top-to-bottom if you're picking this up cold. If you're
just here for the revised simplify step, jump to "Produce an
HTP-compilable ONNX (revised path)".

## Why this document exists

Phase 5 of the specula project needs a QAIRT-compatible ONNX of
`Qwen/Qwen3-0.6B` so we can submit it to Qualcomm AI Hub and receive a
Hexagon v81 QNN context binary. We're doing this on a Snapdragon X2
Elite Extreme laptop running Windows on ARM64.

Three attempts at producing that ONNX on the X2E itself have already
been ruled out:

1. **Local `optimum.exporters.onnx` export on the X2E.** Originally
   ruled out because torch had no `cp312 win_arm64` wheel. **No longer
   true as of 2026-04-23** — PyTorch 2.7+ ships win_arm64 wheels on
   `download.pytorch.org/whl/cpu/` (not on PyPI, which is why
   `pip install torch` still fails on the X2E). Smoke-tested
   2026-04-23 with `torch==2.10.0+cpu`: produces a numerically
   equivalent graph to the x86 export. Use `docs/exporting_on_arm.md`
   for the recipe. The x86 path below remains as a fallback /
   reference but is no longer the only option.
2. **Use the `onnx-community/Qwen3-0.6B-ONNX` pre-export from HF Hub.**
   Works as an artefact, but the graph uses four ORT-internal fused
   ops that QAIRT's QNN converter doesn't recognise:
   - `SimplifiedLayerNormalization` × 57
   - `SkipSimplifiedLayerNormalization` × 56 (com.microsoft)
   - `RotaryEmbedding` × 56 (com.microsoft)
   - `GroupQueryAttention` × 28 (com.microsoft)

   AI Hub rejected the upload at 100 s into the `OPTIMIZING_MODEL`
   phase with `No Op registered for SimplifiedLayerNormalization with
   domain_version of 14`. Full evidence in
   `results/ai_hub_compile_attempt3.log` and
   `results/npu_env_snapshot.txt`.
3. **`onnxruntime_genai.models.builder` with `execution_provider="qnn"`.**
   The previous version of this document recommended this path as
   Microsoft-official. **It is not functional** in the current PyPI
   release (`onnxruntime-genai` 0.13.1, 2026-04-20). The `"qnn"`
   string is silently accepted but there is no `qnn` entry in the
   builder's `ep_attrs` dict (only cpu/cuda/dml/webgpu/trt-rtx), and
   `grep -ri qnn` across the installed package finds zero matches.
   The produced ONNX still contains `com.microsoft::RotaryEmbedding`,
   `SkipSimplifiedLayerNormalization`, and `MultiHeadAttention` — the
   exact ops we were trying to eliminate. The call then crashes at
   `make_genai_config` with `KeyError: 'qnn'`. The previous export
   script `scripts/export_qwen3_qnn.py` is retained in the repo with
   a deprecation note for historical reference.

**Resolution:** use the fallback path, `optimum.exporters.onnx` with
`--no-post-process`. `--no-post-process` disables the ORT fusion pass
that would otherwise introduce the `com.microsoft` ops, so the exported
graph stays in the default ONNX domain and QAIRT can lower every node.
This has been verified to produce a clean graph (see "Verify the
output" below — 7,667 nodes, opset 18, zero `com.microsoft` ops).

This document describes running that export on an x86_64 machine
(Linux or Windows) and transferring the output back to the X2E for the
AI Hub compile step.

## Prerequisites on the x86 machine

- Python 3.10–3.12, x86_64.
- 10 GB+ free disk (HF cache + ONNX output + workspace; the optimum
  fp16 export alone is ~3 GB because `--no-post-process` also skips
  the weight-deduplication pass).
- Internet access (downloads the HF checkpoint from
  `huggingface.co/Qwen/Qwen3-0.6B`, ~1.2 GB).
- Ideally a GPU (CUDA or ROCm) to accelerate the trace. CPU works
  fine too — the reference run completed in ~90 s on an Intel Core
  Ultra 7 155H.
- `git`.

No QAIRT SDK install needed on the x86 machine — we only need it to
*produce* the ONNX. AI Hub does the Hexagon compile in the cloud from
the X2E side.

## Setup (one-time)

```bash
# Clone specula
git clone https://github.com/hotschmoe/specula.git
cd specula

# Fresh venv
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

# Install the export toolchain. `optimum-onnx` is a separate package
# from `optimum` as of optimum 2.x; both are needed.
pip install optimum "optimum-onnx" torch transformers huggingface_hub onnx
```

Versions known to work (reference run on 2026-04-20): `optimum 2.1.0`,
`optimum-onnx 0.1.0`, `torch 2.11.0+cpu`, `transformers 4.57.6`,
`huggingface_hub 0.36.2`, `onnx 1.21.0`. Note that installing
`optimum-onnx` will downgrade `transformers` from 5.x to 4.57.x — this
is intentional; optimum-onnx hasn't caught up to transformers 5 yet.

## Run the export

```bash
python -m optimum.exporters.onnx \
    --model Qwen/Qwen3-0.6B \
    --task text-generation-with-past \
    --no-post-process \
    --dtype fp16 \
    --cache_dir models/.hf_cache \
    models/qwen3-0.6b-optimum
```

What each flag does:

- `--model Qwen/Qwen3-0.6B` — HF model id, auto-downloaded.
- `--task text-generation-with-past` — generative decoder with KV cache
  inputs/outputs. Required; auto-inference picks up only `feature-
  extraction` for this model.
- `--no-post-process` — **the critical flag.** Skips the ORT fusion
  pass so `com.microsoft::` ops never get introduced. Without this, the
  graph looks like the HF pre-export from attempt #2 above.
- `--dtype fp16` — HTP runs fp16 natively, halves the file size.
- `--cache_dir models/.hf_cache` — keeps the HF download inside the
  repo tree (the `.gitignore` already excludes it).

Expected duration:

- With a GPU: ~1 min.
- CPU-only: ~90 s on Core Ultra 7 155H; a few minutes on lower-end
  parts. Most of the wall-clock is the HF download, not the trace.

At the end, optimum runs a PyTorch-vs-ONNX validation sweep and prints
max-diff values per output. fp16 max diffs in the ~1e-3 range are
normal and expected.

## DEPRECATED: original onnxsim-based simplify (do not run)

Sessions 7-9 used `onnxsim` with `--overwrite-input-shape` on all
inputs plus a pre-promoted attention_mask constant. That pipeline
produces a graph that compiles cleanly through AI Hub and loads
cleanly through ORT-QNN, but is **computationally corrupted**:

- Zero-KV + BOS control probe: cosine vs CPU reference = **-0.18**
  (anti-correlated).
- Single-step on prefilled KV: cosine = 0.55, top-5 overlap = 0/5,
  argmax is an Arabic glyph where CPU produces " a".
- 16-step greedy match rate: **0%**.

Session 10 diagnosis bisected the transforms:

| Transform starting from `patched` graph | cos vs source |
|---|---:|
| attention_mask promote-to-constant ONLY | **+1.0000** |
| IsNaN/Where guard elision ONLY | **+1.0000** |
| mask-promote + onnxsim(`--overwrite-input-shape`) | **-0.18** |

Both individual transforms are safe. The combination of
**attention_mask-as-constant + shape-pinned position_ids** causes
onnxsim's constant-folder to fold a position-dependent subgraph
with a wrong assumption — the exact mechanism is unknown, but the
symptom is that the graph anti-correlates with its source.

**Treat `scripts/simplify_qwen3_no_mask.py` as quarantined.** Use
the revised path below instead.

## Produce an HTP-compilable ONNX (revised path)

**Goal:** produce a graph where (a) CPU-ORT output matches the
optimum source with cosine ≥ 0.9999, (b) zero `com.microsoft`
ops, (c) zero Cast-to-BOOL nodes, (d) zero Range nodes. We can't
use onnxsim at all on the interior (it corrupts). Instead we do
two verified-safe rewrites and one targeted surgical fold of just
the attention_mask subgraph.

### Step 1: run the `optimum` export exactly as before

(Same command as the session-7 instructions above.) End state:
`models/qwen3-0.6b-optimum/model.onnx` with 7,667 nodes, no
`com.microsoft` ops.

### Step 2: apply the two safe rewrites

These two transforms individually verified cos = +1.0000 vs
source in session 10. Neither uses onnxsim; both are pure
protobuf edits.

**2a. Promote `attention_mask` from runtime input to
initializer.**

The decode regime we compile for always has all 512 positions
valid (no padding), so `attention_mask` is always `[1]*512` at
runtime. Make it a graph-level constant:

```python
import numpy as np
import onnx
from onnx import numpy_helper

m = onnx.load("models/qwen3-0.6b-optimum/model.onnx", load_external_data=True)

inputs_to_keep = [i for i in m.graph.input if i.name != "attention_mask"]
del m.graph.input[:]; m.graph.input.extend(inputs_to_keep)

arr = np.ones((1, 512), dtype=np.int64)
m.graph.initializer.append(numpy_helper.from_array(arr, name="attention_mask"))
```

**2b. Elide `Where(IsNaN(x), const, x)` NaN-safety guards.**

optimum emits 28 of these (one per layer, wrapping the softmax
output). HTP rejects BOOL in any form — `IsNaN` and `Where`-with-
bool-cond both fail. In the decode-only + full-KV regime, softmax
cannot produce NaN (denominator is always non-zero because the
mask allows ≥ 1 position), so `Where(IsNaN(x), 0, x) == x`
identically. Rewrite:

```python
nodes = list(m.graph.node)
producer = {o: n for n in nodes for o in n.output}
renames, drop = {}, set()
for w in nodes:
    if w.op_type != "Where": continue
    cond, _, false_val = w.input[:3]
    isnan = producer.get(cond)
    if isnan is None or isnan.op_type != "IsNaN": continue
    if isnan.input[0] != false_val: continue
    renames[w.output[0]] = false_val
    drop.add(id(w))
    isnan_users = [n for n in nodes if isnan.output[0] in n.input]
    if len(isnan_users) == 1: drop.add(id(isnan))
for n in nodes:
    for i, inp in enumerate(list(n.input)):
        if inp in renames: n.input[i] = renames[inp]
for o in m.graph.output:
    if o.name in renames: o.name = renames[o.name]
kept = [n for n in nodes if id(n) not in drop]
del m.graph.node[:]; m.graph.node.extend(kept)
```

After 2a + 2b, the graph has:
- Node count around 2,129 (from 7,667).
- 0 `IsNaN`.
- 0 `com.microsoft` ops.
- **2 remaining Cast-to-BOOL nodes**, both inside the
  attention_mask → causal-mask subgraph. They're the specific
  ops HTP will still reject.

**Confirm correctness at this stage** before moving on:

```python
import onnxruntime as ort
# Save to a temp path, load with ORT-CPU, feed zero-KV + BOS at
# position 0, compare argmax + logits to the same probe on the
# optimum source. Expect cos = 1.0000, argmax match.
```

### Step 3: targeted surgical fold of the attention_mask subgraph

**Implementation status: script TBD on x86.** The approach is
described below with enough detail to write; validation is
immediate via CPU-ORT plus the cos-vs-source probe in step 4.
Session 10 verified the approach works by tracing the specific
nodes involved on the X2E side; actual implementation lands on
x86 since that's where the ONNX tooling ecosystem is richer
(onnx-graphsurgeon is x86-only, for example).

With `attention_mask` now a known constant `[1]*512`, the
downstream chain in each attention block looks like:

```
attention_mask → Cast(BOOL) → Flatten → Cast(INT8) → Gather_5
    → Cast(INT8→BOOL) → Reshape → And_1 → ... → Where/Add → scores
```

**Nuance captured session 10:** `Gather_5`'s output depends on
its *data* (the constant flattened mask, always all-ones) AND
its *index* (which traces back to `position_ids` — runtime). So
`Gather_5`'s output shape is runtime-dependent, but its values
are always 1 because the data is all-ones. You cannot replace
`Gather_5` with a bare Constant (wrong shape), but you CAN
replace it with `ConstantOfShape(Shape(index), value=1)` — same
values, shape carried forward from the runtime index.

Algorithm:

1. **Find the BOOL-tainted region.** Walk the graph, marking
   every tensor whose producer is a `Cast(to=BOOL)` node OR
   whose consumer chain will hit a `Cast(to=BOOL)` in < 3 hops.
   This region is what HTP rejects.
2. **Classify each tainted tensor** as:
   - `data-constant`: values are runtime-shape but always the
     same (e.g., all-1 after Gather over all-1 data). Replace
     with `ConstantOfShape(Shape(runtime_dependent_input),
     value=const_value)`.
   - `fully-constant`: both shape and values are known. Replace
     with an `Initializer`.
   - `runtime`: actually varies. These shouldn't exist in the
     BOOL region if attention_mask is the only runtime-BOOL
     source; if you find one, escalate — may need Path B.
3. **Rewrite** each tainted tensor per its class, splice into the
   graph, remove the now-orphaned producer nodes.
4. **Save** the modified graph to `models/qwen3-0.6b-fixed/`
   with a single consolidated external-data sidecar.

Suggested tooling: `onnx-graphsurgeon` (NVIDIA's ONNX surgery
library, x86-only) for the rewrites — it handles node
splice/remove cleanly. Alternative: pure-protobuf edits using
the same style as steps 2a/2b.

**End state:** 0 Cast-to-BOOL, 0 Range, cos = +1.0000 vs source.

Write out:

```bash
# Consolidate weights + the newly folded initializers into one
# external-data sidecar so the AI Hub upload matches the staging
# layout the X2E compile script expects. Target directory:
# models/qwen3-0.6b-fixed/
```

**If step 3 turns out to be more work than budgeted:** fall back
to Path B (externalize RoPE + additive FP16 mask). That path
avoids the BOOL problem at source rather than trying to fold it
out; overall it's more surgery but no tricky graph-analysis
logic.

### Step 4: verify before transferring

```bash
python scripts/inspect_onnx_ops.py --model models/qwen3-0.6b-fixed/model.onnx
```

What you want:
1. Zero `com.microsoft` ops.
2. Zero `IsNaN` ops.
3. Zero `Cast ... to BOOL` ops.
4. Zero `Range` ops.

And a correctness probe (run this as a one-off — the full harness
is X2E-side at `scripts/npu_vs_cpu_correctness.py`, but we want
an early signal on the x86 side before burning another upload):

```python
# Feed zero-KV + BOS (token 151643) at position 0 to both the
# optimum source and the fixed graph on CPU-ORT. Compare
# argmax and cosine. Expect argmax match and cos > 0.9999.
```

### Why this works where onnxsim didn't

onnxsim's constant-folder is a whole-graph pass — with
`attention_mask` pre-promoted AND all input shapes pinned (including
`position_ids=[1,1]`), it has enough known values to fold
position-dependent subgraphs using whatever dummy value it picked
for `position_ids`, not knowing that the value is supposed to vary
at runtime. The surgical fold only evaluates the specific
attention_mask subgraph — it touches nothing that depends on
`position_ids`, so runtime position-dependence is preserved.

## Path B is required for w4a16 — 2026-04-22 finding

**Summary:** AI Hub's w4a16 PTQ pipeline cannot produce a valid QNN
context binary from an ONNX that contains `rotary_emb` as an internal
subgraph. Compile fails at the backend op-validation step after the
quantizer has already produced a DLC — wasting ~100 min of AI Hub
compute per attempt. The fix is to hoist rotary out of the graph so
`position_ids_cos` and `position_ids_sin` are top-level inputs,
mirroring Qualcomm's shipping Qwen3-4B w4a16 bundle. The additive-mask
rewrite we already did in `scripts/rewrite_qwen3_htp.py` (pathbmask)
is necessary but not sufficient for w4a16 — we also need the RoPE hoist.

**Evidence chain** (reproducible from this repo):

1. `results/aihub-compile-log-j563xme75-w4a16-a-FAILED.log` — full AI
   Hub log. Quantizer phase completes successfully; backend rejects
   `/model/rotary_emb/MatMul` with *"has incorrect Value 0, expected
   equal to -32768"* (INT16_MIN).
2. `models/qualcomm-qwen3-4b-ref/.../metadata.yaml` — Qualcomm's own
   Qwen3-4B w4a16 bundle on X2 Elite, QAIRT 2.42. Declares
   `position_ids_cos`, `position_ids_sin` as graph inputs of shape
   `[1,1,N,head_dim/2]`, dtype uint16, **offset -32768** (exactly the
   value the AI Hub backend wanted). Their graph has zero rotary ops.
3. Our `models/qwen3-0.6b-pathbmask-ai-hub-ctx256/model.onnx` —
   inspect with `inspect_onnx_ops.py`: `/model/rotary_emb/*` subgraph
   present, contains Cos/Sin/MatMul/Cast chain.

**Implication:** Path A (attention_mask as initializer) and our
current pathbmask (additive mask, inline rotary) both compile fine at
`--quantize_full_type float16` — that's our Phase 5 / Lever B baseline
and stays usable. Neither compiles at w4a16/w8a16. For the biggest
performance lever (Lever C in the investigation doc — W4A16), we need
a new artifact: **`models/qwen3-0.6b-pathb/`** with rotary hoisted.

---

## Path B implementation contract (2026-04-22 revision)

**Deliverable:** `models/qwen3-0.6b-pathb/model.onnx` + `model.data`,
handed back to the X2E side for prep + AI Hub compile.

**Source:** start from `models/qwen3-0.6b-pathbmask/model.onnx` —
our existing additive-mask rewrite. That artifact is produced by
`scripts/rewrite_qwen3_htp.py` applied to the optimum export. You're
extending it with the RoPE hoist; don't redo the mask rewrite.

If pathbmask isn't on the x86 machine yet, regenerate it from
`models/qwen3-0.6b-optimum/` (the plain optimum export per the
"Run the export" section above) using `scripts/rewrite_qwen3_htp.py`.
The X2E side has been successfully running it at FP16 since commit
7e10670 (Phase 5 close) so the rewrite logic is known-good.

### Expected structure of the rotary_emb subgraph (our optimum export)

Inspected on 2026-04-22 via `onnx.load('models/qwen3-0.6b-optimum/model.onnx', load_external_data=False)`.
The entire rotation computation lives under a single `/model/rotary_emb/*`
namespace (layer-independent — all 28 attention layers share one
rotary output). Key chain:

```
position_ids                                       # [1, seq_len], int64
  → /model/rotary_emb/Unsqueeze_1                  # add a dim
  → /model/rotary_emb/Cast_1 (to float32)          # -- the "interpreted at conversion time" op
  → /model/rotary_emb/MatMul                       # position_ids × inv_freq  -- FAILING OP under w4a16
  → /model/rotary_emb/Transpose
  → /model/rotary_emb/Concat_1                     # double-up to full head_dim
  → /model/rotary_emb/Cos  → /model/rotary_emb/Mul_1      # × attention_scaling
  → /model/rotary_emb/Sin  → /model/rotary_emb/Mul_2      # × attention_scaling
  (final outputs consumed by every layer's q_proj/k_proj RoPE apply)
```

A `Cast` node between the final Mul and the layer consumers may exist
to match the layer interior's dtype; preserve that dtype at the new
graph boundary.

**How the output feeds into layers:** find the tensor names consumed
by `/model/layers.0/self_attn/` (and .1, …, .27) that trace back to
`/model/rotary_emb/Mul_1_output_0` (cos) and `/model/rotary_emb/Mul_2_output_0`
(sin), or whatever the final Cast output is. These are the seams.

### Contract — inputs

Before (pathbmask, 59 inputs):
```
input_ids            [1, 1]             int64
position_ids         [1, 1]             int64
past_key_values.N.key    [1, n_kv, past_len, head_dim]  float32
past_key_values.N.value  [1, n_kv, past_len, head_dim]  float32  (× 56)
attention_bias       [1, 1, 1, ctx]     float32
```

After (pathb, 61 inputs):
```
input_ids            [1, 1]             int64
position_ids         [1, 1]             int64
past_key_values.N.key    [1, n_kv, past_len, head_dim]  float32
past_key_values.N.value  [1, n_kv, past_len, head_dim]  float32  (× 56)
attention_bias       [1, 1, 1, ctx]     float32
position_ids_cos     [1, 1, 1, head_dim]   float32          # NEW
position_ids_sin     [1, 1, 1, head_dim]   float32          # NEW
```

For Qwen3-0.6B: `head_dim = 128`, so both cos/sin inputs have shape
`[1, 1, 1, 128]`. Keep `position_ids` as a graph input even though
it's only used for the absolute-position semantics in
`past_key_values` indexing — the NPU graph consumers may still need
it downstream. (Sanity-check: grep the non-rotary part of the graph
for `position_ids` consumers; if none remain after the hoist, it's
safe to drop — but defaulting to "keep" matches Qualcomm's bundle.)

Dtype rationale: we pass FP32 at the graph boundary and let AI Hub's
quantizer assign the INT16 symmetric encoding during PTQ. Qualcomm's
metadata.yaml shows their final compiled input is uint16 with scale
≈ 3.05e-5 and offset -32768 — that's the encoding the quantizer will
produce from FP32 calibration data. Don't pre-quantize on the x86
side.

**Placement in `graph.input`:** append `position_ids_cos` and
`position_ids_sin` after `attention_bias` (which is currently the
last input). Preserving the old inputs' order keeps the existing
pathbmask calibration infrastructure compatible. The X2E side's
`build_input_specs` will add the two new entries at the end.

### Contract — outputs

Unchanged from pathbmask. The model still emits logits + 56 present
KV tensors in the same order.

### Implementation sketch

Pure protobuf edits (don't use onnxsim — it corrupted the graph in
session 10 and the quarantine is still in effect per the DEPRECATED
section above).

```python
import numpy as np
import onnx
from onnx import helper, TensorProto

m = onnx.load("models/qwen3-0.6b-pathbmask/model.onnx", load_external_data=True)

# 1. Find the rotary_emb terminal outputs (cos path + sin path).
#    These are the tensors consumed by every layer's self_attn that
#    originated in /model/rotary_emb/*. Trace producers from layer
#    consumers back until you land inside rotary_emb.
rotary_cos_tensor = "/model/rotary_emb/Mul_1_output_0"   # VERIFY on the actual export
rotary_sin_tensor = "/model/rotary_emb/Mul_2_output_0"   # VERIFY on the actual export

# (If there's a trailing Cast to match layer dtype, trace past it to
# the Cast output — the new graph inputs should match that dtype.)

# 2. Introduce the two new graph inputs. Shape [1, 1, 1, head_dim]
#    for decode-step ctx; the existing build_input_specs on the X2E
#    pins seq_q=1. Dtype FP32 (AI Hub PTQ quantizes per calibration).
HEAD_DIM = 128
cos_input = helper.make_tensor_value_info(
    "position_ids_cos", TensorProto.FLOAT, [1, 1, 1, HEAD_DIM]
)
sin_input = helper.make_tensor_value_info(
    "position_ids_sin", TensorProto.FLOAT, [1, 1, 1, HEAD_DIM]
)
m.graph.input.append(cos_input)
m.graph.input.append(sin_input)

# 3. Rewire every consumer of rotary_cos_tensor → "position_ids_cos",
#    and rotary_sin_tensor → "position_ids_sin". Iterate node inputs,
#    rename matches.
renames = {rotary_cos_tensor: "position_ids_cos",
           rotary_sin_tensor: "position_ids_sin"}
for node in m.graph.node:
    for i, inp in enumerate(list(node.input)):
        if inp in renames:
            node.input[i] = renames[inp]

# 4. Delete every node whose outputs are exclusively inside
#    /model/rotary_emb/* and have no remaining consumers. Walk the
#    graph in reverse and drop producers that became orphans after
#    step 3. (Anything with a constant output that happens to get
#    consumed outside rotary_emb stays — unlikely in practice.)
rotary_prefix = "/model/rotary_emb/"
keep = []
orphan = set()
live_inputs = {v.name for v in m.graph.input} | \
              {v.name for v in m.graph.initializer}
# First pass: collect all tensor consumers
consumers = {}
for node in m.graph.node:
    for inp in node.input:
        consumers.setdefault(inp, set()).add(id(node))
# Second pass: walk nodes in order, drop any rotary_emb node whose
# outputs have no live consumers
for node in m.graph.node:
    if node.name.startswith(rotary_prefix):
        node_consumers = set()
        for out in node.output:
            node_consumers |= {cid for cid in consumers.get(out, set())
                               if cid not in orphan}
        if not node_consumers:
            orphan.add(id(node))
            continue
    keep.append(node)
del m.graph.node[:]
m.graph.node.extend(keep)

# 5. Save with external data consolidated.
onnx.save(
    m,
    "models/qwen3-0.6b-pathb/model.onnx",
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location="model.data",
    size_threshold=1024,
)
```

Exact node names may drift with optimum version bumps — always
re-verify via a quick `onnx.load` + print of `/model/rotary_emb/*`
nodes before running the rewrite (pattern in the session-12 inspection
at the top of this section).

### Correctness gate — before handoff

Run a CPU-equivalence probe on the new `pathb/model.onnx` before
sending it to the X2E side. The x86 side has ORT, so this is cheap.

```python
import numpy as np
import onnxruntime as ort

REF = "models/qwen3-0.6b-optimum/model.onnx"
NEW = "models/qwen3-0.6b-pathb/model.onnx"

# Helper: compute cos/sin tables for a single decode step.
def rope_tables(position_id, head_dim=128, rope_theta=1000000.0):
    # Qwen3 uses theta=1e6 (confirmed via config.json rope_theta).
    inv_freq = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    freqs = position_id * inv_freq                    # [head_dim/2]
    emb = np.concatenate([freqs, freqs], axis=-1)     # [head_dim], matches optimum's Concat_1
    return (np.cos(emb)[None, None, None, :].astype(np.float32),
            np.sin(emb)[None, None, None, :].astype(np.float32))

# Prompt: simple BOS probe.
BOS = 151643  # Qwen3 BOS token id
pos = 0
cos, sin = rope_tables(pos)

# Empty KV for zero-past probe. Build feed for both sessions.
# ... (full probe logic: run REF with position_ids=[[0]], BOS input,
#      zero KV; run NEW with same + cos/sin; compare last-token logits.)

# Gate:
#   cosine(ref_logits, new_logits) >= 0.9999
#   argmax(ref_logits) == argmax(new_logits)
#   top-5 overlap == 5/5
```

If any of these fails, the rotary rewrite has introduced numerical
drift — likely wrong cos/sin computation (theta mismatch, wrong
concatenation order, missing attention_scaling multiplier). Fix
before handoff; the X2E side's correctness probe is stricter
(cos ≥ 0.95 after w4a16 quantization) so the pre-quant graph must
be near-exact.

**Gotcha on attention_scaling:** optimum's rotary_emb ends with
`× Constant_7` (see `/model/rotary_emb/Mul_1`) — that's the per-layer
attention_scaling factor Qwen3 applies to cos/sin. Find its value
(inspect the Constant_7 initializer) and either:
(a) fold it into the cos/sin values we feed at runtime, **or**
(b) leave it in the graph as a downstream Mul that still runs
    post-hoist (simpler; less chance of silent numerical drift).
Option (b) means the rewrite renames the consumer of Mul_1_output_0
(not Mul_1's inputs), leaving Mul_1 itself in place. Preferable.

### Deliverable checklist

- [ ] `models/qwen3-0.6b-pathb/model.onnx`
- [ ] `models/qwen3-0.6b-pathb/model.data`
- [ ] `models/qwen3-0.6b-pathb/config.json` (copied from optimum)
- [ ] `models/qwen3-0.6b-pathb/tokenizer.json` (copied from optimum)
- [ ] CPU-equivalence probe output: cos ≥ 0.9999 + argmax match + top-5 5/5
- [ ] `inspect_onnx_ops.py` output showing zero nodes under `/model/rotary_emb/`
- [ ] `graph.input` shape list showing `position_ids_cos` / `position_ids_sin`
      appended after `attention_bias`, with `[1, 1, 1, 128]` float32 each

Transfer mechanism: same as Path A (section "Transfer the output back
to the X2E machine"). The X2E side will prep + upload + compile with
`--quant w4a16 --calibration-npz <regenerated Bundle A>`. The existing
Bundle A calibration **must be regenerated** once pathb lands,
because the compile-time `input_specs` now has two extra inputs and
the calibration-data schema validator will reject the old 59-input
bundle against the new 61-input graph. Capture script adaptation is
trivial: add `cos`/`sin` fields to `capture_for_prompt`'s output dict
in the right position (after attention_bias).

---

## Alternative Path B: Qualcomm-style (externalize RoPE + additive FP16 mask)

**When to use this:** if Path A's surgical fold produces a graph
that still fails AI Hub op-lowering, or if we later need
higher-throughput w4a16 quantization. Larger surgery, but the
architecturally robust route — matches how Qualcomm ships their
own published X2E Qwen3-4B binary.

**2026-04-22 update:** this section is superseded by the concrete
implementation contract above. Leaving the original narrative in
place for context on how we arrived at the hoist decision; new
implementation should follow "Path B implementation contract (2026-04-22
revision)".

The Qualcomm Qwen3-4B Genie bundle (inspected session 10, copy
local at
`models/qualcomm-qwen3-4b-ref/qwen3_4b-genie-w4a16-qualcomm_snapdragon_x2_elite/`)
reveals the full production pattern. The two load-bearing
architectural choices:

1. **Externalize RoPE.** The HTP graph contains **zero**
   Cos/Sin/Range ops. `position_ids_cos` and `position_ids_sin`
   are input tensors of shape `[1, 1, seq_len, rope_dim/2]`
   pre-computed on CPU per-step. The graph applies them via
   plain Mul/Add.

2. **Additive FP16 attention mask.** `attention_mask` is a
   runtime input of shape `[1, 1, seq_q, seq_k]` at FP16/INT16
   (not BOOL), containing 0s at valid positions and large
   negative values (-inf equivalent) at masked positions. The
   graph adds it to attention scores pre-softmax. **Zero BOOL
   tensors anywhere in the graph.**

The bundle also uses a 4-part binary split (embed + 3 transformer
chunks), 5 context-length tiers, and 2 AR batch modes (128 for
prefill + 1 for decode), plus full w4a16 quantization — but those
are orthogonal and can be layered later. The two choices above
are what make the graph *naturally* HTP-lowerable without any
constant-folding trickery.

### Implementation sketch (Path B)

We don't have access to Qualcomm's internal export pipeline, but
we can approximate the two key rewrites on top of the optimum
export.

**Externalize RoPE:**

1. In optimum-onnx, RoPE is typically a subgraph like:
   `position_ids → Cast → Gather(sin_cache, position_ids) → ...`
   for sin; same for cos. Find all such subgraphs per layer.
2. Replace each Gather with a new graph input
   `position_ids_cos_{i}` / `position_ids_sin_{i}` (or a shared
   pair across layers if RoPE is layer-independent in Qwen3 —
   it usually is).
3. Delete the Gather + Range + Cos/Sin ops that produced the
   subgraph.
4. On the runtime side, pre-compute cos/sin for the current
   position on CPU and pass them in each decode step.

**Additive mask:**

1. In optimum-onnx, attention mask is computed as:
   `attention_mask → Cast(BOOL) → ... (Range+Gather logic) → Where(mask, scores, -inf)`.
2. Replace the whole subgraph that produces the `Where` second-arg
   condition with a new graph input `attention_bias` of shape
   `[1, 1, 1, 512]` (or `[1, 1, seq_q, seq_k]` for batched
   prefill), dtype FP16.
3. Replace `Where(mask_bool, scores, -inf)` with `Add(scores,
   attention_bias)`.
4. On the runtime side, pre-compute the additive mask on CPU:
   `0.0` for valid positions, `-65504.0` (FP16 min) for masked
   positions. For a fully-populated decode window, the mask is
   all zeros — degenerate but cheap.

**Verification targets** (same as Path A):
- cos ≥ 0.9999 vs optimum source on zero-KV + BOS probe.
- Zero Cast-to-BOOL, zero Range, zero IsNaN, zero Cos, zero Sin.

**Estimated scope:** Path A is ~1 day of surgical edits. Path B
is ~3-5 days — the RoPE externalization in particular needs
careful per-layer identification of the sin/cos Gather pattern
and rewiring of all consumers.

**Recommendation:** try Path A first. Only escalate to Path B
if Path A's output still fails AI Hub compile on op-lowering
grounds, or when we later graduate to w4a16 for perf.

## Inspect ops (sanity check on any stage)

Useful at both the raw optimum output and after the Path A
rewrites, to confirm no `com.microsoft` ops survived and to track
the BOOL/IsNaN/Range counts shrinking to zero.

```bash
python scripts/inspect_onnx_ops.py --model models/qwen3-0.6b-optimum/model.onnx
# later, after Path A rewrites:
python scripts/inspect_onnx_ops.py --model models/qwen3-0.6b-fixed/model.onnx
```

What you want to see at the final stage:

```
opset imports:
  <default>            version 18

... op histogram here ...

all ops are in the default onnx domain
```

What you do **not** want to see:

```
NON-STANDARD ops (will need decomposition or QAIRT custom op):
  com.microsoft::...
```

**Histograms at each stage** (session 10 measurements):

| Stage | nodes | IsNaN | Cast→BOOL | Range | cos vs optimum |
|---|---:|---:|---:|---:|---:|
| optimum (raw export) | 7,667 | 28 | 3 | 3 | 1.0000 (ref) |
| + safe rewrites 2a+2b | 2,129 | 0 | 2 | 0 | 1.0000 |
| + step 3 surgical fold | ~2,050 | 0 | **0** | 0 | 1.0000 (target) |
| **[DEPRECATED]** onnxsim-broken `nomask` | 2,033 | 0 | 0 | 0 | **-0.18** |

The deprecated path looks structurally "clean" (zero BOOL, zero
Range) but silently corrupts the math. Both the structural
histogram AND the correctness probe are required.

If `com.microsoft` ops somehow survive the optimum export:

1. Double-check `--no-post-process` is on the command.
2. Retry with `--optimize O1` explicitly off (it should be off by
   default without `--optimize`).
3. Escalate with the inspect_onnx_ops.py output — may need a targeted
   decomposition pass.

## Transfer the output back to the X2E machine

**Transfer the `qwen3-0.6b-fixed/` directory** (the Path A output,
or `qwen3-0.6b-fixed-pathb/` if you went with Path B). The X2E
side has no onnxsim and won't re-run any graph surgery, so
whatever lands on its disk is what gets uploaded to AI Hub.

**Do NOT** transfer `qwen3-0.6b-nomask/` from the session 7-9
pipeline. That directory is the quarantined output — re-uploading
it reproduces the session 9 binary that fails step 6.

The fp16 export is ~3 GB. Pick your transfer mechanism:

```bash
# scp (simplest over SSH)
scp -r models/qwen3-0.6b-fixed/ <user>@<x2e-host>:<path>/specula/models/

# rsync (resumable, good for slow links)
rsync -av --progress models/qwen3-0.6b-fixed/ \
    <user>@<x2e-host>:<path>/specula/models/qwen3-0.6b-fixed/

# Mapped Google Drive / cloud sync (the previous run used
# G:\Shared drives\MAIN\Junk\qwen3-0.6b-optimum\ as the drop site)
cp -r models/qwen3-0.6b-fixed "/path/to/mounted/drive/"

# USB / your tool of choice
```

Also copy over the supporting files (tokenizer, config, etc.) from
the pre-rewrite directory if the surgical rewrites didn't preserve
them (they won't — the rewrites are protobuf-only):

```bash
cp models/qwen3-0.6b-optimum/*.json models/qwen3-0.6b-fixed/
cp models/qwen3-0.6b-optimum/tokenizer.json models/qwen3-0.6b-fixed/
cp models/qwen3-0.6b-optimum/merges.txt models/qwen3-0.6b-fixed/
cp models/qwen3-0.6b-optimum/vocab.json models/qwen3-0.6b-fixed/
cp models/qwen3-0.6b-optimum/chat_template.jinja models/qwen3-0.6b-fixed/
```

Do **not** commit the ONNX to git. The repo's `.gitignore` excludes
`*.onnx`, `*.onnx_data`, and `*.safetensors` for exactly this reason
(model weights don't belong in VCS; see the download_qwen3_onnx.py
pattern for the preferred "script the download" approach).

## Resume on the X2E side

Once the directory lands at `<specula>/models/qwen3-0.6b-fixed/`
on the X2E machine:

1. **Sanity-check the transfer** (file sizes match what you sent —
   `model.onnx_data` should be ~3.0 GB).

2. **Point the staging + compile scripts at the new source.** The
   session 9 pipeline used `models/qwen3-0.6b-nomask/` as the
   source (the quarantined directory). Update:
   - `scripts/prep_onnx_for_ai_hub.py`: change `SOURCE_ONNX` to
     `models/qwen3-0.6b-fixed/model.onnx`.
   - `scripts/compile_qwen3_ai_hub.py`: change `SOURCE_DIR` to
     `models/qwen3-0.6b-fixed`.

3. **Re-stage and compile:**

   ```powershell
   .venv\Scripts\python.exe scripts\prep_onnx_for_ai_hub.py
   .venv\Scripts\python.exe scripts\compile_qwen3_ai_hub.py --check    # dry-run
   .venv\Scripts\python.exe scripts\compile_qwen3_ai_hub.py --submit   # ~15-25 min
   ```

4. **Expected outcome:** `OPTIMIZING_MODEL` phase completes, followed
   by quantization (if applicable) and final context-binary
   generation. Downloaded artefact:
   `models/qwen3_0_6b_draft_v81_ctx512.bin` (~500–800 MB FP16).
   Keep the previous binary as
   `qwen3_0_6b_draft_v81_ctx512.broken.bin` for comparison.

5. **Validate before declaring step 4 re-closed.** This is the
   lesson from session 10 — compile success is NOT enough. Run
   the full correctness harness:

   ```powershell
   .venv\Scripts\python.exe scripts\npu_vs_cpu_correctness.py
   ```

   Exit criteria:
   - Zero-KV + BOS control: cosine ≥ 0.99 (was -0.18 on broken).
   - Single step on prefilled KV: cosine ≥ 0.95 (was 0.55).
   - 16-step sliding-window greedy match rate ≥ 50% (was 0%).
   - NPU stream produces recognizable English text (was Arabic
     glyphs on repeat).

   If any of these fails, capture `results/bin_inspect.json` on
   the new binary and compare IO shapes to the one in git
   (session 9); the fix to the ONNX should not have changed the
   bin signature (same 58/57 input/output tensors, same dtypes).

6. **If compile fails again on a different op:** capture the failure
   reason, inspect the ops via `inspect_onnx_ops.py`, and either
   regenerate with different optimum flags (on the x86 machine) or
   escalate for a targeted decomposition. See
   `results/npu_env_snapshot.txt` for the prior failure-mode log as
   a template for what to capture. This is also the signal to try
   Path B (Qualcomm-style externalized RoPE + additive FP16 mask)
   instead of Path A.

## Where this fits in the larger plan

- **Scoping doc:** `docs/npu_scoping.md` — full Phase 5 plan, the
  10-step bring-up, toolchain pins, known failure modes.
- **Current session log:** `results/npu_env_snapshot.txt` — steps 1-4
  execution history, including the three AI Hub attempts.
- **Status doc:** `current_status.md` — project-wide phase tracking.

Steps 1-3 of the scoping doc are green. Step 4 is blocked on the work
this doc describes. Step 5 (load the .bin on the X2E via ORT-QNN)
picks up as soon as step 4 produces a working binary — the
`NPUSession` wrapper in `scripts/npu_draft_sidecar.py` is already in
place and was smoke-tested at step 2 against a toy graph. Swap the
model path, rebind to the Qwen3 IO surface, and we're running.

## Reference: alternative paths considered

Documented here so future sessions don't re-derive these from scratch.

- **Manual ONNX graph surgery** to decompose the four fused ops on the
  X2E side. 6-11 hours of work; `GroupQueryAttention` with KV-cache
  semantics is the bulk of the cost and the highest bug risk. Ruled
  out in favour of producing a clean graph at source.
- **`onnxruntime_genai.models.builder` with `execution_provider="qnn"`.**
  Previously recommended as the "Microsoft-official QNN path." Does
  not work in PyPI 0.13.1 — no QNN support present in the package.
  See the "Why this document exists" section and
  `scripts/export_qwen3_qnn.py` for the deprecation note. If a future
  ort-genai release adds QNN support, reconsider.
- **QAIRT custom-op authoring** (Path C in the scoping doc). Ruled out
  at scoping time because Hexagon Skels have to be signed and retail
  Windows blocks testsigning.
- **Pre-compiled Qwen3 in the AI Hub Model Zoo.** A "check first"
  shortcut — if Qualcomm has already published a Qwen3-0.6B QNN
  context binary for Snapdragon X2 Elite, we skip the ONNX step
  entirely. Recommended to browse <https://aihub.qualcomm.com/models>
  before burning the export cycle. See
  `results/ai_hub_model_zoo_check.md` (if present; otherwise perform
  the check yourself).

## Follow-up: w4a16 precision (perf, not correctness)

Every Qualcomm-published HTP LLM on X2 Elite ships at **w4a16** (int4
grouped weights, fp16 activations). The Llama-v3.2-1B benchmark on X2E
is 90.36 t/s at w4a16 vs 27.95 t/s at w4 on identical hardware — a
~3× gap coming directly from Hexagon v81's HMX matrix units having
int4×fp16 as a native op. Details in
`results/ai_hub_model_zoo_check.md`.

**2026-04-22 update:** w4a16 is now an active workstream (Phase 5.5
Lever C in `docs/qwen3_perf_levers_investigation.md`). The x86 export
must produce **Path B (rotary hoisted + additive mask)**, not the
Path A / pathbmask graphs that power our fp16 baseline. See "Path B
is required for w4a16 — 2026-04-22 finding" above for the failure
evidence and "Path B implementation contract (2026-04-22 revision)"
for the deliverable spec.

Pipeline (2026-04-22 version):

1. **x86 side (this doc's remit):** produce `models/qwen3-0.6b-pathb/`
   from the existing optimum export + `pathbmask` rewrite + RoPE hoist.
   Validate CPU equivalence cos ≥ 0.9999.
2. **X2E side:** regenerate the w4a16 calibration bundle for the new
   61-input schema (add cos/sin entries); submit
   `--quant w4a16 --calibration-npz models/calibration/bundle_a_pathb_ctx256.npz`
   via `scripts/compile_qwen3_ai_hub.py`. Expected compile time ~25-40
   min once calibration uploads.
3. **X2E side:** runtime plumbing to compute cos/sin per decode step
   and feed them to the NPU session alongside the existing inputs.
   Update `npu_load_qwen3_bin.py::describe_inputs`,
   `npu_short_prompt_probe.py`, and the async outer loop.
4. **X2E side:** correctness probe (cos ≥ 0.95 tolerated post-w4a16)
   + full sweep against the AC FP16 baseline (18.12 t/s mean).

All w4a16-specific session history (two failed compile attempts,
calibration-data order bug, rotary_emb op-lowering diagnosis) is in
the Lever C section of `docs/qwen3_perf_levers_investigation.md`.
