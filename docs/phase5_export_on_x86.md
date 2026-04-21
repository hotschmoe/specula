# Phase 5 — ONNX export on an x86_64 machine

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

1. **Local `optimum.exporters.onnx` export on the X2E.** Requires
   torch. torch has no `cp312 win_arm64` wheel (older torch 2.1.2
   ships cp38-cp311 win_arm64 only; nothing newer ships win_arm64 at
   all). Hard wall.
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

## Alternative Path B: Qualcomm-style (externalize RoPE + additive FP16 mask)

**When to use this:** if Path A's surgical fold produces a graph
that still fails AI Hub op-lowering, or if we later need
higher-throughput w4a16 quantization. Larger surgery, but the
architecturally robust route — matches how Qualcomm ships their
own published X2E Qwen3-4B binary.

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

Our current export path (fp16 weights + fp16 activations) lands on a
slower HTP datapath. For a first working binary this doesn't matter —
get compile-to-execution working, then revisit precision. When we do
revisit, the path is:

1. On the x86 machine, run the export with w4a16 quantization (either
   via AIMET or AI Hub's `--quantize_full_type int4 --quantize_io`).
2. Either requires ~50-100 calibration prompts; use specula's existing
   `prompts/humaneval_subset.jsonl` + `structured_json.jsonl`.
3. Re-upload and recompile on the X2E side.

Not blocking for the current push; tracked as a perf-tuning pass once
the pipeline is unblocked.
