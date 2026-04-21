# Phase 5 step 6 — aarch64 compile retrospective (session 11, 2026-04-21)

First AI Hub compile cycle on the x86-produced Path A + Path B-mask
artifacts. Both jobs **failed**, but the workbench logs yielded two
load-bearing research findings that redirect the x86 hypothesis table
and unblock the next cycle with a trivial protobuf edit.

Companion to `docs/phase5_step6_export_report.md` (x86 side) and
`docs/phase5_export_on_x86.md` (full plan).

## Jobs

| path key   | job ID       | AI Hub status | elapsed | failure kind                 |
|------------|--------------|---------------|--------:|------------------------------|
| patha      | `jp83n13kg`  | FAILED        |    130s | internal compiler error      |
| pathbmask  | `jp01w619g`  | FAILED        |    130s | pre-compile shape validation |

Full compile logs: `results/ai_hub_logs/jp83n13kg.log/`, `jp01w619g.log/`.

## Finding 1 — AI Hub folds BOOL subgraphs cleanly (hypothesis refuted)

The x86 export report framed Path A vs Path B-mask as a probe of
HTP's strictness on BOOL tensors. Path A's compile log shows the
hypothesis was **wrong on its face**: AI Hub's
`OverrideFoldConstantsPass` folded the entire BOOL-tainted region
to constants before the compiler even looked at op-lowering.

Op histogram before → after the fold pass (from `jp83n13kg.log:18-49`):

| op              | before | after | delta  | note |
|-----------------|-------:|------:|-------:|------|
| Cast            |    348 |     1 |   -347 | Cast-to-BOOL chain vanished |
| ConstantOfShape |     60 |     0 |    -60 | Path A's own ConstantOfShape splice also folded |
| Equal           |     58 |     0 |    -58 | 58× shape-compute BOOL surface gone |
| And             |      2 |     0 |     -2 | |
| LessOrEqual     |      1 |     0 |     -1 | causal-mask BOOL |
| Range           |      3 |     0 |     -3 | position-aware ranges |
| Where           |     86 |     0 |    -86 | all bool-cond Where folded |
| Gather          |    371 |     1 |   -370 | |
| Shape           |    430 |     0 |   -430 | |
| Unsqueeze       |    884 |    59 |   -825 | |
| Op count        |   7580 |  5567 |  -2013 | |

**Implication:** HTP/AI Hub's toolchain handles BOOL subgraphs
competently *as long as the graph has enough concrete-shape
information for its constant folder to reach the BOOL ops*. The
x86-side surgical rewrites (Path A's ConstantOfShape-for-Gather_5
and Path B-mask's Where→Add splice) were unnecessary — not harmful,
but the smaller "just export + pin shapes" path is sufficient in
principle.

This subsumes an open question from the export report: "How strict
is HTP's BOOL rule?" Answer: not strict at all once shapes are
concrete. The surgical-fold work the x86 team did isn't wasted —
Path A gives us a smaller post-fold graph than naïve optimum, and
Path B-mask is architecturally closer to Qualcomm's production
pattern — but it wasn't load-bearing for *compilability*.

## Finding 2 — Dynamic input shapes break AI Hub's compiler

After the successful fold pass, Path A's compile crashed in the
next rewriter step. Relevant trace (`jp83n13kg.log:50-97`):

```
File "/tetra/tetracode/src/www/python/tetrai/compilers/onnx2onnx/_op_identity.py", line 30, in rewrite
    shape = np.broadcast_shapes(x.shape.dims, y.shape.dims)
  File "/tetra/tetra_env/lib/python3.10/site-packages/numpy/lib/stride_tricks.py", line 472
    arrays = [np.empty(x, dtype=[]) for x in args]
TypeError: 'SymbolicDim' object cannot be interpreted as an integer
```

An internal rewrite pass assumes concrete shape dims and falls over
when it hits a symbolic one. Our ONNX declares:

```
input_ids        (-1, -1)
position_ids     (-1, -1)
past_key_values.N.{key,value}   (-1, 8, -1, 128)
attention_bias   (-1, 1, -1, -1)     [Path B-mask only]
```

— all dynamic on batch + sequence. That dynamism propagates into
downstream tensors and the rewriter's `broadcast_shapes` call eats
it.

Path B-mask never made it this far. Its `jp01w619g.log` has exactly
one content line before FAIL: a pre-compile validator rejecting the
static `input_specs` we submit against the dynamic-shape declaration
in the uploaded ONNX. **Same root cause, earlier failure point.**

### Why didn't we see this in session 9?

Session 9's `nomask` compile succeeded because the quarantined
onnxsim pipeline called `onnxsim.simplify(..., overwrite_input_shapes=...)`
which pinned every input dim to a concrete integer inside the ONNX
before upload. AI Hub never saw a `SymbolicDim` and the validator
had nothing to reject. The x86 team correctly diagnosed onnxsim as
numerically corrupting, but replacing it with plain protobuf
surgical folds preserved the dynamic shapes rather than pinning
them.

## Finding 3 — The fix is a pure protobuf edit

Pinning the shapes is a 20-LOC transform on the `ValueInfoProto` of
each graph input: replace each `TensorShapeProto.dim` entry's
`dim_param` (symbolic) with a concrete `dim_value` matching the
compile-time `input_specs`. No compute, no fold, no numerical
impact — session 10's bisection already established that attention-mask-
promote, IsNaN-elide, and similar pure protobuf edits preserve
cos=1.0.

The pin is idempotent, so re-running the prep step on an
already-pinned graph is a no-op.

The pin lives in `scripts/prep_onnx_for_ai_hub.py` (after the
existing `patch_external_data_refs` step), driven by the per-path
specs from `scripts/compile_qwen3_ai_hub.py:build_input_specs` and
`PATHS[key]["extra_input_specs"]`. Single source of truth for
"what shape should each input be" remains the compile script.

## Forward plan for session 11 cont.

1. Land the pin-shapes pass in `prep_onnx_for_ai_hub.py`.
2. Re-stage both paths (idempotent; pin runs automatically).
3. Resubmit **serially** (Path A then Path B-mask). Serial avoids
   halving the upstream bandwidth — each upload is ~25 min single-
   stream vs. ~50 min in parallel on this link. Total wall-clock:
   ~65 min (25 + 7 + 25 + 7) including compile + download.
4. Run `scripts/npu_vs_cpu_correctness.py --path patha|pathbmask`
   on each binary.
5. Fill in the x86 report's 2×2 outcome matrix based on which
   candidates compile AND pass the 4-gate correctness probe.

## Predictions (to verify next cycle)

- **Path A**: the `OverrideFoldConstantsPass` already folded its
  BOOL region in the first run; the only new variable this cycle
  is pinned shapes. Expected: compile succeeds.
- **Path B-mask**: gets past shape validation. Whether its folded
  graph compiles or hits a different pass bug is the open question
  — no op-histogram datapoint for it yet because the first run
  died at validation.
- **Correctness**: if either binary compiles, the x86 CPU-ORT
  cos=1.0 probe guarantees the ONNX input is numerically correct,
  so any wrong decode output on NPU would be HTP numerical
  fidelity (FP16 drift, HMX rounding), not an ONNX rewrite bug.

## Side note: other Qualcomm productions

The Qualcomm Qwen3-4B Genie bundle (`models/qualcomm-qwen3-4b-ref/`)
ships with fully static shapes per variant (one context tier per
compiled binary, one AR batch size per variant). Qualcomm's own
production pipeline implicitly fixes the shape-pin question: they
only ever compile against concrete shapes. Our single-tier ctx-512
decode-only compile matches that pattern.
