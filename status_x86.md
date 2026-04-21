# Phase 5 step 6 — x86 export handoff log

**Operator:** isaac@mass-engineering.com (hotschmoe) via Claude Code.
**Started:** 2026-04-21.
**Host:** x86_64 Windows 11 Pro, Intel B50 Pro (no CUDA), MSYS/bash.
**Branch:** master @ 7f0838b (clean at start).
**Goal:** produce two HTP-compilable Qwen3-0.6B ONNX variants (Path A and
Path B-mask), validate on CPU-ORT, transfer to `Z:\exposed\junk` for the
aarch64 (Snapdragon X2E) team to pick up.

This file is **local-only** (gitignored). Anyone resuming this work cold
can read it top-to-bottom and know exactly where the pipeline was left.
The authoritative plan is `docs/phase5_export_on_x86.md`; this file is
the session logbook on top of it.

## Plan summary

Build **two artifacts** in parallel, document dead ends:

| Artifact | Strategy | Hypothesis tested |
|---|---|---|
| `models/qwen3-0.6b-patha/` | Surgical fold, `ConstantOfShape(..., dtype=BOOL)` replaces Gather_5 | HTP rejects BOOL casts, not BOOL tensors |
| `models/qwen3-0.6b-pathbmask/` | Collapse BOOL region through `Where` → `Add(scores, attention_bias)`, FP16 additive mask as new runtime input, RoPE left in-graph | Adopting Qualcomm's mask pattern alone is enough |

Full Path B (RoPE externalization too) is out of scope for this cycle
unless both A and B-mask fail. See `docs/phase5_export_on_x86.md` §B for
the long form.

## Shared pipeline (both artifacts)

1. Install toolchain in `.venv-x86-export/`.
2. `optimum.exporters.onnx` → `models/qwen3-0.6b-optimum/`.
3. `scripts/rewrite_qwen3_htp.py` — part 1 applies steps 2a + 2b
   (attention_mask-as-initializer + IsNaN/Where elision), saves
   `models/qwen3-0.6b-staged/`.
4. `scripts/survey_bool_region.py` — recon: enumerate BOOL-tainted
   region, classify tensors, surface topology to operator before any
   fold is written.
5. Fork to A and B-mask.

## Progress log

### 2026-04-21 — session 1 kickoff

- Verified clean git state, on 7f0838b, up to date with origin.
- No system Python; using `uv 0.11.7` to manage a dedicated
  `.venv-x86-export/`.
- Added `.venv-*/` to `.gitignore`. `status_x86.md` itself IS tracked
  in git (serves as the handoff log).
- Created 9 tasks covering the pipeline end-to-end.
- Surveyed existing scripts. Key finding: `scripts/patch_gather5_dtypes.py`
  is **prior art for a weaker flavor of Path A** — it inserts Cast
  nodes around Gather_5 (`BOOL → INT8 → Gather → INT8 → BOOL`), but
  keeps BOOL tensors downstream. Our Path A goes further (replaces
  Gather_5 with `ConstantOfShape(Shape(idx), value=True, dtype=BOOL)`),
  so the research signal is distinct, not duplicated.
- Toolchain install completed cleanly; all pins match the doc's
  "versions known to work" row.

## Version pins (installed 2026-04-21)

- uv: 0.11.7
- python: 3.12.13 (`.venv-x86-export/`)
- optimum: 2.1.0 ✓
- optimum-onnx: 0.1.0 ✓
- torch: 2.11.0 (CPU-only on this Intel B50 Pro box) ✓
- transformers: 4.57.6 ✓ (optimum-onnx downgraded from 5.x, expected)
- onnx: 1.21.0 ✓
- onnxruntime: 1.24.4 (matches the X2E pin; CPU-only usage here)
- onnx-graphsurgeon: 0.6.1 (for the Path A/B folds)
- huggingface-hub: 0.36.2 ✓
- tokenizers: 0.22.2
- numpy: 2.4.4

## Resumption playbook (if another team picks this up cold)

1. Read `docs/phase5_export_on_x86.md` top-to-bottom. That's the master
   plan. The session-10 DO-NOT-RUN list applies:
   **do not** execute `scripts/simplify_qwen3_no_mask.py`,
   **do not** feed onnxsim the interior graph.
2. Read this file's **Progress log** to see the last finished step.
3. Check `git log --oneline -20` for per-step commits. Each major
   milestone (venv ready, export done, staged rewrites, each fold,
   each validation) lands as its own commit.
4. Unfinished scripts live under `scripts/`. Outputs under `models/`
   are gitignored — they're rebuildable from scripts.
5. Final deliverables target `Z:\exposed\junk\qwen3-0.6b-patha\` and
   `Z:\exposed\junk\qwen3-0.6b-pathbmask\` with the consolidated
   report `phase5_step6_export_report.md` alongside.

## Open questions forwarded to aarch64

Captured here so they make it into the final report:

1. Does the HTP compiler reject all BOOL tensors, or only Cast-to-BOOL?
   Path A tests this hypothesis directly.
2. Is the 2-Cast-to-BOOL count from session 10 reproducible in a fresh
   optimum 2.1.0 export, or do we see more (e.g. from Less/Greater in
   the causal-mask builder)? Recon will answer.
3. Qualcomm's X2E zoo bundles all use w4a16; our first binaries are
   fp16/fp16. ~3× perf left on the table; tracked as a follow-up pass
   after first-working-binary.

## Notes and deviations from the doc

- **Doc § "After 2a + 2b" histogram is stale.** Doc claims post-stage
  node count "around 2,129" and Cast→BOOL=2. On a fresh optimum 2.1.0
  export those numbers are actually **7,611 nodes and 3 Cast→BOOL**.
  The doc's 2,129 is almost certainly a confused carry-over from the
  quarantined onnxsim output in `qwen3-0.6b-nomask/`. 2a+2b are
  trivial protobuf edits — they only remove 28 `Where` + 28 `IsNaN`
  = 56 nodes. 7667 → 7611 is the correct math.
- **Extra Cast→BOOL node** beyond the doc's 2-count. Not a problem
  — recon will find it and classify it. Could be another node in the
  causal-mask builder the doc didn't inventory.
- **Probe regime chosen: past=511 zeros + BOS at position 511** rather
  than the doc's "zero-KV + BOS at position 0". Reason: staged graph
  pins attention_mask to length 512, so a past=0 probe puts source
  and staged into different mask-length regimes and the two are
  expected to diverge. past=511 matches the HTP decode regime and
  keeps source vs staged in the same shape space. Verified cos=1.0
  on this probe — which is the same bar the doc wanted.

## Recon findings (the actual BOOL topology)

Full dump in `results/phase5_step6_bool_region_dump.txt`. Summary:

**The 3 Cast→BOOL nodes are NOT what the doc implied.** Only one is a
"real" dtype conversion; the other two are BOOL→BOOL identity casts
that optimum inserted redundantly.

| Node | Input dtype | Output dtype | Role |
|---|---|---|---|
| `/model/Cast_2` | INT64 | BOOL | Real cast (`attention_mask` → BOOL) |
| `/model/Cast_5` | BOOL | BOOL | **Identity** (LessOrEqual is already BOOL) |
| `/model/Cast_6` | BOOL | BOOL | **Identity** (Reshape_1 is already BOOL) |

**The attention-mask BOOL region is shared, not per-layer.** One
subgraph produces a BOOL mask, fanned out via `Expand → 28 × Slice_4`.
Per-layer fork starts at `Slice_4`. This means a single subgraph
rewrite benefits all 28 layers — the doc's "per-attention-block"
framing was misleading.

**The mask is additive in the graph already, just via Where.**
Critical finding for Path B-mask: the Where the doc pointed at is
`/model/layers.N/self_attn/Where_2`, but its signature is NOT
`Where(bool_cond, scores, -inf)`. It's:

```
Where_2(
    cond=Slice_4_output_0 (BOOL),
    then=Constant_63 (FP32 scalar 0.0),
    else=Constant_64 (FP32 scalar -inf)
)
-> Add_2(MatMul_output_0 + Where_2_output_0)
```

The Where produces an **additive** FP32 mask (0 or -inf) that gets
Added to scores at `Add_2`. So Qualcomm's "production additive-mask
pattern" is already effectively what the graph computes — just via
a BOOL subgraph instead of a direct FP input. Path B-mask becomes
a much cleaner rewrite than the doc suggested: just splice a new
`attention_bias` FP32 input directly into the 28 `Add_2` nodes and
delete the Where/BOOL chain.

**Structural non-issues (separate from attention mask):**

- 58 `Equal` nodes — these feed `Where(Equal, ConstantOfShape_zero,
  Reshape_int64)` patterns for dynamic-shape resolution, not
  attention masking. Sink cleanly at INT64. Could still fail HTP
  op-lowering (BOOL-cond Where with INT64 branches is the pattern)
  but **not** in scope for this cycle's rewrites.
- 3 `Range`, 1 `LessOrEqual`, 1 `Cos`, 1 `Sin` — the position-aware
  computations (causal triangle, RoPE). Path B-mask removes Range +
  LessOrEqual by killing the causal-mask subgraph. Cos/Sin are
  RoPE; full Path B handles them but B-mask doesn't.

## Progress log additions

### 2026-04-21 — session 1 continued

- optimum export landed: `models/qwen3-0.6b-optimum/` (3.0 GB data,
  7667 nodes, opset 18, 0 com.microsoft — matches doc expectation
  exactly).
- `scripts/rewrite_qwen3_htp.py` written — single entry point with
  `--mode {stage, fold-patha, fold-pathbmask}`.
- Stage mode ran cleanly: 7667 → 7611 nodes (28 `Where(IsNaN, …)` +
  28 `IsNaN` elided), attention_mask promoted to `[1,512]` initializer
  all-ones. 3 Cast→BOOL remain (doc said 2; see deviation note).
- `scripts/probe_cos_vs_source.py` written — reusable probe for
  every candidate we emit.
- Probed staged vs optimum source on CPU-ORT:
  **cos=1.0, max_abs_diff=0.0, argmax match, top-5 5/5. PASS.**
  Report: `results/phase5_step6_probe_staged.json`. 2a+2b is proven
  safe; moving on to BOOL-region recon.
- Recon run — key findings captured in the "Recon findings" section
  above. Most consequential: the mask is already additive (Where_2
  produces FP32 0/-inf, added to scores at Add_2), which dramatically
  simplifies Path B-mask.
- **Path A implemented and validated.** 7,611 → 7,580 nodes. 0
  Cast→BOOL. Probe cos=1.0, max_abs_diff=0.0, argmax match. Report:
  `results/phase5_step6_probe_patha.json`.
- **Path B-mask implemented and validated.** 7,611 → 7,166 nodes.
  0 Cast→BOOL, 0 Range, 0 And, 0 LessOrEqual, 0 BOOL tensors anywhere
  (matches Qualcomm zoo pattern). Probe cos=1.0, max_abs_diff=0.0,
  argmax match. Report: `results/phase5_step6_probe_pathbmask.json`.
- Consolidated findings for the aarch64 team in
  `docs/phase5_step6_export_report.md`. That document is the handoff;
  this file remains the on-x86 operator logbook.
- **Next:** transfer to `Z:\exposed\junk\` per the user's request.
