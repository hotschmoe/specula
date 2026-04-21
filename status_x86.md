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

_(none yet — update as encountered)_
