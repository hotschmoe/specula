# CLAUDE.md — session orientation

Short brief for any agent dropping into this repo. Point to the right
doc, don't duplicate it here.

## Read first, in order

1. **`docs/roadmap.md`** — the strategic plan. Workstreams, dependency
   graph, success criteria, and what's next. Every non-trivial piece
   of work should trace back to a line in here.
2. **`docs/repo_hygiene.md`** — the rules for what to keep, archive,
   or stage for deletion. Follow this every time a file is added or a
   phase closes so the repo stays navigable.
3. **`README.md`** — human-facing project overview (hardware context,
   phase history, directory layout).

## Current priority path (2026-04)

In order:

1. **`docs/qwen3_4b_baseline_methods.md`** + **`qwen3_4b_baseline_all_backends.md`**
   — fill the all-backends baseline matrix for Qwen3-4B using the
   Qualcomm Genie bundle and the HF GGUF.
2. **`docs/one_pipeline_cloud_gpu.md`** — one reusable cloud-GPU
   conversion pipeline (HF → NPU bundle) driven by AIMET SEQ_MSE +
   AdaScale. Operational runbook lives in `docs/rent_cloud_compute.md`.
3. **Roadmap workstreams** (CPU / GPU / heterogeneous async): W1, W2,
   W4 per `docs/roadmap.md`. Don't pivot onto these until the two
   above produce data.

## Active investigation docs

Append new findings to existing docs where possible (see
`repo_hygiene.md` §"Writing new docs vs. editing existing"):

- `docs/w4a16_investigation.md` / `_continued.md` — NPU draft quant
- `docs/qwen3_perf_levers_investigation.md` — Phase 5.5 A/B/C
- `docs/qualcomm_reproduction_4b.md` — the 4B reference anchor
- `docs/npu_results.md` — Phase 5 close writeup

## Working rules (quick reference)

- **Commit at milestones**, not only at phase close. Update
  investigation docs incrementally during the session.
- **Before deleting a log**, confirm its finding is already in a
  markdown doc or a CSV row in `results/csv/`.
- **Markdown is never hard-deleted.** Archive to `docs/archive/` or
  the root `archive/` instead.
- **Heavy or dead artifacts** (large binaries, intermediate export
  dirs, raw logs) go to `marked_for_deletion/` — never `rm` directly.
- **Regeneratable artifacts** (calibration bundles, intermediate
  ONNX exports, compile outputs) don't need to live on disk if the
  generating script + manifest are tracked.
- **Tidy between phases, not inside them** — see
  `docs/repo_hygiene.md` §"When to tidy" for budget triggers.

## Hardware + environment quick facts

- Snapdragon X2 Elite Extreme laptop. 48 GB LPDDR5X unified @ 228 GB/s.
  Adreno X2 GPU, Hexagon NPU. Windows 11 ARM64 native (no WSL for
  GPU/NPU).
- ORT-QNN version must match AI Hub's QAIRT — see
  `docs/npu_ort_qnn_version_match.md`. 1.24.4 ↔ QAIRT 2.42.
- Production target is Qwen3.5 → Qwen3.6 → Gemma4. Qwen3 is the
  current scaffolding for literature comparability.
