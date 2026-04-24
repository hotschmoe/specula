# Repo hygiene — what we keep, what we archive, what we delete

Every research project accumulates artifacts faster than it consumes them.
The rules below keep the repo small enough to navigate and the signal
dense enough to be useful, without spending a session a month tidying up.

The goal is not minimalism. The goal is **future-us (or a collaborator)
lands in the repo and can find the answer in under two minutes.** If a
file is not serving that goal, it does not belong at top level.

## The three-bucket model

Every artifact falls into exactly one of:

| bucket | location | meaning |
|---|---|---|
| **keep** | top-level (`models/`, `results/csv/`, `results/reference/`, `docs/`) | actively referenced, or irreplaceable ground truth |
| **archive** | `archive/`, `docs/archive/` | historically valuable, no longer active |
| **marked_for_deletion** | `marked_for_deletion/` (gitignored) | staged for `rm -rf` once we've verified nothing critical slipped through |

**Markdown never gets hard-deleted.** It goes to `archive/` or
`docs/archive/`. Research prose is cheap to keep and expensive to
re-derive. Binary artifacts and raw logs get staged to
`marked_for_deletion/` and hard-deleted after a soak period.

## Category-specific rules

### `models/`

Keep:
- **Base GGUFs** for the active baseline set (Qwen3 sizes 0.6B / 1.7B /
  4B / 8B at their blessed quants). These are load-bearing for the
  baseline matrix.
- **One blessed vendor reference** per production target (currently:
  `qualcomm-qwen3-4b-ref/` — Qualcomm's shipping Genie bundle).
- **At most 1–2 "best-performing" NPU exports we produced ourselves**,
  with their wrappers + encodings JSON. Currently: Lever B baseline
  (`pathbmask.bin`) and Lever C full pass (`pathb.w8a16-local.bin`).
  Everything else we compiled is either intermediate, dominated, or a
  documented negative result — send it to `marked_for_deletion/`.

Stage for deletion:
- Intermediate ONNX export directories (`*-optimum`, `*-patha`,
  `*-pathbmask` staging, etc.). They are byproducts of running
  `scripts/export_*.py` and regenerate from source.
- Negative-result binaries. The *lesson* lives in a markdown file;
  the binary itself doesn't.
- Calibration `.npz` bundles. Manifests + capture scripts are tracked
  — regenerate on demand.
- Staging / AI-Hub intermediate dirs.

**Regeneratability is the binding test.** If `scripts/` + a manifest
rebuild the artifact, it does not need to live on disk.

### `results/`

Structure:
- `results/csv/` — all measurement data (benchmarks, sweeps). Never
  delete. CSV is the permanent record of *what the hardware did*.
- `results/reference/` — oracle `.md` + `.npz` pairs, probe summaries,
  environment snapshots, self-contained summary bundles. These are
  ground-truth inputs for future regression tests.

Stage for deletion:
- `.log`, `.stdout`, `.stderr` files — raw tool output. Their purpose
  is fulfilled once findings are extracted into an investigation
  markdown doc or a CSV row.
- Per-run output directories (one subdir per `llama-bench` invocation,
  per AI-Hub compile job, etc.). The CSV captures the numbers; the
  directory is chaff.
- Intermediate `.json` / `.txt` dumps that were probe output during a
  session. If a finding is still referenced, it's already in a `.md`.

**Pre-stage check (non-negotiable):** before moving a log to
`marked_for_deletion/`, confirm its key finding is captured in either
a CSV row or an investigation doc. If not, write it down first.

### `docs/`

Keep at top level:
- Active-priority docs driving the current phase (baseline methods,
  the pipeline-under-construction, relevant investigations).
- Reference anchors used across phases (`reference-projects.md`,
  version-pairing docs like `npu_ort_qnn_version_match.md`).
- The roadmap.

Move to `docs/archive/`:
- Closed phase-by-phase session docs (`phase5_step6_*`, etc.) once
  their phase is closed and the findings are rolled into a retro or
  the next phase's doc.
- Stalled / deferred investigations (parked B-items from the
  roadmap rolling backlog).
- One-off artifacts (upstream issue drafts, ask docs for external
  collaborators) once the ask is answered or the issue filed.

Never delete. Archived docs are how we remember what we tried and why
it was shelved.

## When to tidy

- **End of each workstream close-out** (W1, W2, …). The roadmap phase
  just closed — sweep its logs + intermediate artifacts before the
  next phase starts. Fresh memory makes the classification cheap.
- **When a directory exceeds a rough budget:**
  - `models/` > ~30 GB
  - `results/` > ~200 MB
  - `docs/` > ~20 active top-level files
- **Before adding a new ≥5 GB artifact.** If it's worth that much
  disk, it's worth 10 minutes of tidying first so it can be found
  later.

Skip tidying during active investigation — the cost of deleting a
file you actually needed is much higher than the cost of a messy
working dir. Tidy at the *boundary* between phases, not inside them.

## The `marked_for_deletion/` soak

Anything moved there stays for at least one pass of "am I missing
something?" confidence-building. Soak length is a judgment call:

- One focused session of using the repo after the move: fine to
  `rm -rf` if nothing tripped.
- Multi-week soak if the move was large (>50 GB) or covered an
  active investigation area.

`marked_for_deletion/` is gitignored, so it costs only local disk
during the soak. The reclaim target is `rm -rf marked_for_deletion/`
— do it deliberately, not accidentally.

## Writing new docs vs. editing existing

Before starting a new `docs/*.md`, check whether it fits into an
existing investigation doc. Creating a new file is free; the cost
comes six months later when three files cover overlapping terrain and
the reader can't tell which one is current.

Good reasons for a new doc:
- A new workstream with distinct scope.
- A reference/methodology that will be linked to from many places.
- A phase retro that closes a large investigation.

Bad reasons:
- "I have a new finding this session" — append to the existing
  investigation doc.
- "The existing doc is getting long" — long investigation docs are
  fine. Split only when subsections stop cross-referencing each other.

## Commit discipline during multi-session research

- Update investigation docs **incrementally** during the session, not
  at the end.
- Commit at natural milestones (a session's findings landing, a phase
  closing, a compile pipeline becoming stable) — not just at phase
  close.
- Never commit contents of `marked_for_deletion/`.

## TL;DR

1. Every artifact → keep / archive / marked_for_deletion. No fourth bucket.
2. Markdown never hard-deletes; binaries and logs do after a soak.
3. Before deleting a log, confirm its finding is in a `.md` or CSV.
4. Tidy between phases, not inside them.
5. CSVs in `results/csv/` and investigation prose in `docs/` are the
   project's permanent memory. Everything else is scratch.
