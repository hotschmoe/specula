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
- Transferred to `Z:\exposed\junk\phase5_step6\` for the aarch64
  team to pick up. MD5 verified end-to-end on both `model.onnx_data`
  sidecars:
    - patha: `214f4237ecaa0db8de26b6f440f88b40`
    - pathbmask: `86658b4d5b573d07db4fc2db1d4a31a7`
  Transfer bundle contents:
    - `qwen3-0.6b-patha/` (full artifact: ONNX + weights + sidecars)
    - `qwen3-0.6b-pathbmask/` (full artifact: ONNX + weights + sidecars)
    - `phase5_step6_export_report.md` (the handoff document — this
      is what they should read first)
    - `status_x86_snapshot.md` (a copy of this file at transfer time)
    - `phase5_step6_probe_{staged,patha,pathbmask}.json` (CPU-ORT
      cos probe evidence)
    - `phase5_step6_bool_region.json` + `_dump.txt` (recon output;
      useful for understanding the rewrite choices)

## Session 1 complete

All 9 tasks closed. Pipeline produced both target artifacts with
cos=1.0 bit-identical to the optimum source and transferred cleanly
to the NAS. Two independent HTP compile attempts are now possible
with guaranteed-correct ONNX inputs — if either .bin comes back
silently wrong on the aarch64 decode harness, the fault is in
HTP/QNN numerical fidelity, not in our ONNX rewrites.

Last git state on master:
```
62a622a phase5 step 6: both folds shipped, cos=1.0 both paths
0187103 phase5 step 6: recon the BOOL-tainted region on staged ONNX
ca53d3e phase5 step 6: safe rewrites (2a+2b) + cos probe, verified cos=1.0
467d485 phase5 step 6 x86: scaffold venv + handoff log
7f0838b phase 5 step 6 DIAGNOSIS: nomask ONNX is broken, step 4 needs redo
```

### 2026-04-22 — session 2: Path B (rotary hoist) for w4a16

**Trigger:** aarch64 team's two w4a16 compile attempts on `pathbmask`
both failed at QNN op-validation with `/model/rotary_emb/MatMul has
incorrect Value 0, expected equal to -32768` (INT16_MIN). Root cause
diagnosed by inspecting Qualcomm's shipping Qwen3-4B w4a16 bundle:
they hoist rotary_emb out of the compiled graph and feed
`position_ids_cos`/`position_ids_sin` as top-level inputs. We must
match. Per `docs/phase5_export_on_x86.md` §"Path B implementation
contract (2026-04-22 revision)".

**Artifact produced:** `models/qwen3-0.6b-pathb/`
- 61 graph inputs (was 59 in pathbmask): `position_ids_cos` and
  `position_ids_sin` appended after `attention_bias`, shape
  `['batch_size', 'sequence_length', 128]` float32 each.
- 7,131 nodes (was 7,166). All ops in default ONNX domain.
- **Zero `/model/rotary_emb/*` nodes remaining.**

**Surgery seam:** rewired layer-0 `Unsqueeze_6/7.input[0]` directly
to the new graph inputs (one node downstream of the doc-spec seam
at `Cast_4.input[0]`). Result: the entire upstream rotary chain —
including `Mul × Constant_7` (=1.0 for Qwen3-0.6B, identity) and
`Cast_4/5` (FP32→FP32, identity) — pruned cleanly. The two layer-0
Unsqueezes still broadcast cos/sin to all 56 layer-side Mul
consumers, none of which were touched. Net effect numerically
identical to the doc-spec seam for Qwen3-0.6B; for Qwen3.5 the
script asserts on `Constant_7 == 1.0` and would refuse to run,
prompting a fold-into-runtime fix.

**Shape deviation from doc:** doc spec said `[1,1,1,128]` 4D; we
shipped `[batch_size, sequence_length, 128]` 3D. The doc's 4D was
based on Qualcomm's metadata convention for a different seam
(post-layer-Unsqueeze). Our seam is pre-layer-Unsqueeze, so 3D is
correct for the actual graph point. The X2E `build_input_specs`
will read the actual shape from the graph regardless.

**Scripts landed:**
- `scripts/rewrite_qwen3_pathb.py` — pure protobuf rewrite
  (no onnxsim, no graphsurgeon). Asserts Constant_7==Constant_8==1.0
  before proceeding. Iterative dead-node prune.
- `scripts/probe_pathb_equivalence.py` — CPU-ORT gate probe.

**Gate evidence:**

| probe | cosine | argmax match | top-5 overlap |
|---|---:|:-:|:-:|
| pos=0, BOS, zero KV | **1.000000** | ✓ | 5/5 |
| pos=5, synthetic past_kv | **1.000000** | ✓ | 5/5 |

Numerically exact (not just within tolerance) — every dropped node
was provably identity for this model.

**Handoff bundle: `Z:\exposed\junk\phase5_step12_pathb\`**
- `qwen3-0.6b-pathb/` — full artifact (model.onnx + model.data + sidecars)
  - model.data MD5: `86658b4d5b573d07db4fc2db1d4a31a7` (same as
    pathbmask — weights unchanged, only graph topology)
  - model.onnx MD5: `3507c698ac81899b228c6b3aee412c1c`

**Runtime cos/sin formula** (canonical — for X2E `npu_load_qwen3_bin.py`
and calibration capture):

```python
def rope_tables(position_id, head_dim=128, rope_theta=1_000_000.0):
    inv_freq = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    freqs = position_id * inv_freq
    emb = np.concatenate([freqs, freqs], axis=-1)
    cos = np.cos(emb)[None, None, :].astype(np.float32)   # [1, 1, 128]
    sin = np.sin(emb)[None, None, :].astype(np.float32)   # [1, 1, 128]
    return cos, sin
```

`rope_theta=1e6` per `models/qwen3-0.6b-optimum/config.json`.
attention_scaling = 1.0 (no extra multiplier needed).
