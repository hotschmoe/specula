# Qualcomm AI Hub compile — handoff status

Written 2026-04-20 at the end of session 8. One document to read cold
and be productive in 10 minutes on Phase 5 step 4 (getting a working
Hexagon v81 QNN context binary for Qwen3-0.6B).

**If you read one section, read §2 (current blocker + next action).**

## 1. Where this fits

Phase 5 of specula turns the Hexagon NPU into a draft model for
speculative decoding. Target: Qwen3-0.6B draft on NPU + Qwen3-8B CPU
target, compared against the Phase 2 CPU-spec baseline (40.2 t/s at
k=3). Scoping + toolchain pins live in `docs/npu_scoping.md`.

The 10-step bring-up (from that scoping doc):

```
[DONE]    1. Environment snapshot                 (commit 7230210)
[DONE]    2. ORT-QNN sidecar skeleton             (commit 282e84a)
[DONE]    3. Qwen3-0.6B ONNX sourced + CPU-valid  (commit 106c756)
[BLOCKED] 4. AI Hub compile -> Hexagon .bin      <-- YOU ARE HERE
          5. Load .bin via NPUSession, shape-check
          6. Correctness vs CPU, single greedy prompt
          7. Pipe first drafted token through llama.cpp verify
          8. External-drafter bridge for llama.cpp spec decode
          9. First NPU-spec number on 10-prompt humaneval
         10. Sweep k values, write up, close phase
```

Step 5+ plumbing is already in place (`scripts/npu_draft_sidecar.py`,
`NPUSession` class). They just need a working .bin to point at.

## 2. Current blocker + exactly what to do next

**Eight AI Hub compile attempts; last one failed at 465 s with HTP
rejecting a `Cast BOOL -> INT8` node we inserted as a workaround.
Root cause isn't that one node — HTP can't ingest BOOL tensors
anywhere.** Optimum's Qwen3 export uses BOOL throughout attention
masking (126 BOOL tensors, 56 `Where()` consumers across 28 layers).

**The fix is on the x86 machine, not here.** Every failed AI Hub log
contains this warning, repeated for 8 attempts:

```
Onnx model simplification failed due to: No module named 'onnxsim'
Simplified model validation failed
```

AI Hub's own preprocessing wanted `onnxsim` to fold the dynamic mask
subgraph (Range + Shape + Gather + Cast(BOOL)) to a pre-computed
static mask. Their environment doesn't ship it. On the X2E we can't
run it either — `onnxsim` has no Windows-on-ARM wheel.

**Action:** push one more round-trip through the x86 machine.
Instructions are in `docs/phase5_export_on_x86.md` under "Simplify
the graph with onnxsim" (added session 8). Hand-off output lands at
`models/qwen3-0.6b-simplified/` and is ~3 GB, transfer via cloud
drive or scp.

Once the simplified ONNX is on the X2E:

```powershell
# Edit paths in prep_onnx_for_ai_hub.py + compile_qwen3_ai_hub.py to
# point at qwen3-0.6b-simplified/ (currently they point at
# qwen3-0.6b-patched/ which was the dead-end surgical-fix chain).
# Then:
.venv\Scripts\python.exe scripts\prep_onnx_for_ai_hub.py
.venv\Scripts\python.exe scripts\compile_qwen3_ai_hub.py --check    # dry-run, free
.venv\Scripts\python.exe scripts\compile_qwen3_ai_hub.py --submit   # 20-30 min
```

If it succeeds, `models/qwen3_0_6b_draft_v81_ctx512.bin` lands on
disk and we move to step 5.

If it fails, capture the job log the same way previous ones were
captured (see §4 of this doc), diagnose, and iterate.

## 3. The 8-attempt history (compressed)

Each failure got deeper into QAIRT. Evidence logs are all committed
under `results/ai_hub_*/` and `results/ai_hub_compile_attempt*.log`.

| # | Source ONNX | Died at | Cause | Fix |
|---|-------------|---------|-------|------|
| 1 | single file upload | upload | "missing external weights" | upload directory not single file |
| 2 | dir with `.onnx_data` | 0 s (client) | qai-hub rejects extension | rename refs to `.data`, hardlink weights |
| 3 | onnx-community export | 100 s OPTIMIZE | `com.microsoft::SimplifiedLayerNorm` | swap to optimum `--no-post-process` (x86) |
| 4 | optimum, no flags | 130 s OPTIMIZE | int64 IO not accepted | add `--truncate_64bit_io` |
| 5 | +`--truncate_64bit_io` | 470 s CTX-BIN | HTP Gather dtype mismatch | try `--quantize_full_type float16` |
| 6 | +`--quantize_full_type float16` | 440 s CTX-BIN | same Gather; flag no-op | try local ORT optimizations |
| 7 | ORT BASIC (no fusion) | 440 s CTX-BIN | same Gather still | freeze dims + ORT BASIC + surgical Cast |
| 8 | Gather_5 Casts inserted | 465 s CTX-BIN | HTP rejects Cast BOOL -> INT8 | **need onnxsim** (this hand-off) |

Key lessons baked into the tooling, in order of discovery:

- qai-hub uploads need a directory with `.onnx`/`.data`/`.encodings`/`.bin` only.
- ONNX with external data has references on *both* `graph.initializer`
  tensors and `node.attribute.t` tensors (Constant nodes). The prep
  script walks both now.
- ORT 1.24 has a "hardlink attack" check that rejects external-data
  files with >1 hard link. Use `shutil.copy2`, not `os.link()`.
- `--truncate_64bit_io` only handles the IO boundary; internal int64
  tensors stay int64 and will trip ops that don't accept int64.
- ORT's `ORT_ENABLE_ALL` **fuses ops back into `com.microsoft`**
  (FusedMatMul, QuickGelu) — exactly what we just avoided. Use
  `ORT_ENABLE_BASIC` instead for constant folding without fusion.
- HTP has no BOOL support at all. Not in Gather, not in Cast. A BOOL
  tensor anywhere in the HTP-partitioned graph is a compile failure.
- AI Hub preprocessing expects `onnxsim`. Their env doesn't ship it
  and neither does Windows-on-ARM.

## 4. Tooling map (what each script does)

Chain of intermediate ONNX directories under `models/`:

```
qwen3-0.6b-onnx/              attempt 1-2: onnx-community export (fused ops). dead end.
qwen3-0.6b-optimum/           x86 optimum --no-post-process fp16. live source.
qwen3-0.6b-simplified/        <-- WAITING on x86 to produce this
qwen3-0.6b-optimum-frozen/    dim_params pinned to concrete values
qwen3-0.6b-optimum-ortopt/    ORT BASIC applied (no fusion)
qwen3-0.6b-optimum-frozen-ortopt/   freeze + BASIC (attempts 7-8)
qwen3-0.6b-patched/           + surgical Cast nodes around Gather_5 (attempt 8 source)
qwen3-0.6b-patched-ai-hub/    renamed to .data + copied weights, ready to upload
```

Once the x86-simplified source lands, most of these intermediates may
be unnecessary. Re-evaluate whether freeze / ort_optimize / patch
chain is still needed after onnxsim did the heavy lifting.

Scripts (all in `scripts/`):

- `npu_probe.py` — sanity check ORT-QNN + qai-hub auth. §5.3 / §5.4
  of npu_scoping.md. Runnable anytime.
- `npu_draft_sidecar.py` — NPUSession wrapper + tiny MatMul smoke
  test. Step 2 of the scoping doc. **The foundation for step 5+** —
  swap the model path from tiny-smoke.onnx to the compiled .bin and
  bind the 59 inputs.
- `download_qwen3_onnx.py` — pulls onnx-community Qwen3-0.6B-ONNX.
  Historical; no longer the source of truth (uses fused ops).
- `validate_qwen3_onnx_cpu.py` — greedy-decode 32 tokens via ORT-CPU
  to coherence-test an ONNX. Used at step 3 against the onnx-community
  export. Can be repointed at any of the derived ONNXs to spot-check.
- `inspect_onnx_ops.py --model PATH` — dumps op histogram and flags
  non-default-domain ops. Use it after every graph transformation
  to confirm we haven't accidentally reintroduced `com.microsoft`.
- `export_qwen3_qnn.py` — **DEPRECATED** (see file header). The
  `onnxruntime_genai.models.builder` path with
  `execution_provider="qnn"` doesn't actually work on PyPI 0.13.1.
  Kept for reference.
- `freeze_onnx_dims.py` — replace `batch_size` / `sequence_length` /
  `total_sequence_length` / `past_sequence_length` dim_params with
  concrete values from the decode-only compile plan (1, 1, 512, 511).
  Re-runs shape inference. Output: `qwen3-0.6b-optimum-frozen/`.
- `ort_optimize_onnx.py` — ORT graph optimizer at `ORT_ENABLE_BASIC`
  (no op fusion). Constant-folds Range/Shape chains. Output:
  `qwen3-0.6b-optimum-ortopt/`. Currently reads from the non-frozen
  source; edit paths when chaining.
- `patch_gather5_dtypes.py` — surgical Cast insertion around a
  specific Gather node to satisfy HTP dtype requirements. Generic
  mechanism — append to `PATCHES` list for other affected nodes.
  Output: `qwen3-0.6b-patched/`. **Likely unneeded** after onnxsim
  folds the mask subgraph that contains Gather_5.
- `prep_onnx_for_ai_hub.py` — patch external_data refs from
  `model.onnx_data` to `model.data` in BOTH initializers AND Constant
  node tensor attributes, copy weights, verify ORT-CPU loads the
  staged model. Output: `qwen3-0.6b-*-ai-hub/`. **Paths hardcoded at
  top; edit to repoint at the simplified source.**
- `compile_qwen3_ai_hub.py` — upload to AI Hub + submit compile job
  + poll + download .bin. Two modes: `--check` (dry-run, no upload,
  validates specs, prints plan) and `--submit`. Supports
  `--reuse-upload MODEL_ID` to skip the 15-minute upload when only
  compile options changed. Options currently include
  `--truncate_64bit_io` and `--quantize_full_type float16`.

## 5. Env state on this machine (X2E)

Assumed fresh after `uv sync --extra npu`:

- QAIRT 2.45.40.260406 at `C:\Qualcomm\AIStack\QAIRT\2.45.40.260406\`
- onnxruntime-qnn 1.24.4 (ships signed QAIRT v81 stack internally)
- qai-hub 0.48.0 (API token from prior voice_project setup)
- Hexagon v81 confirmed on this SoC (X2E94100, CRD08480)
- Python 3.12 ARM64 venv at `.venv/`

Inventory snapshot is in `results/npu_env_snapshot.txt`. Re-run
`scripts/npu_probe.py` any time you need to reconfirm.

Critical platform quirks (WoA-specific):

- **torch has no cp312 win_arm64 wheel.** `pip install torch` fails.
  Any HF-to-ONNX tracing path needs a non-WoA machine.
- **onnxsim has no win_arm64 wheel at all.** Graph simplification
  has to happen on the x86 side.
- **ORT 1.24 rejects external-data files with >1 hard links** as a
  "hardlink attack." prep_onnx_for_ai_hub.py uses copy, not link.

## 6. Gotchas worth remembering

- Qwen3-0.6B is not in the Qualcomm AI Hub Model Zoo. Qwen3-4B is
  (w4a16/Genie for X2E), but 4B is too big for a draft slot.
  Llama-3.2-1B is also there for X2E (90 t/s at w4a16) but tokenizer
  mismatch vs our Qwen3-8B target. Full zoo survey in
  `results/ai_hub_model_zoo_check.md`.
- Genie is Qualcomm's LLM runtime, alternative to ORT-QNN.
  Pre-compiled zoo assets target Genie, not ORT-QNN directly. A
  Phase 5.5 pivot to Genie is in scope if ORT-QNN hits a wall, but
  out of scope for step 4 as specified.
- w4a16 (int4 weights, fp16 activations) is Hexagon v81's HMX native
  fast path. Our current export is fp16/fp16 — correctness-first,
  but later passes should migrate to w4a16 for ~3x speedup. Requires
  calibration data (use `prompts/humaneval_subset.jsonl` +
  `prompts/structured_json.jsonl`). Follow-up, not blocker.
- `onnxruntime-genai` with `execution_provider="qnn"` LOOKS like it
  should work but silently doesn't (attempts pre-session-7 burned on
  this). Deprecation note is in `scripts/export_qwen3_qnn.py`.
- Input specs for the compile pin decode-only shapes:
  `seq_len=1, past_seq_len=511, total_seq_len=512`. Prefill stays on
  CPU per scoping doc §6.4. CONTEXT_MAX=512 for fast iteration; bump
  to 2048 once we have a working pipeline.

## 7. Dependencies on the x86 operator

For a new agent coordinating with the x86 side, the contract is:

**x86 side produces:** `models/qwen3-0.6b-simplified/` containing:
- `model.onnx` (simplified graph, ~0.5-2 MB after constant folding)
- `model.onnx_data` (fp16 weights, ~3 GB)
- `config.json`, `tokenizer.json`, `merges.txt`, `vocab.json`,
  `tokenizer_config.json`, `special_tokens_map.json`,
  `chat_template.jinja`, `added_tokens.json`, `generation_config.json`
- Pre-transfer, verify on the x86 machine:
  `python scripts/inspect_onnx_ops.py --model models/qwen3-0.6b-simplified/model.onnx`
  should show zero `com.microsoft` ops, zero `Range` nodes, zero
  `Cast ... to BOOL`.

**Transfer path:** any. cp to a mounted cloud drive is what's been
used so far (`G:\Shared drives\MAIN\Junk\...`).

**x86 env that works (reference):** Intel Core Ultra 7 155H, Python
3.12 x86_64. Install:
`pip install optimum optimum-onnx torch transformers huggingface_hub onnx onnxsim`.
See `docs/phase5_export_on_x86.md` for full procedure.

## 8. File map / where to look

Start here:
- `qlcom_compile_status.md` (this file) — current state + next action
- `docs/phase5_export_on_x86.md` — what the x86 operator runs
- `docs/npu_scoping.md` — the 10-step plan, toolchain pins
- `current_status.md` — project-wide phase tracking (Phase 5 section)

Evidence from earlier sessions:
- `results/npu_env_snapshot.txt` — env inventory + session notes
- `results/ai_hub_model_zoo_check.md` — Qualcomm precompiled survey
- `results/ai_hub_compile_attempt{2,3,8}.log` + `attempts4-7_summary.log`
  — compile history
- `results/ai_hub_j*/*.log` — full AI Hub job logs with Hexagon
  compiler traces for the three rich failures (5, 7, 8)
- `results/tiny_npu_smoke.onnx` — step 2 smoke-test graph (gitignored,
  regenerated by `npu_draft_sidecar.py`)

Git log for step 4:

```
8a25c14 phase 5 step 4f: add onnxsim step to x86 doc; hit the BOOL wall
390646b phase 5 step 4e: chain ONNX passes to get past HTP op-dtype gauntlet
dc6a0fc phase 5 step 4d: point pipeline at optimum export + fix two prep bugs
7c2589c phase 5 step 4d: optimum --no-post-process produces clean QNN ONNX  (from x86)
bfef3bb phase 5 step 4c: stage x86 export handoff + AI Hub zoo check
07d3473 phase 5 step 4b: AI Hub compile hits op-lowering wall, plan pivot
0e799f2 phase 5 step 4a: stage ONNX + compile script (compile job in flight)
```

## 9. If onnxsim doesn't save us

Plan B paths, each previously discussed and ruled out for the main
plan but still valid as fallbacks:

1. **Manual BOOL eradication pass.** Replace every BOOL tensor with
   INT8, every `And` with `Mul`, every `Where(bool, ...)` with
   equivalent `Where(int8, ...)` or `score + log(mask)` additive
   masking. ~6-12 hours of careful ONNX surgery. Deterministic but
   high bug risk (must verify logits still match HF reference after
   rewrite).

2. **Pivot to Genie runtime.** Download Qualcomm's pre-compiled
   Qwen3-4B-w4a16 bundle (works on X2E today), run via Genie, treat
   as "hardware capability demo" even though size is wrong for spec
   decode. `results/ai_hub_model_zoo_check.md` has the download URL.

3. **Different draft model entirely.** Llama-3.2-1B-w4a16 is
   pre-compiled for X2E at 90 t/s. Tokenizer mismatch with Qwen3-8B
   target, so would require either a Llama-family target or a
   tokenizer-remapping layer in the bridge. Bigger scope shift.

4. **Train our own decomposed attention export.** Skip optimum
   entirely; write a custom PyTorch-to-ONNX tracer on the x86 side
   that emits standard-ops-only attention from the start. Biggest
   effort, fully controllable output.

Default order if onnxsim ALSO fails: try #2 as a quick "hardware
works" milestone while continuing to debug #1 or #4 offline. #3 is
independent and can run in parallel.

## 10. Quick-start commands (copy-paste)

Re-check env on this machine:

```powershell
.venv\Scripts\python.exe scripts\npu_probe.py
.venv\Scripts\python.exe scripts\npu_draft_sidecar.py   # tiny-graph HTP smoke test
```

After receiving the x86-simplified ONNX at `models/qwen3-0.6b-simplified/`:

```powershell
# 1. Verify it's clean
.venv\Scripts\python.exe scripts\inspect_onnx_ops.py `
    --model models\qwen3-0.6b-simplified\model.onnx

# 2. Edit paths in prep_onnx_for_ai_hub.py and compile_qwen3_ai_hub.py
#    (3 lines each; search for 'qwen3-0.6b-patched' and replace with
#    'qwen3-0.6b-simplified' and 'qwen3-0.6b-simplified-ai-hub')

# 3. Stage for upload
.venv\Scripts\python.exe scripts\prep_onnx_for_ai_hub.py

# 4. Dry-run (free)
.venv\Scripts\python.exe scripts\compile_qwen3_ai_hub.py --check

# 5. Actual submit + download (~20-30 min)
.venv\Scripts\python.exe scripts\compile_qwen3_ai_hub.py --submit
```

On a compile failure, pull the AI Hub job log with:

```powershell
.venv\Scripts\python.exe -c "import qai_hub as hub; print(hub.get_job('<JOB_ID>').download_job_logs('results/ai_hub_<JOB_ID>/'))"
```

then grep for `ERROR` lines to see what QAIRT hit.

## 11. Success criterion

Step 4 closes when `models/qwen3_0_6b_draft_v81_ctx512.bin` exists on
disk and its size is somewhere in 500-1500 MB (fp16 HTP context
binary for a 0.6B model at ctx 512 should be in that ballpark).
Opening it requires step 5 — either load it via `NPUSession` (likely
needs a small extension for context-binary loading rather than ONNX
loading) or via AI Hub's test-job path for on-cloud device execution.

At that point the scoping doc's bring-up plan resumes at step 5.
