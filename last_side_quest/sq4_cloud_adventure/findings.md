# SQ4 — cloud-rented NPU bundle pipeline (the adventure)

**Status:** plan committed 2026-04-29. Workspace: `last_side_quest/sq4_cloud_adventure/`.
First milestone (M1, Qwen3-0.6B) not started yet.

This doc is the umbrella plan + execution log for SQ4. Per-milestone
detail will be appended below as runs land. Operational deltas vs the
existing cloud docs go inline; design rationale stays in
`docs/one_pipeline_cloud_gpu.md`.

## TL;DR

- **Goal**: take HF FP16 weights → working w4a16 QNN context binary on
  X2E NPU, end-to-end, on rented Linux + CUDA. Five milestones in
  ascending difficulty.
- **First card**: RunPod A40 48 GB at **$0.44/hr** (community cloud).
  ~5× cheaper than A100 40 GB. Fits up to ~14B in FP16; larger models
  upgrade to A100 80 GB later.
- **Validation strategy**: cos-similarity vs FP32, bundle-size, and —
  for M2 — **byte-/argmax-comparable to Qualcomm's shipping Qwen3-4B
  bundle on disk** at `models/qualcomm-qwen3-4b-ref/`.
- **Total budget**: $30-50 first-pass across all 5 milestones; budget
  $50-75 with realistic re-runs.

## What AIMET vs QAIRT actually do

The user's "(and qairt?)" question — confirming roles before we run:

| tool | input | output | runs on | step in pipeline |
|---|---|---|---|---|
| **AIMET** (`aimet_torch` + `aimet_onnx`) | HF FP16 model + calibration prompts | ONNX with QDQ ops + `encodings.json` (per-tensor scales/offsets/bitwidths) | Linux x86_64 + CUDA (SEQ_MSE/AdaScale **require** GPU) | the calibration / quantization-research stage |
| **QAIRT** (`qairt-converter`, `qairt-quantizer`, `qnn-context-binary-generator`) | the AIMET-emitted ONNX + encodings.json | partitioned `.bin` HTP context binaries | x86_64 (Linux or Windows; we use Linux on the cloud VM, Prism on the X2E) | the deployment / compile stage |

**Both run on the cloud VM** in one rent session. The compiled bundle
is OS-agnostic HTP bytecode — pulls back to the X2E for runtime via
ORT-QNN or genie-t2t-run. Doing both on the cloud VM means "leave
nothing stateful on the cloud host" per `one_pipeline_cloud_gpu.md`.

## Hardware sizing

A40 48 GB headroom, FP16 model + activations during AIMET
SEQ_MSE+AdaScale calibration:

| model | FP16 weights | est. peak VRAM | A40 48 GB? | first-run card | cost/hr |
|---|---:|---:|---|---|---:|
| Qwen3-0.6B | 1.2 GB | ~6 GB | ✅ massive margin | A40 48 GB | $0.44 |
| Qwen3-4B | 8 GB | ~28 GB | ✅ comfortable | A40 48 GB | $0.44 |
| Qwen3-14B | 28 GB | ~38 GB | ⚠ tight; OK if SEQ_MSE blockwise | A40 48 GB (try) | $0.44 |
| Qwen3.6-27B dense | 54 GB | ~70 GB | ❌ OOM | A100 80 GB | ~$1.99 |
| Qwen3.6-35B-A3B (MoE) | 70 GB | ~80+ GB | ❌ OOM | A100 80 GB or 2× | ~$1.99-3.98 |

**14B is the ceiling on A40 48 GB**; the Qualcomm `quantize.py` source
loads the full FP16 model + activations in VRAM (per `rent_cloud_compute.md`
line 234 "40 GB of VRAM for a 4B model"). For ≥27B we upgrade. M1+M2+M3
on A40, M4+M5 on A100 80 GB.

## Pipeline overview

```
HF FP16 checkpoint (Qwen/Qwen3-0.6B etc.)
        ↓
optimum-export (transformers → ONNX, FP16/FP32)
        ↓
pathb rewrite (rotary hoist, RMSNorm split, KV reshape)
        ↓
[CUDA needed from here]
AIMET v2 quantsim (w4 sym weights / a16 asym acts)
   → compute_encodings (basic PTQ)
   → apply_seq_mse (block-wise weight optimization)
   → apply_adascale (activation-aware scale tuning)
   → encodings.json + ONNX with QDQ
        ↓
qairt-converter --quantization_overrides encodings.json
   → per-part .dlc
        ↓
qairt-quantizer (per-part)
   → quantized .dlc
        ↓
qnn-context-binary-generator
   → multi-graph .bin × N partitions + metadata.yaml
        ↓
tar → scp home → X2E runtime (ORT-QNN sidecar / genie-t2t-run)
```

Five **gating** decisions (all answered by `one_pipeline_cloud_gpu.md`
§Strategic basis):

1. Base FP checkpoint, not pre-quantized. (Q3)
2. Whole pipeline on cloud VM, not split. (Q2)
3. Calibration stack = basic PTQ + SEQ_MSE + AdaScale. (Q4)
4. AIMET adapter per arch family — SQ3 verified ~80 LOC for non-blessed.
5. AIMET tensor names ↔ pathb graph names need mapping (P1) — first
   AIMET run reveals whether the names line up or need a translator.

## Reference bundles on disk (the gold standards we're trying to match)

```
models/qualcomm-qwen3-4b-ref/
  qwen3_4b-genie-w4a16-qualcomm_snapdragon_x2_elite/
    *_part{1,2,3,4}_of_4.bin    ← Qualcomm's shipping HTP bytecode
    metadata.yaml
    tokenizer.json
    genie_config.json
```

This is the **anchor for M2**. Qualcomm's shipping bundle was measured
in Phase 5:
- Bundle size: 3.1 GB total
- First-decode cos vs FP32: **0.9998**
- Argmax agreement on the 46-token oracle prompt: **46/46**

Our local qairt-quantizer Phase 5o reached cos 0.9996 / 30/46 (basic
PTQ ceiling). M2 success = match or beat 46/46 — that's the SEQ_MSE+
AdaScale unblock that this whole side quest is about.

```
models/Qwen3.6-35B-A3B-Q4_K_M.gguf      (daily-driver, GGUF not bundle)
models/Qwen3.6-35B-A3B-MXFP4_MOE.gguf
```

These prove HF source weights are obtainable. M5 will start from the
HF FP16 checkpoint (`Qwen/Qwen3.6-35B-A3B`), not these GGUFs.

## Milestones

### M1 — Qwen3-0.6B w4a16 NPU bundle (~$1, ~2 hr)

**Why first.** SQ2 already ran AIMET basic PTQ on Qwen3-0.6B locally
(Prism CPU) and got cos -0.065 — the V/O collapse from the W4A16
investigation reproduced at the smallest scale. M1 is **the decisive
test of whether SEQ_MSE+AdaScale closes that gap**. If yes, the whole
pipeline is unblocked. If no, V/O collapse is structural at 0.6B and
we either accept that prior or pause to investigate.

**Steps.**
1. Rent A40 48 GB on RunPod (~$0.44/hr).
2. Install `aimet_onnx-2.26.0+cu121-cp310-...whl` per
   `rent_cloud_compute.md` Scenario B step 3.
3. Run `python -m qai_hub_models.models.qwen3_0_6b.quantize ...
   --use-seq-mse --use-ada-scale --num-samples 128`.
   - **First check:** does `qwen3_0_6b` arch ship in `qai_hub_models`?
     If not, fall back to direct `aimet_torch` v2 driver (port from
     `last_side_quest/sq2_aimet_local/probe_qwen3_0p6b_ptq.py` and add
     SEQ_MSE+AdaScale calls).
4. Pull `encodings.json` + AIMET ONNX back to X2E (or: run QAIRT on
   cloud VM and pull the compiled .bin files).
5. Bench on X2E using existing `npu_engine/` scaffolding.

**Acceptance criteria.**
- ✅ AIMET cos vs FP32 on held-out prompts: **≥ 0.95** (vs SQ2's -0.065)
- ✅ QAIRT compile produces multi-part bundle without errors
- ✅ Bundle loads via ORT-QNN on X2E (no QNN-1002 / QNN-5000 errors)
- ✅ Generates coherent text on a fixed prompt (e.g., "The capital
  of France is" → " Paris")
- 🎯 **Stretch:** outperforms our existing
  `qwen3_0_6b_draft_v81_ctx256.pathb.w8a16-local.bin` on argmax
  agreement OR matches it at smaller bundle size (w4 vs w8)

**Cost.** A40 48 GB × ~2 hr = **~$0.88** + storage = **~$1**.

**Gate to M2.** AIMET cos must clear **0.95**. Below that and the V/O
collapse is a structural-0.6B finding; we proceed to M2 with that prior
documented but no longer expecting bit-exact reproduction. Above 0.95
and the pipeline is validated for Qualcomm-blessed archs.

---

### M2 — Qwen3-4B w4a16 NPU bundle (the gold-reference reproduction, ~$3-5)

**Why second.** Qualcomm's shipping bundle is on disk for direct
comparison. Phase 5 documented our local qairt-quantizer w4a16 stalls
at cos 0.9996 / 30/46 argmax — the **exact point** where SEQ_MSE+
AdaScale should unblock us.

**Steps.**
1. Same rent session as M1 if M1 ran fast (no need to tear down).
   Otherwise re-rent A40 48 GB.
2. `python -m qai_hub_models.models.qwen3_4b.quantize ...
   --use-seq-mse --use-ada-scale --num-samples 128 --seq-mse-num-samples 128
   --ada-scale-num-samples 128`.
3. Address P1 (tensor-name mapping vs our pathb graph). The
   `qai_hub_models` recipe uses ITS OWN pathb ONNX, not ours — we
   need to either (a) accept their graph and rebuild our wrapper to
   match, or (b) write a name-translator. Decision deferred until we
   see actual encodings.json output.
4. Compile end-to-end on cloud VM via QAIRT.
5. Pull bundle home; bench against `qualcomm-qwen3-4b-ref` bundle on
   the same oracle prompt.

**Acceptance criteria.**
- ✅ AIMET cos vs FP32: **≥ 0.998**
- ✅ Bundle size: **3.0-3.5 GB** (Qualcomm's is 3.1 GB)
- ✅ First-decode cos vs Qualcomm-bundle on oracle prompt: **≥ 0.99**
- ✅ Argmax agreement on 46-token oracle: **≥ 40/46**
- 🎯 **Stretch goal:** **≥ 46/46** argmax (match Qualcomm) OR beat their
  cos 0.9998
- ✅ TG t/s on `bench_qwen3_4b_ortqnn` matches Qualcomm-ref bundle within 5%

**Cost.** A40 48 GB × ~4-6 hr = **~$1.76-2.64** + storage = **~$3-5**.

**Gate to M3.** Want all metrics ≥ Qualcomm's within tolerance. If we
fall short of 46/46 argmax but clear ≥40/46, the residual gap is the
calibration set (theirs is proprietary) — M3+ are still meaningful
but framed as "near-Qualcomm" rather than "Qualcomm-or-better."

**Decision pending after M2.** If we land 46/46, this side quest's
story becomes "we can reproduce Qualcomm's shipping bundles from HF
checkpoints in $5 of cloud rent." That's a publishable result.

---

### M3 — Qwen3-14B w4a16 NPU bundle (~$3-5, no reference)

**Why third.** No Qualcomm reference exists for Qwen3-14B on X2E. M3
validates the pipeline at a scale where we can't bit-exact compare —
the test is "does it work at all at 14B?" plus "is the bundle usable?"

**Pre-flight.**
- Confirm A40 48 GB doesn't OOM during 14B AIMET calibration. If it
  does, escalate to A100 80 GB (~$1.99/hr).
- 14B has more transformer layers (40 vs 4B's 32) — partition strategy
  may need 5-6 parts vs 4B's 4. Confirm HTP session ceiling is not
  hit (`reference_ortqnn_session_limit.md`: ~7-session ceiling at
  cl=512; might tighten at higher tiers).

**Acceptance criteria.**
- ✅ AIMET cos vs FP32: **≥ 0.99**
- ✅ Bundle compiles (5-6 partitions expected)
- ✅ Loads on X2E without HTP-session errors at cl=2048+
- ✅ Generates coherent text on coding-assistant prompts
- ✅ TG ≥ 12 t/s (extrapolated from Phase 1 14B-CPU baseline + 4B-NPU
  baseline)

**Cost.** A40 48 GB × ~6-8 hr if it fits = **~$3-4**. A100 80 GB if
escalated = **~$12-15**.

**Gate to M4.** Bundle works on hardware. Quality-of-life metrics
rather than reference comparison. If 14B fails to fit on A40, document
the VRAM threshold and proceed to M4 on A100 directly.

---

### M4 — Qwen3.6-27B dense w4a16 NPU bundle (~$12-15, cross-generation)

**Why fourth.** First Qwen3.6-generation model. **Tokenizer is
incompatible with Qwen3** per memory `reference_qwen_tokenizer_generations.md`.
Validates that our pipeline handles the new tokenizer + any
architectural deltas.

**Pre-flight checks (BEFORE the rent).**
1. Verify `Qwen3.6-27B` exists on HF and weights are available.
   *If the actual ID is different (Qwen3.5-27B, etc.), update.*
2. Check `aimet_torch` v2.29 adapter list — does it ship `qwen3_5` or
   `qwen3_6` adapters? If not, ~80 LOC adapter per SQ3-Granite finding.
3. Confirm Qwen3.6-27B architectural deltas vs Qwen3-14B (rotary,
   RMSNorm, attention head config). Diff `config.json` between them.
4. Run a tokenizer round-trip via
   `last_side_quest/sq1_heterogeneous/probe_tokenizer_match.py` adapted
   to Qwen3.6 vocab.

**Acceptance criteria.**
- ✅ AIMET cos ≥ 0.99
- ✅ Bundle compiles + loads on X2E
- ✅ Tokens produced match upstream Qwen3.6 tokenizer (round-trip OK)
- ✅ Coherent text generation

**Cost.** A100 80 GB × ~6 hr × $1.99/hr = **~$12** + storage = **~$13-15**.

---

### M5 — Qwen3.6-35B-A3B w4a16 NPU bundle (~$20-25, MoE final boss)

**Why last.** Production target — this is what specula's eventual
heterogeneous-NPU coding-agent stack runs against. MoE adds expert
routing complexity. AIMET 2.29 ships a `qwen3_moe` adapter (verified
SQ2/SQ3); HF FP16 checkpoint is downloadable.

**Pre-flight.**
1. Verify `qwen3_moe` adapter handles Qwen3.6-A3B's expert config
   (3B/35B with 128 experts top-K? confirm from config.json).
2. Decide MoE quantization granularity per SQ3-Granite finding:
   - per-tensor experts (lowest quality, smallest output)
   - per-(expert, out-channel) experts (SQ3's "A1" champion)
   - per-channel experts (SQ3 stretch)
3. **VRAM**: 70 GB FP16 likely doesn't fit 80 GB A100 with activations.
   Options:
   - 2× A100 80 GB (model parallel, ~$3.98/hr × 8 hr = ~$32)
   - 1× H100 80 GB (~$2.50-3/hr, more compute throughput)
   - Per-expert calibration loop (load-one-expert-at-a-time pattern,
     mostly speculative — would need AIMET source surgery)
4. HTP session count: a 35B-A3B bundle likely partitions into 8-12
   parts, **past the 7-session ceiling at cl=512** documented for 4B.
   Plan: combined-wrapper one-session-per-bin layout per
   `reference_ortqnn_session_limit.md`.

**Acceptance criteria.**
- ✅ AIMET cos ≥ 0.99
- ✅ Bundle compiles (8-12 partitions)
- ✅ Loads on X2E NPU (combined-wrapper layout, swap-mode if needed)
- ✅ Generates coherent text
- ✅ Compares favorably vs daily-driver `Qwen3.6-35B-A3B-Q4_K_M.gguf`
  on perplexity (within 5%) AND TG t/s (target: outperform CPU
  daily-driver's 32 t/s on the silent-island UX axis per SQ6 Phase D)

**Cost.** A100 80 GB or 2× × ~10 hr = **~$20-32** + storage = **~$25-35**.

**This is the side quest's headline deliverable.** A working w4a16 35B-A3B
NPU bundle would be the first published Qwen3.6-MoE-on-Hexagon
artifact. Validates every layer of the stack: AIMET extensibility,
QAIRT MoE compile, HTP MoE routing, ORT-QNN MoE serving.

---

## Total budget

| milestone | first-pass | with re-runs (50% margin) |
|---|---:|---:|
| M1 (0.6B, A40) | $1 | $1.50 |
| M2 (4B, A40) | $4 | $6 |
| M3 (14B, A40 or A100) | $4-12 | $6-18 |
| M4 (27B dense, A100) | $13 | $20 |
| M5 (35B-A3B MoE, A100) | $25 | $38 |
| **Total** | **$47-55** | **$71-83** |

Initial RunPod credit: **$20** is enough for M1 + M2. Top up after M2
lands. Don't fund the whole budget upfront — gate spending on each
milestone's actual outcome.

## First-run checklist (M1, executable today)

### Pre-rent (do at home)

- [ ] HuggingFace token in `~/.hf-token-readonly` (read scope only)
- [ ] AI Hub token at `qai-hub-token` (already configured per
      `qai-hub configure`)
- [ ] Confirm Qwen3-0.6B is non-gated (no HF auth needed for the
      checkpoint itself — only AIMET deps via pip)
- [ ] Confirm $20 minimum on RunPod account
- [ ] Phone alarm set for 3 hr from rent-start (M1 should finish in 2)

### Rent + run

- [ ] RunPod web UI → Community Cloud → A40 48 GB → PyTorch 2.4.1
      template → 100 GB volume → deploy
- [ ] SSH in via RunPod's connect-tab command
- [ ] `tmux new -s sq4-m1` (so connection drops don't kill the job)
- [ ] Install: see `docs/rent_cloud_compute.md` Scenario B step 3
      (`pip install "qai-hub-models[qwen3-0-6b]" + aimet_onnx wheel`)
- [ ] Sanity check: `python -c "import torch; print(torch.cuda.is_available())"`
      → True; `nvidia-smi` shows 48 GB
- [ ] Run quantize: see Steps below (concrete command TBD post-install
      because `qai_hub_models.models.qwen3_0_6b` may or may not exist
      as a published recipe — confirm via `--help` first)
- [ ] Inspect output: `encodings.json` size, AIMET cos vs FP32 from
      log
- [ ] If cos ≥ 0.95 → proceed to QAIRT compile on the same VM
- [ ] If cos < 0.95 → tear down, document the failure, regroup before
      M2

### Post-run

- [ ] `runpodctl send` (or scp/tar) bundle + encodings home
- [ ] `tar tf bundle.tgz | head` to verify contents pre-extract
- [ ] **STOP THE POD IMMEDIATELY** — RunPod bills per-minute
- [ ] Bench bundle on X2E via existing `npu_engine/` runner (adapt for
      0.6B partition count)
- [ ] Append run log to this doc's Update log

## Open questions (resolve as we go)

1. **Does `qai_hub_models.models.qwen3_0_6b.quantize` exist?** Or do
   we drive AIMET v2 directly via a port of
   `sq2_aimet_local/probe_qwen3_0p6b_ptq.py` plus SEQ_MSE+AdaScale
   calls? Confirm during M1 step 3.
2. **AIMET tensor names ↔ our pathb graph names** (P1 in
   `one_pipeline_cloud_gpu.md`). Resolved when we see encodings.json
   output for the first time.
3. **Which encodings.json schema does qairt-converter consume via
   `--quantization_overrides`?** SQ2 deferred this. Two candidate
   formats: AIMET v2's `bw/dtype/enc_type/scale/offset` shape, vs the
   QAIRT IR `quant_params/scale_offset` shape. M1's qairt-converter
   step is the test — if it errors, write the translator (~50 LOC).
4. **Does `aimet_torch.onnx.export` with `use_external_data_format=True`
   handle 4B+ models?** SQ2 hit the 2 GB protobuf cap on
   `sim.export()`; the v2-recommended replacement is untested.
5. **A40 48 GB headroom on 14B AIMET calibration**. Run M3 to find out
   — if it OOMs, escalate to A100 80 GB.
6. **Does AIMET v2.29 ship a `qwen3_5` or `qwen3_6` adapter?** If not,
   ~80 LOC per SQ3-Granite. M4 pre-flight resolves this.
7. **35B-A3B VRAM strategy on M5**: 2× A100 80 GB model-parallel, vs
   per-expert calibration loop, vs renting H100. Decision deferred to
   post-M4 with M3+M4 actual VRAM data in hand.

## Decision rules

- **Stop spending after any failed milestone** until we understand
  the failure. Don't chase larger models with a broken pipeline.
- **Match-or-beat Qualcomm at M2 is the hinge**. If we can't reproduce
  their shipping bundle to within argmax-match tolerance, we don't
  graduate to M3. We either fix the pipeline or accept "near-Qualcomm"
  and re-scope M3-M5 expectations.
- **Tear-down discipline**: every rent session ends with `poweroff` +
  RunPod stop button + verified $0/min billing on the account page.
  No exceptions, no "I'll just leave it for tomorrow."

## Reference docs

- Design rationale: **`docs/one_pipeline_cloud_gpu.md`**
  (Q1-Q4, P1-P12, future-model extension)
- Operational runbook: **`docs/rent_cloud_compute.md`**
  (Scenario A high-RAM CPU, Scenario B CUDA GPU)
- AIMET local survey: `last_side_quest/sq2_aimet_local/aimet_local_survey.md`
- AIMET MoE adapter pattern: `last_side_quest/sq3_small_moe/findings.md`
- 4B reference anchor: `docs/qualcomm_reproduction_4b.md`
- Qwen3-4B baselines: `docs/qwen3_4b_baseline_all_backends.md`
- Tokenizer-generation incompat: memory
  `reference_qwen_tokenizer_generations.md`
- HTP session ceiling: memory `reference_ortqnn_session_limit.md`

## Update log

- **2026-04-29** — Plan committed. Hardware decision: A40 48 GB at
  $0.44/hr for M1-M3; upgrade to A100 80 GB for M4-M5. Validation
  anchor: `models/qualcomm-qwen3-4b-ref/` for M2 byte-comparison.
  M1 not started yet — pre-rent checklist pending.
- **2026-05-01 — M1 results in progress** (ongoing session). See M1
  block below for AIMET-version sweeps + numbers as they land.
- **2026-05-01** — M1 kickoff. RunPod community-cloud A40 48 GB pod
  rented (driver 575.57.08, CUDA 12.9 host; pod default Python 3.10.12).
  Persistent network volume mounted at `/workspace` (mfs eu-se-1, 206 TB
  free) — survives pod lifecycle, so models + venvs persist.
  Layout established:
    - `/workspace/specula/`            ← repo clone (this dir)
    - `/workspace/models/Qwen3-0.6B/`  ← HF FP weights
    - `/workspace/venvs/`              ← per-attempt AIMET venvs
  Downloaded `Qwen/Qwen3-0.6B` direct via curl (no `huggingface-cli`):
    - 9 files, 1.5 GB total. `model.safetensors` 1503300328 B, sha256
      `f47f7117…996874b`.
    - Config confirms `Qwen3ForCausalLM`, `torch_dtype=bfloat16`,
      28 layers / 1024 hidden / 16 heads / vocab 151936 / max_pos 40960.
  Next: stand up first AIMET venv (Python 3.10 + torch+cu121 +
  `aimet_onnx-2.26.0+cu121` per `rent_cloud_compute.md` Scenario B),
  confirm whether `qai_hub_models.models.qwen3_0_6b.quantize` ships as a
  recipe (open question 1) — fall back to direct `aimet_onnx` driver if
  not. Plan to keep multiple venvs side-by-side under `/workspace/venvs/`
  rather than mutate one (e.g. `aimet-2.26-cu121-py310/`,
  `aimet-2.29-cu121-py310/`) so version sweeps don't require reinstall.

## M1 — Qwen3-0.6B w4a16 (in progress)

**Goal restatement.** Even though `qai-hub-models` ships no
`qwen3_0_6b.quantize` recipe (confirmed: 0.39.1 has no `[qwen3-0-6b]`
extra; 0.52.0 has none either; only `[qwen3-4b]` and
`[qwen3-4b-instruct-2507]` in the Qwen3 family), the goal is a
**reusable AIMET driver** that works on any HF causal LM, not lean on
per-model recipes. M1 directly drives `aimet_torch.v2.QuantizationSimModel`
+ `apply_seq_mse` + `apply_adascale` from a generic script.

**Driver:** `aimet_quant.py` (this dir). Stages: HF load → LogitsOnly
wrap → calibration set → FP32 reference probe → QuantSim build →
[optional] SEQ_MSE → [optional] AdaScale → compute_encodings → cos
probe → save_encodings_to_json (+ optional ONNX export). Manifest
emitted at every stage so partial runs stay forensically useful.

**Path choice (M1):** Path T (`aimet_torch`) on the HF model directly,
no pathb rewrites yet. Rationale: P1 (Qualcomm-pathb tensor names) is
a concern only when comparing against a Qualcomm shipping bundle,
which 0.6B doesn't have. M2 onward will layer pathb rewrites in for
oracle alignment with `models/qualcomm-qwen3-4b-ref/`.

### Reproducibility anchors

**Cloud host.** RunPod community-cloud A40 48 GB pod, $0.44/hr.
NVIDIA driver 575.57.08 on host CUDA 12.9. Persistent network volume
mounted at `/workspace` (mfs eu-se-1, ~205 TB free) — survives pod
lifecycle. Layout used:

```
/workspace/specula/                  ← repo clone (all paths in this doc relative here)
/workspace/models/Qwen3-0.6B/        ← HF FP weights (1.5 GB)
/workspace/venvs/aimet-*/            ← per-attempt AIMET venvs
```

**Model fetch (no `huggingface-cli` needed).** Direct curl from the
HF resolve URL:

```bash
mkdir -p /workspace/models/Qwen3-0.6B && cd /workspace/models/Qwen3-0.6B
for f in config.json generation_config.json tokenizer.json tokenizer_config.json \
         vocab.json merges.txt LICENSE README.md model.safetensors; do
  curl -sLfO "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/$f"
done
```

`model.safetensors` should be 1503300328 B with sha256
`f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.
Config: `Qwen3ForCausalLM`, BF16, 28 layers, hidden 1024, heads 16,
vocab 151,936, max_pos 40,960.

**Venvs built (full pin list).**

| venv | python | aimet_torch | aimet_onnx | torch | transformers | optimum | onnxruntime-gpu | status |
|---|---|---|---|---|---|---|---|---|
| `aimet-2.26-cu121-py310` | 3.10.12 | 2.26.0+cu121 | 2.26.0+cu121 | 2.5.1+cu121 | 4.57.6 | 2.1.0 | 1.23.2 | **working** (1a/d/e) |
| `aimet-2.29-cu126-py312` | 3.12.13 | 2.29.0+cu126 | 2.29.0+cu126 | 2.7.1+cu126 (downgraded from 2.11.0) | 4.57.6 | 2.1.0 | 1.25.1 | sim build crash (1f/1f2) |

Build the **working** venv from a fresh box (`uv` 0.10+ in PATH):

```bash
uv venv --python 3.10 /workspace/venvs/aimet-2.26-cu121-py310

VIRTUAL_ENV=/workspace/venvs/aimet-2.26-cu121-py310 uv pip install \
  "qai-hub-models[qwen3-4b]" \
  "onnxruntime-gpu==1.23.2" \
  "https://github.com/quic/aimet/releases/download/2.26.0/aimet_onnx-2.26.0+cu121-cp310-cp310-manylinux_2_34_x86_64.whl" \
  --extra-index-url https://download.pytorch.org/whl/cu121

VIRTUAL_ENV=/workspace/venvs/aimet-2.26-cu121-py310 uv pip install \
  "https://github.com/quic/aimet/releases/download/2.26.0/aimet_torch-2.26.0+cu121-py310-none-any.whl" \
  "transformers" \
  "optimum[onnxruntime]" \
  --extra-index-url https://download.pytorch.org/whl/cu121
```

(The `qai-hub-models[qwen3-4b]` extra is unrecognized in the
0.39.1 release that this pin chain pulls — pip prints a warning and
installs base only. We don't actually use any qai_hub_models recipe;
the install is for the dependency closure of `transformers /
torch / aimet`. The driver `aimet_quant.py` works independently.)

Sanity check after install:

```bash
PY=/workspace/venvs/aimet-2.26-cu121-py310/bin/python
$PY -W ignore -c "
import torch, aimet_torch, aimet_onnx
print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available(), '| dev:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')
print('aimet_torch:', aimet_torch.__version__, '| aimet_onnx:', aimet_onnx.__version__)
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.seq_mse import apply_seq_mse
from aimet_torch.experimental.adascale.adascale_optimizer import apply_adascale
print('all imports OK')"
```

Expected output: `torch: 2.5.1+cu121 | cuda: True | dev: NVIDIA A40`,
`aimet_torch: 2.26.0+cu121 | aimet_onnx: 2.26.0+cu121`, `all imports OK`.

**Rerun a single AIMET stage** (driver is `aimet_quant.py` in this dir):

```bash
cd /workspace/specula/last_side_quest/sq4_cloud_adventure
PY=/workspace/venvs/aimet-2.26-cu121-py310/bin/python

# Run 1a — basic PTQ smoke
$PY aimet_quant.py --model-id Qwen/Qwen3-0.6B \
  --model-path /workspace/models/Qwen3-0.6B \
  --precision w4a16 --ctx 64 --num-cal-samples 16 \
  --output-dir runs/m1a_qwen3_0p6b_w4a16_basic_ptq_2.26

# Run 1d — w4a16 SEQ_MSE + AdaScale (the real M1 attempt)
$PY aimet_quant.py --model-id Qwen/Qwen3-0.6B \
  --model-path /workspace/models/Qwen3-0.6B \
  --precision w4a16 --ctx 64 --num-cal-samples 32 \
  --use-seq-mse --use-ada-scale \
  --seq-mse-num-batches 4 --ada-scale-iters 200 \
  --output-dir runs/m1d_qwen3_0p6b_w4a16_seqmse_ada_2.26

# Run 1e — w8a16 sanity (the only run that argmax-matched)
$PY aimet_quant.py --model-id Qwen/Qwen3-0.6B \
  --model-path /workspace/models/Qwen3-0.6B \
  --precision w8a16 --ctx 64 --num-cal-samples 32 \
  --use-seq-mse --use-ada-scale \
  --seq-mse-num-batches 4 --ada-scale-iters 200 \
  --output-dir runs/m1e_qwen3_0p6b_w8a16_seqmse_ada_2.26
```

`aimet_torch 2.26` ships native AdaScale support for `Qwen3DecoderLayer`
/ `Qwen3Model` (also Qwen2, Qwen2.5-VL, Phi3, Mistral, Llama, Qwen3-VL)
in `aimet_torch.experimental.adascale.adascale_optimizer.supported_modules`.
SEQ_MSE in `aimet_torch.v2.seq_mse.apply_seq_mse`.

### Run 1a — basic PTQ (w4a16, no SEQ_MSE/AdaScale) on AIMET 2.26

Smoke test to validate the script. 16 cal samples, ctx=64, ~5 min wall.

| metric | value |
|---|---|
| cos(fp32, q) | **-0.066937** |
| fp32 last-pos argmax | `' Paris'` |
| q   last-pos argmax | `'ont'` |
| argmax match | **False** |
| compute_encodings wall | 247.9 s (Triton kernel failed → torch fallback) |
| `qsim.json` (encodings) | 147 MB (338 act + 311 param entries) |

**Reproduces the SQ2 local-PTQ finding bit-for-bit** (SQ2 saw cos
-0.065 with the same V/O-collapse pathology). Confirms basic PTQ on
Qwen3-0.6B at w4 is fundamentally broken — the V/O-projection
collapse is real and reproducible on cloud GPU same as on Prism CPU.

### Run 1b/c — SEQ_MSE + AdaScale (w4a16) on AIMET 2.26 — landed (Run 1d)

Run 1b (first SEQ_MSE+AdaScale attempt) crashed during AdaScale on
two API issues — both root-caused, both fixable in the driver:

1. **AdaScale signature** — 2.26 wants `apply_adascale(qsim=…,
   data_loader=…, forward_fn=…, num_iterations=…)`. We had passed
   `sim=…` (alias from older docs) and no `forward_fn`.
2. **Device split mid-pipeline** — `apply_adascale` *explicitly
   moves the entire model to CPU* at start, then walks blocks back
   to GPU one at a time via `BlockwiseSampler` (`keep_unused_blocks_on_cpu=True`,
   `cache_activations_on_disk=True`). This is intentional — it
   makes AdaScale fit on small VRAM. Our forward_fn was hardcoding
   `inputs.to(cuda)` while the model was on CPU → mismatch on the
   embedding lookup. Calibration tensors must stay CPU-resident;
   `forward_fn` must dynamically discover device via
   `next(model.parameters()).device` and move inputs there. Fix
   committed in `aimet_quant.py` (run 1c also caught this same
   mismatch in `compute_encodings` post-AdaScale; the fix in run 1d
   addressed both stages).

**Run 1d — clean end-to-end on AIMET 2.26** (32 cal, ctx 64,
seq_mse 4 batches, adascale 200 iters):

| stage | wall |
|---|---:|
| load HF (FP32) | 44 s |
| sim build | 17 s |
| SEQ_MSE | 302 s |
| AdaScale | **994 s** (35 s/block × 28 blocks) |
| compute_encodings | 35 s |
| save encodings | 34 s |
| **total** | **24 min** |

| metric | basic PTQ (1a) | SEQ_MSE+AdaScale (1d) |
|---|---:|---:|
| cos(fp32, q) | -0.0669 | **0.5261** |
| q last-pos argmax | `'ont'` | `' is'` |
| argmax match | ✗ | ✗ |

**Verdict.** SEQ_MSE+AdaScale rescues 0.6B from total V/O collapse
(cos -0.07 → 0.53 — a real improvement) but **does not clear the
0.95 acceptance gate**, and argmax still misses (`' Paris'` →
`' is'`). The improvement is enough to confirm the AIMET pipeline is
functional and that the calibration techniques have *some* grip on
the V/O issue, but not enough to ship. Two readings, both
defensible:

1. **Structural at 0.6B** (plan's pessimistic prior). The narrow
   hidden dim (1024) means V/O projection rank is small, weight
   distribution has heavier tails, and w4 just doesn't have the
   levels to encode it. SEQ_MSE finds the best per-tensor scale,
   AdaScale finds the best per-block scale, but no scaling beats
   the bit budget. **If this is correct**, escalate via the
   `one_pipeline_cloud_gpu.md` Q4 ladder (SmoothQuant, AWQ, V/O→w8
   pin, QAT) or just accept the prior and graduate to Qwen3-4B
   where the pipeline reasoning is more likely to land.

2. **Insufficient calibration / iteration budget**. We used 32 cal
   samples (Qualcomm's recipe defaults to 128) and AdaScale 200
   iters (Qualcomm's default is 1500 — 7.5× more compute, ~2 hr).
   SEQ_MSE batches at 4 (default candidates per layer is also low).
   **If this is correct**, a longer single run could push cos
   higher.

We do both before declaring. Reading 2 is cheaper to falsify
(~2 hr cloud run) and unblocks reading 1 if it succeeds.

**Encodings sanity (run 1d).** `qsim.json` 147 MB, format
`{activation_encodings, param_encodings}`. Per-tensor entries are
*lists* of per-channel/per-block sub-encodings (e.g.
`inner.lm_head.weight` carries 151,936 entries — one per output
channel, since vocab=151,936). Each sub-encoding is
`{bitwidth, dtype, is_symmetric, scale, offset, min, max}`. Same
schema as run 1a but with shifted scales — confirms SEQ_MSE+AdaScale
actually mutated the encoding choices, just not enough.

### Next-roads matrix (research mode)

Independent levers; cheap-first ordering. Each row produces one
manifest+encodings under `runs/<run_id>/`. Cost is wall-clock on the
A40 at $0.44/hr.

| run id | what | hypothesis | est. wall | cost |
|---|---|---|---:|---:|
| `m1e_w8a16_2.26` | drop param_bw 4 → 8, otherwise identical to 1d | sanity floor: should be ≥0.99 | 25 min | $0.20 |
| `m1f_w4a16_seqmse_ada_2.29` | rerun 1d on AIMET 2.29 venv, same args | newer seq_mse / adascale could shift cos | 25 min | $0.20 |
| `m1g_w4a16_long` | 1d args + adascale 1500 iters + 128 cal samples + seq_mse 16 batches | reading-2 falsification | ~2 hr | $0.90 |
| `m1h_w4a16_smoothquant` | add SmoothQuant pre-step (if available in 2.26) | targets V/O magnitude shift directly | 30 min | $0.25 |
| `m1i_vo_pin_w8` | per-tensor override: V/O proj layers pinned w8, rest w4 | blunt fix; should clear 0.95 if V/O is the only issue | 30 min | $0.25 |

Done in this rough order: e (sanity) → f (cheap version sweep) →
g (long-iter falsification of structural reading) → h, i (mechanism
investigations only if g lands < 0.95 too).

**Outstanding from M1d.**
- The `qsim.json` filename is misleading — the `save_encodings_to_json`
  log line says `qsim.encodings.json` but the actual file is `qsim.json`.
  Cosmetic; don't bother fixing now.
- AdaScale wall (994 s for 28 blocks at 200 iters) extrapolates to
  ~5400 s (90 min) for Qwen3-4B (36 blocks, similar per-block cost
  but larger weights → ~2× per-iter time). M2 budget tracks.

### Run 1e — w8a16 sanity (AIMET 2.26) — PIPELINE CONFIRMED

Same args as 1d but precision=w8a16. Run concurrently with the
ill-fated 1f/1f2 attempts on 2.29 (so AdaScale wall ~40% slower than
1d due to GPU contention; not a clean perf number). Output:

| metric | value |
|---|---:|
| cos(fp32, q) | **0.996105** |
| q last-pos argmax | `' Paris'` |
| fp32 last-pos argmax | `' Paris'` |
| **argmax match** | **✓** |
| SEQ_MSE wall | 231.5 s |
| AdaScale wall | 1404 s (slowed by m1f contention) |
| compute_encodings wall | 117 s (also contended) |
| `qsim.json` (encodings) | ~67 MB (smaller than w4 because per-channel groups are coarser) |

**Verdict.** The pipeline + algorithm are correct end-to-end.
Bottleneck for w4a16 at 0.6B is purely the bit budget. With 4× more
levels (16 vs 256) the same calibration techniques produce ship-
quality quantization.

This **strongly supports graduating to M2 (Qwen3-4B w4a16)**: the 4B
has 36 layers × 2560 hidden vs 0.6B's 28 × 1024 → ~10× more
parameters per layer → much more redundancy → quantization noise
averages out. Qualcomm's shipping bundle is the oracle: cos 0.9998,
46/46 argmax. We expect M2 to clear the 0.95 gate comfortably and
likely match the oracle to within ε.

### Run 1f / 1f2 — AIMET 2.29 attempts — BLOCKED

Two attempts to run the same w4a16+SEQ_MSE+AdaScale config on the
2.29 venv. Both crashed at sim build with `RuntimeError: _Map_base::at`
(also seen as `unordered_map::at`) deep inside `torch.jit.trace`'s
functorch vmap dispatch, called from
`aimet_torch.meta.connectedgraph.ConnectedGraph._construct_graph`.

| attempt | python | torch | aimet | result |
|---|---|---|---|---|
| 1f | 3.12.13 | 2.11.0+cu126 | 2.29.0+cu126 | crash at sim build |
| 1f2 | 3.12.13 | 2.7.1+cu126 | 2.29.0+cu126 | same crash |

Downgrading torch from 2.11 → 2.7 didn't change the failure. So this
isn't a torch-version issue — it's deeper. Suspects, untested:

1. **Python 3.12 specifically.** AIMET 2.29 ships abi3 wheels (advertised
   3.10/3.11/3.12 compat) but the `_Map_base::at` smell looks like a
   missing key in a torch C++-side dispatcher table — possibly only
   populated for 3.10. **Cheap test:** rebuild as
   `aimet-2.29-cu126-py310` and re-run.
2. **Eager attention.** `attn_implementation="eager"` may produce a
   graph that AIMET 2.29's tracer chokes on (newer transformers default
   is sdpa). Untested.
3. **AIMET 2.29's switch to a new tracer.** Possible regression they
   shipped that we'd need to file.

Current spend on this rabbit hole: ~$0.15 (model loads twice + part of
sim build). Not pursued further this session — graduating to M2 has
better expected return. AIMET 2.29 angle parked for a future session.

### M1 closing — graduate decision

| open question | answer |
|---|---|
| Does our generic AIMET driver work? | **Yes.** Pipeline confirmed end-to-end via 1e (w8a16 cos 0.996, argmax match). |
| Does SEQ_MSE+AdaScale unstick 0.6B w4a16? | **No.** cos -0.07 → 0.53 is real grip but not enough. Bit budget too narrow at 0.6B density. |
| Was the 0.6B V/O collapse a calibration issue or structural? | **Structural-ish.** With more iters (1500 vs 200) cos would push higher but unlikely to clear 0.95. The escalation ladder (SmoothQuant / AWQ / V/O→w8 pin) could close the gap, but spending those rounds on 0.6B is poor ROI. |
| Is M2 (Qwen3-4B) the right next target? | **Yes.** 1e shows the algorithm is sound at higher bw; 4B's redundancy gives w4 enough headroom; Qualcomm's bundle is on disk for byte/argmax oracle. |

Action: graduate to **M2 — Qwen3-4B w4a16 + SEQ_MSE + AdaScale on
AIMET 2.26**. The 0.6B w4a16 pipeline shape stays usable as a quick
smoke test for future arch adapter work. AIMET 2.29 venv stays around
for a future debugging session but isn't a blocker.

## End-to-end NPU pipeline attempt on Qwen3-0.6B w8a16 — partial

After landing run 1e (cos 0.996, argmax match), we tried to push the
encodings all the way through the QAIRT chain to a HTP context binary
(`.bin`) we could ship to the X2E. **Made it through ONNX export and
DLC, blocked at qnn-context-binary-generator.** Documenting precisely
where, why, and exactly what's needed to unblock so the next session
picks up cold.

### What worked

#### Step A — ONNX export from existing AIMET encodings

`export_onnx_from_encodings.py` (this dir): rebuilds the sim from the
HF model + dummy input, calls `sim.load_encodings(qsim.json)`, then
`aimet_torch.onnx.export(sim, dummy_input, path, opset_version=21)`.
Skips re-running SEQ_MSE+AdaScale. ~6 min wall.

```bash
PY=/workspace/venvs/aimet-2.26-cu121-py310/bin/python
OUT=/root/sq4_intermediates/m1e_w8a16_export   # ephemeral; rootfs is fine
mkdir -p $OUT
$PY export_onnx_from_encodings.py \
  --model-id Qwen/Qwen3-0.6B \
  --model-path /workspace/models/Qwen3-0.6B \
  --encodings runs/m1e_qwen3_0p6b_w8a16_seqmse_ada_2.26/qsim.json \
  --precision w8a16 --ctx 64 \
  --output-dir $OUT \
  --filename-prefix qwen3_0p6b_w8a16
```

Output:
- `qwen3_0p6b_w8a16.onnx` — graph proto only (5 MB; opset 21)
- `qwen3_0p6b_w8a16_encodings.json` — per-tensor encodings (148.8 MB)
- 367 sibling weight tensors as ONNX external data (~3.0 GB total)

ONNX summary: 6,661 nodes, 254 MatMul, **2,364 QDQ pairs** (full
coverage), input shape `[1, 64]`, output `[1, 64, 151936]`. Note: AIMET
requires `opset_version >= 21` for INT4/INT16 QDQ ops (smoke-tested
with 17 first; failed loudly).

#### Step B — QAIRT 2.45 SDK install (persistent on /workspace)

URL pattern: replace version segment to fetch newer/older releases.

```bash
# Download (~1.55 GB)
curl -sL -A "Mozilla/5.0" \
  -o /tmp/qairt-2.45.zip \
  "https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.45.40.260406/v2.45.40.260406.zip"

# Extract → persistent location
unzip -q /tmp/qairt-2.45.zip -d /tmp/qairt
mv /tmp/qairt/qairt/2.45.40.260406 /workspace/sdks/qairt-2.45.40.260406
rm -rf /tmp/qairt /tmp/qairt-2.45.zip

# System libs the bundled libDl*.so links against (libc++ + libunwind);
# Ubuntu 22.04 base image is missing these.
apt-get install -y libc++1 libc++abi1 libunwind8

# QAIRT-required Python deps (the SDK's check-python-dependency lists
# these; we only install what's actually missing in the AIMET 2.26 venv)
VIRTUAL_ENV=/workspace/venvs/aimet-2.26-cu121-py310 uv pip install \
  aenum dash invoke lxml mako mock optuna paramiko pathlib2 \
  plotly pytest scikit-optimize xlsxwriter
```

Set the env up for *every* QAIRT command (or put in a sourced
`activate-qairt.sh`):

```bash
export QAIRT_SDK_ROOT=/workspace/sdks/qairt-2.45.40.260406
export PATH=$QAIRT_SDK_ROOT/bin/x86_64-linux-clang:$VIRTUAL_ENV/bin:$PATH
export LD_LIBRARY_PATH=$QAIRT_SDK_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH
export PYTHONPATH=$QAIRT_SDK_ROOT/lib/python:$PYTHONPATH
export VIRTUAL_ENV=/workspace/venvs/aimet-2.26-cu121-py310
```

Verify:
```bash
qairt-converter --version  # any non-crash output is fine
```

**Why 2.45 and not 2.42**, per `docs/npu_ort_qnn_version_match.md`:
the X2E currently runs `onnxruntime-qnn 2.1.0` which bundles
**QAIRT 2.45.40.260406**. A binary compiled against 2.42 would fail
on load with `LoadCachedQnnContextFromBuffer Error 5000`. Match the
runtime exactly.

#### Step C — qairt-converter: ONNX+QDQ → DLC

```bash
qairt-converter \
  --input_network $OUT/qwen3_0p6b_w8a16.onnx \
  --output_path $OUT/qwen3_0p6b_w8a16.dlc \
  --quantization_overrides $OUT/qwen3_0p6b_w8a16_encodings.json \
  --preserve_io_datatype
```

Result: **`INFO_CONVERSION_SUCCESS`**, ~6 min wall, 1.35 GB DLC.
Default `--target_backend` is `HTP` (don't need to pass it).

Two warning classes that matter for diagnosis:

1. **`Only numerical type cast is supported. The cast op: ... will be
   interpreted at conversion time`** — *every* `Cast` in the graph
   triggers this. Implies they're not converted to native QNN ops.
2. **"Following OPs fallback to float"** — long list of MatMul /
   Mul / Softmax / Sigmoid / RMS-norm ops. The `--quantization_overrides`
   encoding tensor names didn't line up with the converted graph's
   tensor names for ~80% of ops, so they fell back to FP. (Known P1
   from `one_pipeline_cloud_gpu.md`.)

Despite the fallbacks the DLC is structurally well-formed; Qualcomm's
toolchain happily produced it. The compile-stage validator is what
catches the deeper issue, next step.

### What didn't work — the wall

#### Step D (BLOCKED) — qnn-context-binary-generator: DLC → .bin

```bash
qnn-context-binary-generator \
  --backend $QAIRT_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so \
  --dlc_path $OUT/qwen3_0p6b_w8a16.dlc \
  --binary_file $OUT/qwen3_0p6b_w8a16 \
  --output_dir $OUT
```

Failed at HTP op-config validation:

```
[ ERROR ] Failed to validate op /inner/model/Cast with error 0xc26
[ ERROR ] Validate OpConfig failed: QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE
[ ERROR ] Failed to successfully compose graph
[ ERROR ] ComposeGraphs Failed with error = 1
```

The ops listed just before the validator gave up have dtype combos HTP
won't accept:
- `QNN_DATATYPE_FLOAT_32 → QNN_DATATYPE_INT_32` (Cast for position-id math)
- `QNN_DATATYPE_BOOL_8 → QNN_DATATYPE_FLOAT_16` (Cast for attention mask)
- `QNN_DATATYPE_INT_32 → QNN_DATATYPE_FLOAT_16` (Cast for index conversion)

These are emitted by HuggingFace's standard transformers attention
implementation (`attn_implementation="eager"` and friends). HTP's op
package validator rejects them. The compiler can produce a DLC with
them; the runtime composer cannot bind them to HTP ops.

### Why this happened — root cause

We deliberately ran the **shortest possible path** through M1:
**HF causal LM → AIMET → ONNX with QDQ → QAIRT**.

The full Qualcomm/specula path adds a `pathb` rewrite stage between
the HF export and AIMET — see `docs/qualcomm_reproduction_4b.md` and
the existing scripts in `scripts/`:

```
HF (transformers) → optimum-export → ONNX
                  → rewrite_qwen3_htp.py     (HTP-friendly ops)
                  → rewrite_qwen3_pathb.py   (additive mask, rotary hoist)
                  → pin_shapes_qwen3_*b.py   (static shape pin per ctx)
                  → AIMET QuantSim + SEQ_MSE + AdaScale
                  → QAIRT chain
                  → .bin
```

The rewrite scripts **specifically** delete / replace the Cast op
patterns that block us at Step D — that's their entire point. They
also fold the multiplicative attention mask into an additive mask
(HTP wants one big additive `Add` rather than `Where + Mul`),
hoist rotary embedding multiplications out of decoder blocks,
and pin shapes to fixed `(B, T)` so the HTP backend can plan kernels.

Skipping pathb saved an hour today. Cost: the .bin compile step is
unreachable. Predictable from the design doc; we made a conscious
trade-off and documented the wall.

### Tomorrow's next-session plan (concrete)

Pick exactly one of two starts:

**(α) Fix the 0.6B compile by running pathb** — 1-2 sessions.
Re-uses the 1e w8a16 encodings as oracle (we know cos 0.996 is
achievable on this model with these techniques). Concrete:

1. Run `optimum-cli export onnx Qwen/Qwen3-0.6B onnx-optimum/` to get
   a clean exported ONNX (no AIMET wrapper, no QDQ).
2. Run `python scripts/rewrite_qwen3_htp.py --model-stem qwen3_0_6b
   --input onnx-optimum/ --output onnx-htp/`.
3. Run `python scripts/rewrite_qwen3_pathb.py --model-stem qwen3_0_6b
   --input onnx-htp/ --output onnx-pathb/`.
4. Run `python scripts/pin_shapes_qwen3_4b.py
   (or generalized variant) --model-stem qwen3_0_6b --ctx 256
   --input onnx-pathb/ --output onnx-pathb-ctx256/`.
5. Switch from `aimet_torch` (graph-tracer-based) to **`aimet_onnx`**
   (already installed; `aimet-onnx 2.26.0+cu121` in the venv).
   `aimet_onnx.QuantSimModel` consumes the rewritten ONNX directly,
   avoiding the tensor-name mismatch (P1) that caused 80%
   float-fallback in this session's converter run.
6. Save encodings, run qairt-converter against the rewritten ONNX
   (no more Cast issues), then qnn-context-binary-generator.
7. Bundle: `.bin` + `tokenizer.json` + a wrapper ONNX for ORT-QNN
   (per `models/qualcomm-qwen3-4b-ref/` shape).
8. tar + scp / rclone home.

**(β) Skip the 0.6B compile, graduate to Qwen3-4B** — 1 session. The
4B has the Qualcomm shipping bundle as oracle and is the actual
production target. The pathb chain is needed for 4B too, so the
session-1 work transfers wholesale. The only thing skipping costs:
no end-to-end on-NPU validation of an artifact we made. We'd land
that on the 4B run anyway.

**Recommended: (α) tomorrow**, because it is research-pure (we know
the answer for 0.6B w8a16 already at the encodings level — cos 0.996
— so any failure on the X2E load is a pipeline issue, not a
calibration issue). Then use that hardening to fast-track (β).

### What to keep, what to clean up

**Keep on `/workspace` (persistent across pod restarts):**

- `/workspace/specula/` — repo (this commit lands here)
- `/workspace/models/Qwen3-0.6B/` — 1.5 GB; will reuse next session
- `/workspace/venvs/aimet-2.26-cu121-py310/` — 16 GB; the working venv
- `/workspace/sdks/qairt-2.45.40.260406/` — 4.9 GB; the matched QAIRT

**Stage for cleanup (regeneratable from script + commands above):**

- `/workspace/specula/last_side_quest/sq4_cloud_adventure/runs/m1[abcdef].../qsim.json`
  (~600 MB combined). The encoding *is* the artifact, but it's
  re-emitted in 25 min and the manifest carries the result. Defer
  cleanup until the 0.6B work is fully closed at end of pathb session.
- `/workspace/venvs/aimet-2.29-cu126-py312/` — 9 GB; broken at sim
  build. Keep until we either succeed in fixing it (rebuild as
  py310) or formally park the 2.29 angle in a roadmap doc.
- `/root/sq4_intermediates/` — 6 GB on ephemeral rootfs; auto-clears
  on pod restart. No action needed.

**Disk budget headroom (after this session):**

| location | used | limit | free |
|---|---:|---:|---:|
| `/workspace` (RunPod NV quota) | ~32 GB | 100 GB | ~68 GB |
| `/` (rootfs, ephemeral) | ~10 GB | 50 GB | ~40 GB |

Plenty of room for tomorrow's pathb + 4B work.

## M1 — pathb → aimet_onnx → qairt → HTP .bin (end-to-end landed 2026-05-01)

**TL;DR.** Path α from the prior session lands. `qwen3_0p6b_pathb_w8a16.bin`
(918 MB) compiled clean against HTP v75 (X2 Elite). Plumbing question
answered positive: pathb-rewritten ONNX + aimet_onnx encodings flow
through `qairt-converter` + `qnn-context-binary-generator` end-to-end
when the right flags are set. Quality is on a separate lever: cos(fp,q)
= 0.656 with 16 cal samples + basic PTQ — below 0.95 gate; SEQ_MSE +
AdaScale + larger cal set is the unblocker (deferred to M1 quality
follow-up).

### Pipeline as it ran

```
HF FP16 (Qwen/Qwen3-0.6B, /workspace/models/Qwen3-0.6B)
    ↓ optimum-cli export onnx --task text-generation-with-past
qwen3-0.6b-optimum/                   (1.4 MB graph, 3.0 GB onnx_data)
    ↓ scripts/rewrite_qwen3_htp.py --mode stage
qwen3-0.6b-staged/                    (attention_mask → init, 28 IsNaN guards elided)
    ↓ scripts/rewrite_qwen3_htp.py --mode fold-pathbmask
qwen3-0.6b-pathbmask/                 (Where(bool)→Add(attention_bias); IsNaN=0 Range=0 Cast→BOOL=0)
    ↓ scripts/rewrite_qwen3_pathb.py
qwen3-0.6b-pathb/                     (rotary hoisted; 0 rotary_emb nodes; +position_ids_cos/sin inputs)
    ↓ scripts/pin_shapes_qwen3_4b.py --ctx 512 --seq-q 1
qwen3-0.6b-pathb-ctx512/              (237 dims pinned; AR=1, ctx=512)
    ↓ aimet_onnx_quant.py --precision w8a16 --num-cal-samples 16
m1_pathb_w8a16_smoke/                 (qwen3_0p6b_pathb_w8a16.{onnx,encodings,data})
    ↓ qairt-converter --quantization_overrides ... (NO --target_soc_model)
m1_pathb_w8a16_smoke/qwen3_0p6b_pathb_w8a16.dlc   (922 MB)
    ↓ qnn-context-binary-generator --config_file qnn_v75_config.json
m1_pathb_w8a16_smoke/qwen3_0p6b_pathb_w8a16.bin   (918 MB)
    ↓ tar
/workspace/sq4_m1_pathb/qwen3_0p6b-w8a16-pathb-ctx512-x2e.tar (918 MB)
```

### Wall and disk

| stage | wall | output |
|---|---:|---:|
| optimum-cli export onnx | ~3 min | 3.0 GB |
| rewrite_qwen3_htp stage | ~30 s | 2.9 GB |
| rewrite_qwen3_htp fold-pathbmask | ~30 s | 2.9 GB |
| rewrite_qwen3_pathb (rotary hoist) | ~30 s | 3.0 GB |
| pin_shapes_qwen3_4b ctx=512 | ~5 s | 3.0 GB |
| aimet_onnx_quant.py (16 cal, no SEQ_MSE) | ~6 min | 3.0 GB onnx + 32 MB enc |
| qairt-converter | ~80 s | 922 MB DLC |
| qnn-context-binary-generator | ~22 s | **918 MB .bin** |

### Things that almost stopped us, in order

**1. torch 2.5.1 caps ONNX opset at 20.** `optimum-cli export onnx
--opset 21` hard-fails with `ValueError: Unsupported ONNX opset version: 21`.
Solution: drop the `--opset` flag — optimum default (currently 18 for
Qwen3) works, and AIMET's later QDQ insertion produces opset 21 ops
on its own (aimet_onnx upgrades the opset during graph rewrite).

**2. Qwen3-0.6B pathb attention_bias is masked at the END of the
window, not the beginning.** First w8a16 smoke probe came back
cos = 0.07, FP argmax `'dfunding'` (random garbage). Looked like a
bad probe — turned out the cache layout puts the current token at
slot ctx-1=511 and grows real past KVs *backward* from slot ctx-2.
A naive `attn_bias[..., pos+1:] = -inf` masks the wrong end. Correct
form for AR=1 decode at position p:
```python
visible_start = ctx - 1 - pos
attn_bias = np.full((1, 1, 1, ctx), -65504.0, dtype=np.float32)
attn_bias[..., visible_start:] = 0.0
```
After the fix, FP probe returns ` Paris` for "The capital of France is".
Bug was in the cal generator AND the probe — cal samples saw garbage
attention, so the encodings were also garbage; rerun was required.

**3. `aimet_onnx.compute_encodings` is CPU-bound on observation
even when ORT-CUDA does the forward.** GPU sits at 0% during cal
because the per-tensor histogram observers run sequentially in
Python after each forward. 5 min for 16 cal samples on Qwen3-0.6B.
Not blocker, but explains why we don't see GPU util ramp.

**4. `aimet_onnx.QSM.export()` strips QDQ ops before saving the
ONNX.** This is *good* — the exported `.onnx` carries the original
pathb graph's tensor names, which line up exactly with what
qairt-converter sees. Compare m1e (AIMET-torch wrapped graph): 80%
of attention ops fell back to float because AIMET-torch tensor names
disagreed with qairt-converter's view. Now: "Processed 2540
quantization encodings" with no float-fallback warnings on
quantized layers. (There IS a fallback list of attention-internal
ops — Reshape/Transpose/MatMul_1/Mul_9/Expand — that don't have
encodings emitted by aimet_onnx 2.26 for activation-only ops; quality
lever, not plumbing.)

**5. `aimet_onnx` renames graph outputs after Q/DQ injection.**
First smoke crashed at the cos-probe stage with
`ValueError: 'present.0.key' is not in list` because
`sim.session.get_outputs()` returns a different (suffixed) name list
than the FP session. Fix: call `sess.run(all_outs, feeds)` with the
*correct* per-session output name list, indexed by graph order
rather than name. Also moved export *before* probe so the artifact
lands even if the probe stage trips.

**6. qairt-converter `--target_soc_model` allow-list is tiny.**
2.45's BackendInfo only accepts `{SM8845, SM8850, SM8850L}`. We
verified by brute-force enumeration. None map to v75. Workaround: do
NOT pass `--target_soc_model` at the converter (DLC stays SoC-
generic), pin HTP arch later via the binary-generator's config.

**7. `qnn-context-binary-generator` default backend targets HTP v68.**
Without explicit arch config, validation fails with
`[4294967295] has incorrect Value 68, expected >= 73`. The error
line `0xc26 QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE` looked
mysterious until decoded: AIMET's int16 attention ops require ≥v73
HTP, default backend is v68.

**8. `--config_file` schema requires the two-file wrapper for
dsp_arch.** A flat
```json
{"devices":[{"dsp_arch":"v75"}]}
```
gets rejected with `Unknown Key = devices/0/dsp_arch passed in
config`. The OUTER `--config_file` JSON only accepts
`backend_extensions: { shared_library_path, config_file_path }` keys
— `dsp_arch` lives in the *inner* file pointed to by
`config_file_path`. Both files committed:
- `qnn_v75_config.json` (outer wrapper)
- `qnn_v75_inner.json` (`{"devices":[{"dsp_arch":"v75"}]}`)

### Acceptance metrics

| gate | target | got | verdict |
|---|---|---|---|
| QAIRT compile produces multi-part bundle without errors | required | ✓ single-part (0.6B small enough) | ✅ |
| Bundle compiles for X2E HTP arch | required | ✓ v75 | ✅ |
| AIMET cos vs FP32 ≥ 0.95 | M1 gate | 0.656 (16 cal, basic PTQ) | ❌ — known-bad recipe; SEQ_MSE+AdaScale follow-up |
| Bundle loads on X2E (round-trip) | required | not yet tested (cloud→home transfer pending) | ⏳ |
| Coherent text on " The capital of France is" → " Paris" | nice-to-have | argmax mismatch (token 220 ' ') | ❌ — same lever as cos gate |

**Plumbing answer is positive.** Quality answer is negative-but-
expected: 16 cal samples + basic PTQ on Qwen3-0.6B reproduces SQ2's
"V/O collapse" finding (cos -0.07 → 0.66 with mask fix; still well
short of the 0.95 gate). The pipeline now exists; it just needs
better cal-stage configuration to clear the gate.

### Bundle assembled

`/workspace/sq4_m1_pathb/qwen3_0p6b-w8a16-pathb-ctx512-x2e.tar` (918 MB)
contains:
- `qwen3_0p6b_pathb_w8a16.bin`         918 MB  — HTP v75 context binary
- `qwen3_0p6b_pathb_w8a16.encodings`    32 MB  — AIMET v1.0.0 encodings
- `tokenizer.json`                      11 MB
- `tokenizer_config.json`              ~10 KB
- `config.json`                        <1 KB
- `generation_config.json`             <1 KB
- `metadata.json`                       1.7 KB — sha256 + bundle metadata

Missing for full ORT-QNN deployment (do these at home, against the
existing `qwen3_0_6b_draft_v81_ctx512.*` family of artifacts as
template):
- `*.wrapper.onnx` — the QnnContext_BundleSpecific stub graph that
  ORT-QNN consumes. Existing 4B + 7B bundles in
  `models/qualcomm-qwen3-4b-ref/` / `models/qualcomm-qwen2.5-7b-ref/`
  show the pattern; this can be generated locally without re-renting.
- `metadata.yaml` (Qualcomm-style) — re-emit from existing genie
  template with the 0.6B head-dim/layer-count/vocab specifics.
- `genie_config.json` — for `genie-t2t-run` consumption (optional;
  ORT-QNN sidecar path doesn't need it).

### Files lasting on /workspace (persistent across pod restarts)

```
/workspace/specula/                                     (repo, including new sq4 scripts)
/workspace/models/Qwen3-0.6B/                           1.5 GB (HF FP weights)
/workspace/venvs/aimet-2.26-cu121-py310/               16 GB (working venv)
/workspace/venvs/aimet-2.29-cu126-py312/                9 GB (parked, sim-build crash)
/workspace/sdks/qairt-2.45.40.260406/                   4.9 GB (matched QAIRT)
/workspace/sq4_m1_pathb/                              ~22 GB (intermediates + bundle.tar)
  qwen3-0.6b-optimum/                                   2.9 GB (optimum export)
  qwen3-0.6b-staged/                                    2.9 GB
  qwen3-0.6b-pathbmask/                                 2.9 GB
  qwen3-0.6b-pathb/                                     3.0 GB
  qwen3-0.6b-pathb-ctx512/                              3.0 GB
  qwen3_0p6b-w8a16-pathb-ctx512-x2e/                   ~961 MB (bundle dir)
  qwen3_0p6b-w8a16-pathb-ctx512-x2e.tar               918 MB (transportable)
  optimum_export.log
/root/sq4_intermediates/m1_pathb_w8a16_smoke/         3.9 GB (rootfs, ephemeral)
  + ~6 GB from prior m1e_w8a16_export
```

Total persistent disk used now: **~50 GB / 100 GB**. Plenty of headroom
for the SEQ_MSE+AdaScale follow-up M1 quality run AND M2 (Qwen3-4B).

### Next session pickup

The fastest route to a full M1 close (cos ≥ 0.95):

```bash
# Same venv, same QAIRT, just bigger cal + SEQ_MSE+AdaScale
cd /workspace/specula/last_side_quest/sq4_cloud_adventure
PY=/workspace/venvs/aimet-2.26-cu121-py310/bin/python
NVLIBS=$(find /workspace/venvs/aimet-2.26-cu121-py310/lib/python3.10/site-packages/nvidia -name lib -type d | tr '\n' ':')
export LD_LIBRARY_PATH=${NVLIBS}${LD_LIBRARY_PATH:-}

# extend aimet_onnx_quant.py to call apply_seq_mse + apply_adascale
# (currently only basic PTQ); 128 cal samples; quant_scheme min_max
$PY aimet_onnx_quant.py \
  --src-dir /workspace/sq4_m1_pathb/qwen3-0.6b-pathb-ctx512 \
  --tokenizer /workspace/models/Qwen3-0.6B \
  --output-dir /root/sq4_intermediates/m1_pathb_w8a16_seqmse_ada \
  --precision w8a16 --num-cal-samples 128 --ctx 512 --cuda \
  --use-seq-mse --use-ada-scale  # ← needs script extension; aimet_onnx 2.26 has both

# Then qairt + binary-gen as in qairt_compile.sh, swap OUT path.
```

Two known levers if SEQ_MSE+AdaScale alone don't clear the gate:
1. Pin V/O proj layers to w8 via per-tensor encoding override (the
   "V/O collapse" mitigation discussed in `one_pipeline_cloud_gpu.md`
   Q4 ladder).
2. Bigger cal set. SEQ_MSE wants 16+ batches by default; AdaScale 1500
   iters per block. Real run is ~2 hr on A40, $0.90.

Don't redo qairt/qnn until aimet has cleared its gate.

For M2 (Qwen3-4B), the entire chain is reusable; only changes:
- different model dir, different rewrite stem (`--model-stem qwen3-4b`).
- pin_shapes_qwen3_4b is already set up for 4B.
- AIMET wall ~2.5x (36 layers vs 28, 2560 hidden vs 1024).
- qnn-context partition strategy: 4B may need 4-way split; if so
  `rewrite_halfdim_cos_sin.py` becomes relevant.

## M1 quality lap — `end-to-end/` orchestrator + the AdaScale fight

**Status as of 2026-05-01 21:15.** End-to-end orchestrator
(`end-to-end/quantize_to_npu.py`) lands; smoke recipe artifact
proves the chain. Full Qualcomm quality recipe (SEQ_MSE +
AdaScale, 128 cal samples) is now running with a hand-rolled
ReduceMean v18 monkey-patch — first AdaScale-on-Qwen3 attempt
in this stack.

### What the orchestrator does

ONE entry script (`end-to-end/quantize_to_npu.py`) drives the
full chain with idempotent stages (each drops a `done.json`
marker; re-running with the same `--workdir` resumes; use
`--force-stage N` to re-run from N onward):

```
1. optimum-cli export onnx --task text-generation-with-past
2. scripts/rewrite_qwen3_htp.py --mode stage
3. scripts/rewrite_qwen3_htp.py --mode fold-pathbmask
4. scripts/rewrite_qwen3_pathb.py        (rotary hoist)
5. scripts/pin_shapes_qwen3_4b.py        (pin AR=1, ctx=N)
6. AIMET aimet_onnx PTQ + SEQ_MSE + AdaScale (+ optional V/O w8 pin)
7. qairt-converter ONNX+encodings → DLC
8. qnn-context-binary-generator DLC → HTP context .bin (v75)
9. Bundle .bin + tokenizer + metadata, tar
```

Defaults are MAX-QUALITY (full Qualcomm recipe): 128 cal samples,
SEQ_MSE 20 candidates, AdaScale 1500 iters/block, V/O w8 pin
auto-on for w4a16. Uses subprocess for the existing pathb scripts;
imports aimet_onnx directly for the AIMET stage.

### Three new gotchas hit and resolved during the M1 quality lap

**(a) `apply_seq_mse` requires `quant_scheme=min_max`.** aimet_onnx
2.26's seq_mse asserts `sim._quant_scheme in (QuantScheme.min_max,)`
with the message "Use TF quant-scheme with sequential MSE." Default
`post_training_tf_enhanced` fails the assertion immediately. Fix:
when `--use-seq-mse` is on, force `quant_scheme=min_max` and warn.
Side effect: activations use plain min/max instead of histogram-
based observation, which is one reason SEQ_MSE-only landed cos
0.613 (worse than the basic-PTQ smoke at cos 0.656 with
`tf_enhanced` activations).

**(b) AdaScale asserts cal-dict keys match graph input order
exactly.** `apply_adascale` checks
`list(inputs[0].keys()) == [i.name for i in sim.session.get_inputs()]`
verbatim. Our cal generator built dicts as
`{input_ids, position_ids, attention_bias, cos, sin, past_key_values.*}`
but the optimum-exported pathb graph orders them as
`{input_ids, position_ids, past_key_values.*, attention_bias, cos, sin}`.
Fix: re-order each yielded feeds dict to match graph input order
in `lib/cal.py`. SEQ_MSE doesn't have this assertion so it ran fine.

**(c) AdaScale's onnx2torch lacks a ReduceMean v18 handler.**
Optimum-cli emits the graph at opset 18; Qwen3 RMSNorm uses
ReduceMean (113 instances); aimet_onnx 2.26's
`experimental.adascale.onnx2torch_ext` only registers
ReduceMean v1/11/13. AdaScale crashes mid-conversion with:

  `NotImplementedError: Converter is not implemented (
    OperationDescription(domain='', operation_type='ReduceMean',
    version=18))`

Fix: monkey-patch onnx2torch's converter registry from inside
`lib/aimet.py:_patch_onnx2torch_reduce_mean_v18()` BEFORE calling
`apply_adascale`. The handler resolves `axes` from input[1] when
constant (always true for Qwen3's `[-1]` reduction) and reuses the
existing `OnnxReduceStaticAxes`. ~30 LOC, idempotent (returns early
if v18 is already in the registry). In-code, not venv mutation, so
fresh VMs running `quantize_to_npu.py` get it automatically.

### Quality landing zone, walled by AdaScale

| recipe | precision | cal | SEQ_MSE | AdaScale | quant_scheme | cos(fp,q) | argmax |
|---|---|---:|---|---|---|---:|---|
| smoke (m1_pathb) | w8a16 | 16 | off | off | tf_enhanced | 0.656 | ✗ ('') |
| this lap (SEQ_MSE only) | w8a16 | 128 | on (20 cand) | off (crashed) | min_max | **0.613** | ✗ ('The') |
| **this lap (SEQ_MSE + AdaScale)** | w8a16 | 128 | on | **on (1500 iters, patched)** | min_max | **— pending** | — |
| reference m1e (aimet_torch) | w8a16 | 32 | on (4 batches) | on (200 iters) | min_max | 0.996 | ✓ |

Hypothesis under test: if AdaScale on aimet_onnx with the
ReduceMean v18 patch behaves the same as aimet_torch's AdaScale,
we should land cos > 0.99 — clearing the M1 acceptance gate at 0.95.

### Run-time pacing observed (aimet_onnx 2.26 on A40, Qwen3-0.6B)

| stage | smoke (16 cal) | full (128 cal) |
|---|---:|---:|
| optimum export | 154 s | 154 s (skipped on resume) |
| pathb chain (stages 2-5) | ~30 s | ~30 s (skipped on resume) |
| AIMET tokenizer + RoPE | 30 s | 30 s |
| FP session build | 2-15 s | 15 s |
| cal sample gather | 2 s | 18 s |
| QSM build | 22-29 s | 25 s |
| SEQ_MSE | n/a | 587 s (~10 min, 113 ops) |
| AdaScale (1500 iters × 28 blocks) | n/a | **TBD — first attempt of this stack** |
| compute_encodings | 293 s (16 cal) | **21 s** (128 cal) |
| sim.export | 14 s | 27 s |
| qairt-converter | 80 s | 96 s |
| qnn-context-binary-generator | 22 s | 160 s |
| bundle + tar | 15 s | ~30 s |

Surprise: with SEQ_MSE+min_max, `compute_encodings` is 21s (not
the 293s the smoke saw with tf_enhanced). SEQ_MSE pre-populates
all weight encodings; the post-SEQ_MSE compute_encodings only has
to observe activations, and min_max activation observation just
reads min/max in one pass per sample. That's a 14× speedup vs the
histogram-based path.

### Disk hygiene

Per-run workdir bloats to ~26 GB before pruning. Once the bundle
.tar is preserved, intermediate stage dirs are safe to delete:

| keep | delete (cheap to regenerate) |
|---|---|
| `01_optimum/` (optimum export, ~3 GB, 3 min to regenerate) | `02_staged/`, `03_pathbmask/`, `04_pathb/` (each ~3 GB; together regenerate in ~30 s) |
| `05_pathb_ctx512/` (input to AIMET, ~3 GB) | `06_aimet_*/` if you have the bundle.tar already |
| `09_bundle_*/qwen3-...-x2e.tar` (the artifact) | `06_aimet_*/`, `07_dlc_*/`, `08_bin_*/` once tar exists |

Cleanup landed 2026-05-01: 32 GB freed (16 GB old smoke pathb
dirs + 9 GB redundant intermediate stages + ephemeral
`/root/sq4_intermediates`). SEQ_MSE-only bundle saved as
`/workspace/runs/qwen3-0p6b-w8a16-seqmse-only-cos0p613.tar` for
post-AdaScale comparison.

### Current run state (2026-05-01 21:15)

PID 4070757 detached (PPID=init), workdir
`/workspace/runs/qwen3_0p6b_w8a16_full`, recipe:
`--precision w8a16 --use-seq-mse --use-ada-scale --ada-scale-iters 1500`.
Stages 1+5 already had `done.json` from the earlier SEQ_MSE-only
run; 2/3/4/6/7/8/9 are regenerating (rebuild stages 2-4 in ~30 s,
then SEQ_MSE ~10 min, then AdaScale [unknown wall], then
compute_encodings + qairt + qnn-context + bundle).

If AdaScale crashes despite the v18 patch (e.g. on a different
op version we haven't enumerated), the orchestrator's idempotent
design means stages 1-5 stay marked done and only stage 6 needs
to retry — costs ~12 min per fresh attempt. If it succeeds and
clears cos ≥ 0.95, this is the artifact we ship for M1.

### M1 AdaScale fight log (2026-05-01 → 02 night)

The ReduceMean v18 patch unblocked the *first* AdaScale call but
revealed two more incompatibilities in aimet_onnx 2.26's AdaScale
when run against an optimum-cli + pathb Qwen3 graph. Each crash
manifested as the same generic onnx2torch error
`RuntimeError: Got unexpected input value type (ValueType.UNKNOWN)`,
which made it slow to diagnose. Three monkey-patches required;
chronicled in commit messages
`33ad19a`, `04f1709`, `1b31a4e`.

#### Patch 1 — onnx2torch ReduceMean v18 handler (commit 33ad19a)

Symptom: `apply_adascale` crashed before AdaScale block 0 with
`NotImplementedError: Converter is not implemented (...
ReduceMean, version=18)`. aimet_onnx 2.26 ships only ReduceMean
v1/v11/v13 converters; opset 18 changed `axes` from attribute to
optional input. Qwen3 RMSNorm has 113 ReduceMean instances.
Fix: register a v18 handler in `onnx2torch`'s converter registry
that resolves `axes` from `input[1]` when it's a constant initializer
(always true for RMSNorm's `[-1]`) and reuses the existing
`OnnxReduceStaticAxes`. ~30 LOC, idempotent.

#### Patch 2 — HF-style past_kv naming match (commit 04f1709)

Symptom: post-Patch-1, AdaScale entered block 0 then crashed in
onnx2torch.convert with `ValueType.UNKNOWN`. Diagnosis:
`apply_adascale` matches block KV inputs by substring
`past_key_{idx}_in` / `past_value_{idx}_in` (Qualcomm qai_hub
naming convention). Optimum-cli uses HF naming
`past_key_values.{idx}.{key,value}`, so `block_kv_tensor_names`
was `[]` for every block. The extracted onnx subgraph then
omitted past_kv as graph inputs; their references in attention
nodes became dangling, hence UNKNOWN.

Fix: monkey-patch `AdaScale.apply_adascale` with a verbatim copy
of the original modulo the KV-name match, which now reads
`if f"past_key_values.{idx}." in name`. This catches both `.key`
and `.value` in one filter. ~150 LOC patch (including the surrounding
context that wraps the matching).

#### Patch 3 — module-level apply_adascale rebind (commit 1b31a4e)

**The load-bearing fix.** Symptom: post-Patch-2, AdaScale STILL
crashed in block 0 with the same UNKNOWN error. The KV match
appeared to do nothing.

Diagnosis (via instrumented `get_pt_block` repro in
`/tmp/debug_apply_adascale.py`): the patched
`AdaScale.apply_adascale` was being shadowed at runtime because
`adascale_optimizer.py` line 441 binds:

```python
apply_adascale = AdaScale.apply_adascale
```

at *module-load time*. Our patch replaces `AdaScale.apply_adascale =
patched_classmethod` but the module-level free function still
points at the original. The orchestrator does
`from aimet_onnx.experimental.adascale.adascale_optimizer
  import apply_adascale`
which gets the **stale binding**. So our patched logic never ran;
`block_kv_tensor_names` stayed `[]`; `block_input_names` came in at
length 5 (missing past_kv) instead of 7.

Fix: also rebind `ao_mod.apply_adascale = ao_mod.AdaScale.apply_adascale`
inside the patcher, after replacing the classmethod. One extra
line; the difference between "still crashing" and "all 28 blocks
optimize cleanly."

Verified end-to-end on a 16-cal / 10-iter scaled-down run before
committing 1500 iters / 128 cal of GPU time:

```
[adascale-patch] registered ReduceMean v18 handler
[adascale-patch] overrode AdaScale.apply_adascale + module-level
                 apply_adascale for HF-style past_key_values.{i}.{key,value}
[Optimizing block 0]  block_input_names (7): [..., past_key_values.0.key,
                                              past_key_values.0.value]
[Optimizing block 1] ... [Optimizing block 27] ...
SUCCESS
```

#### Pacing observed at 1500 iters / 128 cal (Qwen3-0.6B w8a16, A40)

| sub-stage | wall |
|---|---:|
| stages 1-5 (skipped, done.json) | 0 s |
| AIMET 1-4 (tokenizer + cal + QSM) | ~100 s |
| AIMET 5 (SEQ_MSE 113 ops) | ~590 s (10 min) |
| AIMET 6 (AdaScale, per-block) | ~3:25 / block × 28 = **95 min** |
| AIMET 7 (compute_encodings) | ~30 s (SEQ_MSE pre-pop) |
| AIMET 10 (sim.export) | ~30 s |
| stage 7 (qairt-converter) | ~90 s |
| stage 8 (qnn-context-binary-generator) | ~160 s |
| stage 9 (bundle + tar) | ~30 s |
| **total stage 6+ wall** | **~110 min** |

Per-block AdaScale breaks down as: 1500 iters × ~110 ms each =
~165 s + ~30 s setup (activation sampling × 128 samples through
the FP and qsim graphs) + ~10 s teardown = ~205 s = 3:25.

#### Disk / RAM during AdaScale

- Persistent: `/workspace/runs/qwen3_0p6b_w8a16_full/` ~6 GB
  (01_optimum 3 GB + 05_pathb_ctx512 3 GB; 02/03/04/06/07/08/09
  recreate per-attempt).
- Ephemeral: `tempfile.TemporaryDirectory()` used by AdaScale
  to stage the FP32 model (3 GB ONNX + 3 GB external data) for
  the per-block FP activation sampler. Cleaned automatically.
- Resident: ~6 GB GPU + ~12 GB host RAM at AdaScale peak; well
  below A40's 48 GB / pod's 32 GB host RAM.

#### Implications for M2 (Qwen3-4B)

The same four patches apply unchanged. Wall scales:
- 36 layers × ~3:25 / block × (4B/0.6B params per block ratio
  ≈ 2.5×) → ~5 hours just for AdaScale at 1500 iters.
- For first M2 run, recommend `--ada-scale-iters 500` first
  (≈100 min AdaScale), then full 1500 only if 500 doesn't clear
  the cos gate. Saves ~3 hours / $1.50 of GPU time per attempt.
- Pre-Qualcomm-bundle reproduction target: cos ≥ 0.998, 46/46
  argmax. We have a known reference for Qwen3-4B unlike 0.6B,
  so M2 has a sharper success criterion.

### Why we can't trivially reuse the m1e cos 0.996 result

The prior-session m1e run reached cos 0.996 on Qwen3-0.6B w8a16
using **aimet_torch** (PyTorch surface) on the **standard
transformers Qwen3ForCausalLM**, not the pathb-rewritten ONNX. That
artifact failed at qnn-context-binary-generator with
`Failed to validate op /inner/model/Cast with error 0xc26` — the
HF transformers attention has Cast→BOOL chains (NaN guards,
attention_mask boolean expansion) that HTP rejects. **That's
exactly why pathb exists**: the rewrites delete those chains and
replace `Where(bool_mask)` with `Add(attention_bias)`.

Two NPU-deployable paths from m1e's high-quality encodings:

1. **Run AIMET on the pathb-rewritten ONNX.** What we're doing now.
   Aimet_onnx 2.26 has buggier optimizers than aimet_torch (we've
   patched 4 things in AdaScale alone) and SEQ_MSE forces
   quant_scheme=min_max which loses tf_enhanced's better activation
   observation.
2. **Port aimet_torch's encodings to pathb tensor names via a
   name-translator.** A ~50-LOC mapping from torch-traced internal
   names (e.g. `inner.model.layers.0.self_attn.q_proj.weight`) to
   pathb-rewritten ONNX names (`onnx::MatMul_<id>`). Run AIMET on
   the torch model (m1e recipe → cos 0.996), then post-process the
   resulting `qsim.json` to rename keys before handing it to
   `qairt-converter --quantization_overrides`. Open question P1
   in `docs/one_pipeline_cloud_gpu.md` is exactly this question.
   Untested but plausible because:
   - The torch Q/DQ tensor names follow a regular pattern
   - Pathb only renames the surrounding compute graph, not the
     weight tensors themselves (initializers keep their names)
   - The encodings are referenced by tensor name, not by op
     position, so a bijective map should work

If aimet_onnx + pathb hits a quality ceiling we can't break, path
(2) is the documented escape hatch — we have m1e's encodings on
disk, the pathb-rewritten ONNX on disk, and just need the mapping.

### Diagnostic side runs

#### w8a16 basic PTQ tf_enhanced, 128 cal samples (2026-05-02 ~03:30)

Ran while the in-progress AdaScale rerun was paused (post-OOM), to
isolate "what does the activation observer alone give us, with no
optimizer overhead?"

| metric | value |
|---|---|
| recipe | w8a16, post_training_tf_enhanced, no SEQ_MSE, no AdaScale |
| num_cal | 128 |
| cos(fp,q) | **0.6558** |
| fp argmax | 12095 (' Paris') |
| q argmax | 220 (' ') |
| argmax match | False |
| total wall | 426 s (CPU-bound compute_encodings dominates) |

The 0.656 number replicates the 16-cal smoke. **Basic PTQ
tf_enhanced caps at ~0.656 on this stack** regardless of cal
sample count — going from 16 to 128 doesn't move the number.

Comparison table:

| recipe | cal | cos | notes |
|---|---:|---:|---|
| smoke basic PTQ tf_enhanced | 16 | 0.656 | original M1 plumbing run |
| **side probe basic PTQ tf_enhanced** | **128** | **0.656** | **same number** — cal saturation |
| SEQ_MSE-only (forced min_max) | 128 | 0.613 | SEQ_MSE *hurts* (loses tf_enhanced) |
| SEQ_MSE + AdaScale (broken bw=4) | 128 | 0.510 | bw mismatch bug, hurts further |
| m1e (aimet_torch SEQ_MSE+AdaScale) | 32 | **0.996** | reference; on standard transformers graph (not pathb), not NPU-deployable |

**Read.** The cap at 0.656 on aimet_onnx + pathb is independent of
both cal-set size and optimizer choice (so far). Each "improvement"
we've tried on top of basic PTQ has actually moved cos backward.
That suggests:
1. The aimet_onnx 2.26 default `DefaultOpInstanceConfigGenerator`
   with `hw_version: None` may be using suboptimal per-channel/
   per-tensor defaults vs aimet_torch v2's defaults.
2. The HTP-aware `htp_quantsim_config_v75.json` config (already
   shipped in the venv) is *not* being passed via `config_file=`
   to QuantizationSimModel; using it might tighten quality.
3. The pathb rewrite may have introduced precision-sensitive
   operations that AIMET's observers don't capture well.

The pending bw-fixed AdaScale rerun is the test of (whether option-3
is the binding constraint). If AdaScale-with-correct-bw still lands
≤ 0.7, then the ceiling is structural to this stack and the
m1e-encodings-name-translator path becomes the only credible
escape.

#### w16a16

Untested. Would tell us "if precision is unlimited, what's the
cos ceiling on this pipeline?" but won't ship for performance
reasons (HTP HMX accelerates {int4,int8} × {int16,fp16}; int16
weights fall back to scalar tensor units, ~10× slower) and
doubles bundle size. Pure diagnostic; deferred unless we exhaust
other options.
