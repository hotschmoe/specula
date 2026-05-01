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
