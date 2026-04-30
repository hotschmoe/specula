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
