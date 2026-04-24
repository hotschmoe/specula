# One-pipeline cloud GPU conversion — HF → Snapdragon X2E NPU bundle

**Status:** design doc, 2026-04-23. Complements
`docs/rent_cloud_compute.md` (which stays as the operational
runbook for a single rent session) by adding the architectural
case for folding the conversion into one reusable cloud-resident
script, plus the strategic basis for every design choice and the
pitfalls we expect along the way.

## Goal

Give specula one deterministic, reproducible "model conversion"
pipeline that takes an HF FP16/BF16 checkpoint and produces a
working Snapdragon X2E NPU context binary — and, in a follow-on
iteration, a matched GGUF for CPU/GPU targets. The pipeline runs
on one rented Linux + CUDA VM for ~3-6 hours at ~$3-10, end to end,
and leaves nothing stateful on the cloud host: every artifact
needed to re-run lives in-repo.

Target users: us (future specula sessions running Qwen3.5 / 3.6 /
Gemma4 graduations), and — stretch — any external reproducer on a
fresh Linux + CUDA box.

## TL;DR

1. **One script on a cloud x86 Linux + CUDA VM** runs the complete
   conversion pipeline: HF download → ONNX export → pathb rewrites
   → AIMET SEQ_MSE+AdaScale quantize → QAIRT partition / convert /
   quantize / ctx-bin-gen → tarball to NAS or scp home.
2. **Input:** HF model ID (`Qwen/Qwen3-4B`), arch adapter name,
   precision, ctx. **Output:** ready-to-load QNN context binary(s)
   + encodings.json + wrapper ONNX + manifest.yaml.
3. **Why cloud:** AIMET SEQ_MSE + AdaScale require CUDA
   (V100/A100 per Qualcomm's own `quantize.py` check). Everything
   else runs on X2E too, but bundling the whole pipeline into the
   cloud script makes re-running independent of X2E availability
   and trivially shareable.
4. **Compounding win:** one AIMET calibration run produces encodings
   that can (later) fuel both the NPU binary AND a matched GGUF
   for CPU/GPU via llama.cpp tooling. Calibration is the reusable
   asset; container formats are backend-specific.
5. **Cheap first validation:** run Qwen3-0.6B through the pipeline
   on a 24 GB card (~$1, ~1 hour) before spending A100 hours on
   Qwen3-4B. Retroactively validates Lever C's unresolved V/O
   collapse as a side effect.

---

## Strategic basis (why this shape)

### Q1: Does w4a16 help GPU/CPU too, not just NPU?

Yes for the **bandwidth** win (every LLM decoder is weight-fetch-
bound at TG); no for the **container** (each backend has its own
format):

| backend | production container | w4 hardware path | notes |
|---|---|---|---|
| Hexagon NPU | QNN context binary (`.bin`), uint8 KV, uint16 activations | HMX int4 MatMul — the published ~3× vs w8 | Qualcomm's blessed format |
| Adreno GPU (llama.cpp OpenCL) | GGUF Q4_K_M / Q4_0 | Dequant-on-fly to fp16; no dedicated int4 compute in current OpenCL kernels | Bandwidth-only win today |
| ARM CPU (llama.cpp + NEON) | GGUF Q4_K_M | NEON dot-products on Q4_K_M directly; SME2 + KleidiAI adds a real int4 path once unstalled (B8) | Already our Phase-1 baseline |

Implication for pipeline design: **the calibration work is backend-
portable, the container isn't.** AIMET emits per-tensor scales and
offsets; those scales can drive a QNN context binary *or* (via a
small translator) a GGUF Q4_K_M. One cloud run, multiple backends,
if we invest in the double-export step.

For **prefill** (compute-bound), all three backends care less about
weight bitwidth. The w4a16 win is a TG story first, prefill second.

### Q2: Everything in one cloud VM?

Yes. `docs/qualcomm_reproduction_4b.md` shows the conversion
pipeline (optimum export, pathb rewrites, QAIRT converter /
quantizer / ctx-bin-gen) already runs on X2E under Prism. The only
cloud-*necessary* step is the AIMET SEQ_MSE + AdaScale quantize
which needs CUDA.

But for **reproducibility**, the whole thing should live in the
cloud script:

- Easier for a fresh outside-team reproducer (no X2E required).
- Decouples conversion from X2E availability (X2E becomes only a
  run target, not a build target).
- Makes the recipe immutable between conversions (the cloud script
  fully describes the build; no "you also need to run these steps
  on X2E afterwards").

Trade-off: x86 Linux + CUDA adds ~$3-10 per conversion. At research
cadence of a few conversions per year, negligible vs engineer time.

### Q3: Base FP or pre-quantized HF checkpoint?

**Unambiguously FP (BF16) from the original provider.** AIMET needs
the full-precision weight distribution + full-precision forward
pass on calibration samples to pick optimal scales. Starting from
a pre-quantized variant wastes information:

- **GPTQ / AWQ HF checkpoints**: int4 weights with group scales
  already baked. AIMET would either treat them as fp (introducing
  dequant artifacts before recalibration) or need a custom
  encodings translator (not a production path).
- **GGUF**: llama.cpp-specific; would require dequant → ONNX
  export → AIMET. Full round-trip waste.
- **FP16/BF16 original** (e.g. `Qwen/Qwen3-4B`): AIMET sees true
  weight + activation distributions. This is the shape AIMET is
  designed for.

Source-of-truth is always the provider's FP release. Qwen team
ships BF16 for Qwen3 family; assume the same for Qwen3.5, 3.6,
Gemma4 until proven otherwise. Check the repo's `config.json`
`torch_dtype` field before cloud-running.

### Q4: What calibration technique stack gives best results?

Qualcomm's `qai_hub_models.models.qwen3_4b.quantize` default is
**SEQ_MSE + AdaScale**. Empirically sufficient for Qwen3-4B (their
shipping bundle: cos 0.9998 on Part 2 L11 per our oracle probe,
3.1 GB bundle vs our 4.8 GB basic-PTQ attempt).

Escalation ladder when the default isn't enough (e.g. if Qwen3.5-8B
shows V/O-projection collapse at w4 like Qwen3-0.6B did):

1. **SEQ_MSE + AdaScale** (baseline; always on)
2. **+ SmoothQuant** (shifts activation magnitude into weights;
   directly targets the V/O collapse pattern we diagnosed in
   `w4a16_investigation.md` session 17)
3. **+ AWQ** (weight-only scale search driven by activation
   magnitudes; same class of fix via different mechanism)
4. **+ per-tensor overrides** pinning V/O to w8 (blunt but
   reliably closes the gap; slightly larger binary)
5. **QAT** (quantization-aware training; nuclear — separate
   fine-tune run per model)

First pipeline iteration: ship step 1. Steps 2-4 become flags on
the cloud script; step 5 remains special-case.

---

## Pipeline architecture

### Single script contract (proposed)

```
scripts/convert_hf_to_htp.py                   # does not exist yet
  --model-id Qwen/Qwen3-4B                    # HF repo ID
  --arch-adapter qwen3                        # qwen3 | qwen35 | gemma4 | llama3 | ...
  --precision w4a16                           # w4a16 | w8a16 | w4a16-mixed
  --ctx 512                                   # context length to pin
  --ar 1                                      # AR=1 (decode) | AR=128 (prefill) | both
  --calibration seq_mse+ada_scale             # [+ smoothquant, + awq, + pin_vo]
  --num-samples 128                           # calibration set size
  --partition auto                            # follows arch-adapter's seam map
  --output rclone://nas/specula-builds/       # or s3://, file:///workspace/out
```

### Stages

```
[cloud VM, x86 Linux + CUDA]
├── 1. install deps (one-shot; cache in persistent volume)
│       qai-hub-models[arch] + aimet_onnx + torch+cu121 + QAIRT 2.42 SDK
├── 2. huggingface-cli download <model-id>       → ./hf/
├── 3. optimum export → ONNX                      → ./onnx-optimum/
├── 4. arch-adapter rewrites:
│       rewrite_qwen3_htp.py                      → ./onnx-staged/
│       rewrite_qwen3_htp.py --fold pathbmask     → ./onnx-pathbmask/
│       rewrite_qwen3_pathb.py                    → ./onnx-pathb/
│       pin_shapes_qwen3_Xb.py --ctx N            → ./onnx-pathb-ctxN/
├── 5. capture_calibration_Xb.py                  → ./cal/bundle.npz
├── 6. AIMET SEQ_MSE + AdaScale  (CUDA step)
│       QuantizationSimModel against pathb ONNX + cal bundle
│       → ./aimet/encodings.json, ./aimet/model.qdq.onnx
├── 7. partition per arch-adapter seam map
│       → ./parts/part1.onnx ... partK.onnx (+ per-part encodings)
├── 8. qairt-converter × K parts                  → ./dlc-fp32/
├── 9. qairt-quantizer × K parts                  → ./dlc-w4a16/
│       (consumes AIMET encodings via --quantization_overrides)
├── 10. qnn-context-binary-generator × K parts
│        (weight-shared bundle config)            → ./bin/
├── 11. build wrapper ONNX(s) for ORT-QNN
├── 12. manifest.yaml with: model-id, git sha, arch-adapter name,
│        calibration flags, QAIRT version, AIMET version,
│        per-tensor encodings summary
└── 13. tar + upload to --output destination

[X2E; consumer side, unchanged]
Fetch bundle → preflight load via ORT-QNN → sweep harness.
```

### Arch adapter layer

Each arch adapter is a Python module encoding:

- Which optimum export flags to use.
- The rewrite chain (rotary hoist, additive mask, etc.) — may
  reuse Qwen3's across close-family models.
- Partition seam map (layer indices where bin splits go).
- IO dtype convention (uint8 past_kv / uint16 activations by
  default — Qualcomm's reference).
- QAIRT compile flags specific to the arch (if any).

Today's `rewrite_qwen3_htp.py` + `rewrite_qwen3_pathb.py` +
`pin_shapes_qwen3_4b.py` become `scripts/arch_adapters/qwen3.py`.
Qwen3.5 / 3.6 re-use with per-model `--model-stem`. Gemma4 and
Llama3 are new adapters (W6.a / W6.b in `docs/roadmap.md`).

### Double-export for CPU/GPU (follow-on, not day-one)

After the NPU binary lands, `scripts/aimet_to_gguf.py` (also
doesn't exist yet) consumes the same encodings.json and produces a
GGUF with matched scales. This gives the CPU/GPU target path the
same calibration quality as the NPU path — closing the "is our
spec decoder's draft and target on the same quantization basis?"
question. Not day-one; tracked in W9.b of the roadmap.

---

## Cloud provider selection

Operational runbook for the rent itself (install commands, SSH
pattern) lives in `docs/rent_cloud_compute.md`. Provider summary
for the AIMET GPU step:

### Recommended

- **RunPod (Community Cloud)** — A100 40GB at ~$1.00-1.50/hr
  (spot often lower). Pre-built PyTorch 2.4+ containers ship with
  CUDA 12.1 matching the `aimet_onnx+cu121` wheel. Pay-per-minute
  billing, no long commitment. **Primary choice.**
- **Lambda Labs** — A100 40GB at ~$1.29/hr on-demand. Reliable,
  simpler UX than RunPod, slightly more expensive. Good fallback
  if RunPod availability is spotty.

### Acceptable

- **Vast.ai** — A100 40GB at ~$0.80-1.50/hr. Marketplace model,
  variable reliability. Cheapest when you get lucky.
- **GCP `a2-highgpu-1g`** — $3.67/hr on-demand. Expensive but
  simplest path if you already have GCP credits + IAM set up.

### Not sufficient (documented to OOM for ≥4B SEQ_MSE)

- A10G / L4 (24 GB) — fits basic PTQ, not SEQ_MSE reliably on 4B+
- RTX 4090 (24 GB) — same
- V100 16 GB — below Qualcomm's documented threshold
- T4 (16 GB) — below threshold

24 GB cards are fine for Qwen3-0.6B and likely Qwen3-1.7B drafter
runs (AIMET VRAM scales with model size; Qualcomm's V100/A100
baseline was set for the 4B target). Pipeline flag for future:
`--gpu-size-gb 24` picks a smaller-model path with reduced
batch / sample parallelism.

### Budget

Per-conversion spend:

| model | VRAM needed | typical hourly | wall | per-run cost |
|---|---|---|---|---|
| Qwen3-0.6B (SEQ_MSE) | 24 GB | $0.50/hr | 1-2 h | $0.50-1 |
| Qwen3-4B (SEQ_MSE) | 40 GB | $1.30/hr | 2-5 h | $3-7 |
| Qwen3-8B (SEQ_MSE) | 80 GB | $1.90/hr | 4-8 h | $8-15 |
| Gemma4 family (TBD) | TBD by size | — | — | — |

Budget $10-20 per model-family graduation. Trivial vs the cost
of stalling on PTQ quality.

### CPU-only fallback (no GPU, ~$0)

Per the Qualcomm source (`qai_hub_models/_shared/llm/quantize.py`),
`allow_cpu_to_quantize=True` is set for Qwen3-4B — basic PTQ runs
on CPU without `--use-seq-mse --use-ada-scale`. ~10 hours on a
fast desktop CPU. **Basic PTQ only** → same quality tier as AI
Hub cloud APIs which we already measured at worse-than-our-local
PTQ. Useful only as a reference datapoint, not a shippable path.

---

## Step-by-step runbook (first conversion — until the script exists)

This is the operational path for the **first** run, before
`scripts/convert_hf_to_htp.py` is written. It's what we actually
type. Once it works end to end, lift into the script.

Full expanded rent-session guide is in `docs/rent_cloud_compute.md`;
summary + deltas only below.

### 0. Validate cheap first — Qwen3-0.6B on 24 GB card

Rent a 24 GB pod (~$0.50/hr, ~1-2 h). Run the pipeline on
Qwen3-0.6B (our currently-stalled Lever C drafter). Expected
outcome: SEQ_MSE + AdaScale closes the V/O projection collapse
that basic PTQ left at cos 0.33. **If yes:** pipeline is
validated, and we've retroactively closed Lever C positive. **If
no:** V/O collapse is a fundamental 0.6B-class property,
documented with full-gate evidence; graduate to Qwen3.5 with that
prior.

Cost of this check: ~$1. Value: decisive signal on pipeline + on
the whole w4a16 investigation.

### 1. Rent pod (4B target)

- RunPod → PyTorch 2.4.1 template → 1× A100 40GB → **100 GB
  persistent storage** (AIMET intermediates are large) → deploy.
- SSH command from RunPod's connect tab. Save it.

### 2. Install deps

```bash
# Qualcomm's published line (from rent_cloud_compute.md §3):
pip install "qai-hub-models[qwen3-4b]" onnxruntime-gpu==1.23.2 \
  https://github.com/quic/aimet/releases/download/2.26.0/aimet_onnx-2.26.0+cu121-cp310-cp310-manylinux_2_34_x86_64.whl \
  -f https://download.pytorch.org/whl/torch_stable.html

# QAIRT 2.42 SDK (x86 Linux build):
curl -sL -A "Mozilla/5.0" \
  https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.42.0.251225/v2.42.0.251225.zip \
  -o qairt-2.42.zip
unzip qairt-2.42.zip -d /opt/qairt-2.42
source /opt/qairt-2.42/bin/envsetup.sh
export PATH=/opt/qairt-2.42/bin/x86_64-linux-clang:$PATH

# Sanity
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import aimet_onnx; print(aimet_onnx.__version__)"   # expect 2.26.0
qairt-converter --version
```

### 3. Configure AI Hub token

```bash
qai-hub configure --api_token $AIHUB_TOKEN
```

Needed for `qai-hub-models` package imports; AIMET itself pulls
weights from HF, not AI Hub.

### 4. Run the AIMET step

```bash
python -m qai_hub_models.models.qwen3_4b.quantize \
  --output-dir /workspace/out/qwen3_4b_seqmse \
  --precision w4a16 \
  --use-seq-mse \
  --use-ada-scale \
  --num-samples 128 \
  --seq-mse-num-samples 128 \
  --ada-scale-num-samples 128
```

`nvidia-smi` in a second shell should show 30-38 GB VRAM used on
40 GB card. Runtime 2-5 h.

### 5. Run the QAIRT toolchain

Cross-reference `docs/qualcomm_reproduction_4b.md` Phase 5 for the
4-part partition scheme and the seam map. Port the Windows-path
commands to Linux:

```bash
# per-part converter
for part in 1 2 3 4; do
  qairt-converter \
    --input_network parts/part${part}.onnx \
    --output_path dlc-fp32/part${part}.fp32.dlc \
    --preserve_onnx_output_order --remove_unused_inputs \
    --quantization_overrides aimet/encodings.json
done

# per-part quantizer (consumes AIMET encodings via the overrides above)
for part in 1 2 3 4; do
  qairt-quantizer \
    --input_dlc dlc-fp32/part${part}.fp32.dlc \
    --output_dlc dlc-w4a16/part${part}.w4a16.dlc \
    --input_list cal/part${part}_input_list.txt \
    --weights_bitwidth 4 --act_bitwidth 16 \
    --use_per_channel_quantization --use_per_row_quantization
done

# multi-part weight-shared bin
qnn-context-binary-generator \
  --backend libQnnHtp.so \
  --dlc_path dlc-w4a16/part1.w4a16.dlc,dlc-w4a16/part2.w4a16.dlc,dlc-w4a16/part3.w4a16.dlc,dlc-w4a16/part4.w4a16.dlc \
  --binary_file bin/qwen3_4b_w4a16_seqmse \
  --config_file compile_config.json           # weight_sharing_enabled: true
```

Final `.bin` files land in `./bin/`.

### 6. Transfer + shut down

```bash
tar czf qwen3_4b_w4a16_seqmse_bundle.tgz bin/ aimet/encodings.json \
    manifest.yaml wrappers/*.onnx
# upload to NAS via rclone / scp / runpod UI
```

**Immediately stop the pod.** An A100 at $1.50/hr costs $36/day
if forgotten. Set a phone alarm. Non-negotiable.

---

## Pitfalls

Catalog of every gotcha we've hit or anticipate. Add as we run the
pipeline.

### P1. AIMET tensor names vs our pathb graph

`qai_hub_models.models.qwen3_4b.quantize` runs against Qualcomm's
**internal** pathb ONNX topology. The encodings.json it emits may
not match our rewritten pathb tensor names 1:1.

Mitigations:
- **Option A:** feed our pathb ONNX directly into AIMET
  `QuantizationSimModel` (bypass the Qualcomm wrapper). ~4 h of
  AIMET API to work through; produces encodings against our exact
  names.
- **Option B:** author a name-translation layer (map Qualcomm's
  tensor names → ours by structural position).

Option A is cleaner long-term. First-run pragmatic: try
Qualcomm's wrapper, diff the names. If close-enough auto-match,
ship; else switch to Option A.

### P2. AIMET's output format vs QAIRT's expectations

AIMET emits ONNX + encodings.json with AIMET-specific schema.
QAIRT's `qairt-quantizer --quantization_overrides <json>` accepts
a JSON format we've been best-guess-authoring (see
`w4a16_investigation_continued.md` Phase 5.5.1 progress log — x86
team is still iterating on schema per `qairt-quantizer --help`).

Unknown whether AIMET's encodings.json is exactly QAIRT-consumable
without translation. If not: both are JSON dicts of per-tensor
`{scale, offset, bitwidth, is_symmetric}`, so a translator is
tractable (~50 LOC). Budget one ghost cycle.

### P3. Multi-part weight-shared compile (10-graph bundle)

Qualcomm's Qwen3-4B bundle is 4-part weight-shared with 10 graphs
per bin (5 ctx × 2 AR). Getting all 10 graphs weight-shared in a
single `qnn-context-binary-generator` invocation is flag magic
encoded in Qualcomm's sample scripts, not obvious from `--help`.

First-iteration pragma: ship single-graph single-ctx single-AR per
part. Matches our existing 0.6B pattern. Add multi-graph
weight-sharing when Phase 6 W1.b (AR=128 prefill on NPU) starts.

### P4. HF rate limit on first run

Qwen3-4B is ~17 GB. HF rate-limits anonymous downloads. Pre-
authenticate: `huggingface-cli login` before the download. Also:
RunPod pods lose state between sessions unless you put the
download directory on a persistent volume.

### P5. Disk I/O during SEQ_MSE

AIMET writes many intermediate tensors during SEQ_MSE. 100 GB
disk is the floor; more is better. Silent ENOSPC partway through
is a real failure mode per Qualcomm's own warnings. Explicitly
request 100+ GB at RunPod deploy time — do not rely on defaults.

### P6. Pod-forgetting = bill shock

A100 at $1.50/hr = $36/day. Set an alarm. If RunPod offers a
schedule-auto-stop, configure it at deploy time. Budget safety
beats memory.

### P7. QAIRT version drift

Cloud script produces binaries via QAIRT 2.42. ORT-QNN on X2E is
1.24.4, which **requires** 2.42 (see
`reference_ort_qnn_qairt_match.md`). If we ever bump cloud QAIRT
to 2.45 without also updating ORT-QNN on X2E, binaries fail with
`LoadCachedQnnContextFromBuffer Error 5000`.

Mitigation: pin QAIRT version in the cloud script manifest;
fail-fast at the start of the script if `qairt-converter --version`
doesn't match the manifest's declared version.

### P8. Cloud VM x86_64 Linux vs X2E Windows ARM

QAIRT 2.42 runs on x86_64 Linux cleanly; binaries it produces are
platform-neutral (QNN context binary format is HTP-SoC-keyed, not
host-OS-keyed). No cross-compile friction expected.

But: if we ever introduce host-specific helpers (Linux-only Python
libs with no Windows equivalent), we lose the "also runs on X2E"
escape hatch. Keep the cloud path and the X2E-native path (per
`qualcomm_reproduction_4b.md`) both functional as a matter of
discipline.

### P9. Architecture adapter drift

Qwen3.5 / 3.6 / Gemma4 may use different rotary schemes, different
norms (RmsNorm variants), different attention (sliding window,
MoE gating). Our rewrite scripts assume Qwen3-family rotary hoist
geometry. Per-arch adapter must be audited before first run on a
new family — not assumed to transfer.

Per-family cost: ~1 session each. Track in W6 of
`docs/roadmap.md`.

### P10. Calibration distribution vs inference distribution

AIMET calibrates on whatever samples we feed. If calibration is
code-heavy (humaneval subset) and inference is conversational, the
activation ranges AIMET picks may not cover the real use case →
degraded cos on out-of-distribution prompts.

Mitigation: broader calibration set (humaneval + chat + arXiv
prose mix), or re-calibrate for new use cases. For the first run:
reuse `bundle_a_pathb_ctx256`'s distribution and document in the
manifest so a future cos regression can be attributed correctly
(calibration-set issue vs pipeline issue).

### P11. First-run SEQ_MSE takes longer than the docs claim

Qualcomm's rent doc says 2-5 h for 4B SEQ_MSE. HF downloads, AIMET
dep installs, and first-time cold caches can stack another 1-2 h
on top. Budget 6-8 h total wall-clock for the first run; 3-5 h for
subsequent runs (with persistent volume caching).

### P12. `qai_hub_models.models.qwen3_4b.quantize --help` may lie

The flag names (`--use-seq-mse`, `--use-ada-scale`,
`--seq-mse-num-samples`) are from our read of the rent doc +
Qualcomm's published install line. Qualcomm's CLI surface changes
between `qai-hub-models` releases. Run `--help` first on the
actual installed version before copy-pasting our command line.
Update `rent_cloud_compute.md` Step 5 if flags diverge.

---

## Future-model extension

### Qwen3.5-8B

Expected smooth. Same Qwen family → same arch adapter (`qwen3.py`
with `--model-stem qwen35-8b`). Larger model → 80 GB VRAM card
(+$0.40/hr over 40 GB card). First graduation target after Phase
5.5 Lever C closes.

### Qwen3.6

Architecture audit required first (W6.a). Probable same rotary,
same norm, same attention as 3.5 → adapter reuse likely. If Qwen
team changes anything structural, adapter fork needed.

### Gemma4

Definitely new adapter (W6.b). Different RoPE variant likely,
different norm, possibly sliding-window or multi-query attention.
Budget 1-2 sessions for adapter development before the first
cloud run.

### Llama3 / Phi / Mistral / TinyLlama

Each new family: ~1 session adapter development + 1 cloud run.
Covered in W9.c (model-family survey).

---

## Validation gates

Every cloud run outputs artifacts that must pass the same checks
on the X2E side before the bundle is considered shippable:

1. **Structural.** Binary loads on ORT-QNN 1.24.4.
   `scripts/npu_load_qwen3_bin.py` preflight reports expected
   input/output counts, all dtypes match the manifest.
2. **Numerical.** Short-prompt probe
   (`scripts/npu_short_prompt_probe.py`) on fib-p0: cos ≥ 0.95 vs
   CPU fp32 reference. Multi-step match 100% on 3 consecutive
   generation steps.
3. **Throughput.** Steady-state latency
   (`scripts/probe_npu_steady_state_latency.py`) lands in the
   expected band for the model size (per the side-quest
   extrapolation: ~0.6 ms/layer × layer count).
4. **End-to-end.** Async-pipelined sweep
   (`scripts/sweep_npu_spec.py --mode async-pipelined -n 200`).
   Beats the current same-target high-water mark (for Qwen3-8B
   target today: 18.12 t/s AC k=2; for future targets, the
   respective baseline).

Run that passes 1-3 but fails 4: "correct binary, not a product"
— still useful as pipeline validation. Run that fails 2:
pipeline regression; debug before moving to the next model.

---

## Open questions (resolved by doing)

- Does AIMET's encodings.json format QAIRT-consume directly, or
  does it need a translator? (P2)
- Do Qualcomm's quantize.py tensor names match our pathb graph
  1:1? (P1)
- Does the multi-graph weight-shared bundle flag work from QAIRT
  CLI without the Qualcomm sample scripts? (P3)
- For Qwen3-0.6B specifically: does SEQ_MSE + AdaScale close the
  V/O projection collapse we saw with plain PTQ (cos 0.33 →
  ≥ 0.95)? Retroactively un-sticks Lever C if yes.
- For Qwen3-4B specifically: does our reproduction hit Qualcomm's
  65% argmax agreement ceiling (`qualcomm_reproduction_4b.md`
  Phase 5o) or push past it once SEQ_MSE lands? Decisive signal
  on whether their proprietary calibration set matters or the
  pipeline does.

Each answered in the first cloud run. Capture in the run manifest
for future comparison.

---

## Relation to other docs

- **`docs/rent_cloud_compute.md`** — operational runbook for the
  rent session itself (provider steps, exact pip lines, SSH). This
  doc references it; don't duplicate that content here.
- **`docs/qualcomm_reproduction_4b.md`** — the X2E-native version
  of the same pipeline, phase-by-phase. Validates every step
  (except the CUDA-only AIMET step) runs on ARM Windows under
  Prism.
- **`docs/w4a16_investigation.md`** / **`_continued.md`** — why
  we're investing in this pipeline (the V/O-projection collapse
  at w4 on 0.6B that SEQ_MSE + AdaScale is expected to close).
- **`docs/roadmap.md`** W9 — the workstream that converts this
  design doc into `scripts/convert_hf_to_htp.py`. Script lives
  under W9.b.
- **`results/qwen3_4b_genie_w4a16_probe.md`** — proof that
  ORT-QNN accepts the Qualcomm-style full-quant IO convention
  the cloud pipeline will produce.

---

## Recommended pre-rent work (before burning any pod minutes)

1. **Scope `scripts/convert_hf_to_htp.py` skeleton** — even just
   the CLI argparse + stage skeleton + per-arch adapter
   interface. Gives the rent session a real script to exercise
   instead of ad-hoc commands; test plan is "fill in each stage
   until the whole thing runs."
2. **Pick the validation model** — Qwen3-0.6B first on a 24 GB
   card is the highest-value cheap run. Confirm the 0.6B weights
   + our pathb ONNX live where the cloud script expects them.
3. **Write the arch adapter for Qwen3** — wraps
   `rewrite_qwen3_htp.py` / `rewrite_qwen3_pathb.py` /
   `pin_shapes_qwen3_*b.py` behind a single importable interface.
   Factors out the `--model-stem` dispatch that's currently
   per-script.
4. **Pre-auth HF + copy HF cache location** — `huggingface-cli
   login` on the pod on first use, point `HF_HOME` to the
   persistent volume. Avoids re-download on subsequent sessions.
5. **Manifest schema** — what goes in `manifest.yaml`? Model-id,
   git sha, arch-adapter version, calibration flags + sample
   counts, QAIRT + AIMET + torch versions, per-tensor encodings
   summary (min/max/mean scale per layer type). Write the schema
   now so the cloud script emits the right thing on first run.

Estimated pre-rent investment: ~1 focused session. Produces:
- A script skeleton worth exercising.
- A Qwen3 adapter that transfers directly to 3.5 / 3.6.
- A reproducibility manifest spec that every future conversion
  inherits.

Rent session then becomes: run script against Qwen3-0.6B, gate on
cos ≥ 0.95; run against Qwen3-4B, gate against Qualcomm's oracle;
commit outputs + manifests; done.

---

## Update log

- **2026-04-23** — Doc created. No cloud run yet; design + pre-rent
  scoping only. Next action: either scope
  `scripts/convert_hf_to_htp.py` before renting (recommended), or
  run the first session ad-hoc to validate the AIMET→QAIRT
  handoff, then lift into the script.
