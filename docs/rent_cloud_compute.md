# Renting cloud Linux compute for NPU export and quantization

Two distinct rental scenarios this project hits — different goals,
different hardware shapes, different costs. Pick by what you're trying
to produce, not by reflex.

## Decision tree — do I even need to rent?

Before reading further, check the cheaper-or-free alternatives:

1. **Precompiled X2 Elite Genie bundle on Qualcomm's CDN.** Some models
   ship a chipset-specific bundle directly. Look for
   `release_assets.json` in `huggingface.co/qualcomm/<MODEL>` — if X2
   Elite is listed under `chipset_assets`, just download the zip.
   *Example:* Qwen3-4B (`huggingface.co/qualcomm/Qwen3-4B/raw/main/release_assets.json`
   has direct S3 URLs for X2 Elite, X Elite, 8 Elite, QCS9075, …).
   **No rental, no AI Hub compile job needed.**

2. **Pre-quantized AIMET ONNX intermediate on the public CDN.** Most
   non-gated `qai-hub-models` ship a pre-quantized bundle at
   `qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/<model_id>/v2/<model_id>.zip`.
   The `qai_hub_models.models.<model_id>.export` command auto-downloads
   it (~15-30 GB), then uploads to AI Hub Workbench for the X2-Elite-
   specific compile. *No 150 GB RAM materialization, no AIMET-ONNX-
   Linux requirement* — runs end-to-end on Windows-on-ARM with ~100 GB
   free disk. *Example:* Qwen2.5-7B-Instruct works this way (verified
   2026-04-25). **No rental needed; basic PTQ quality only.**

   To check before launching an export: `curl -sI "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/<model_id>/v2/<model_id>.zip"`.
   200 = pre-quantized, 403 = not published.

3. **Gated upstream weights blocking #2.** Models with restrictive
   upstream licenses (Llama family, some Mistral) **cannot** have a
   public pre-quantized intermediate — Qualcomm can't legally
   redistribute even a quantized version. The export then falls through
   to local FP16 materialization, which needs ~150 GB RAM+swap.
   **This is Scenario A below.** *Example:* Llama-3.1-8B-Instruct.

4. **Want better than basic PTQ quality?** Sequential MSE + AdaScale
   close the cos-divergence gap on weight-sensitive layers. They
   require CUDA + 24-40 GB VRAM. **This is Scenario B below.**
   *Example:* Qwen3-4B Phase 5p quality work.

## The two rental scenarios at a glance

| | Scenario A: high-RAM CPU box | Scenario B: CUDA GPU box |
|---|---|---|
| **Goal** | Run AI Hub Workbench export for a model that has no public pre-quant intermediate (gated weights) | Run SEQ_MSE / AdaScale on top of basic PTQ to close quality gap on a model we already export |
| **Output** | basic-PTQ AIMET ONNX → uploaded → cloud compile → Genie bundle | better-calibrated `encodings.json` → fed back into qairt-converter on the X2E |
| **OS** | Linux x86_64 (AIMET-ONNX wheels are Linux-only; Windows fallback warns and may degrade) | Linux x86_64 (same constraint plus CUDA driver) |
| **RAM** | 150 GB+ recommended (peak during FP16 materialization of an 8B-class model); 192 GB headroom is comfortable | 64 GB system + the GPU's VRAM |
| **GPU** | none required — the local box just materializes FP16 weights and runs ONNX export; quantization-aware compile is on Qualcomm's cloud | A100 40GB minimum for SEQ_MSE on 4B; 80GB for headroom or larger models |
| **Disk** | ~250 GB (HF cache for FP16 weights + AIMET intermediate + upload staging) | ~100 GB (HF cache + AIMET intermediate writes) |
| **Cost** | ~$1-2/hr × 2-4 hrs = **$2-8 one-time** | ~$1.50/hr × 3-6 hrs = **$5-10 one-time** |
| **Bottleneck** | network (FP16 weight download, then 30 GB upload to AI Hub) | GPU (SEQ_MSE iteration over weight tiles) |
| **When to skip** | Whenever path #2 above works | Whenever basic PTQ is good enough for the workload |

The two scenarios can compose on a single rental: a CUDA box has plenty
of system RAM too, so renting a GPU instance and running both jobs
back-to-back is fine if you want both deliverables in one session.

## Scenario A: high-RAM CPU box for FP16 export

### When this scenario applies

You hit this when path #2 of the decision tree fails — typically a
gated upstream model (Llama-3.1-8B, Llama-3.2-3B, etc.). The
`qai-hub-models` export then falls through to:

1. Pull FP16 weights from HuggingFace (gated, needs auth).
2. Materialize the model in memory (~16 GB just for the safetensors,
   peaks at 60-150 GB during ONNX trace + AIMET-encoding insertion).
3. Run AIMET-ONNX to insert quant-sim ops + emit encodings.
4. Upload the resulting ONNX + encodings to AI Hub Workbench.
5. AI Hub compiles the model for your target chipset (~hours,
   no local compute).
6. Download the Genie bundle (~5-6 GB).

Steps 2-3 are the OOM-prone phase on a 48 GB Windows-on-ARM box. They
also require AIMET-ONNX, which has Linux-only wheels (the Windows
fallback warns and may produce a degraded encoding). On a Linux box
with 192+ GB RAM both issues vanish and the export becomes routine.

### Sizing

| provider + instance | vCPU | RAM | $/hr (on-demand) | $/hr (spot) | notes |
|---|---:|---:|---:|---:|---|
| AWS `r7a.8xlarge` | 32 | 256 GB | $1.93 | ~$0.65 | AMD Genoa, fast single-thread |
| AWS `m7a.16xlarge` | 64 | 256 GB | $3.71 | ~$1.20 | more cores than needed for export |
| GCP `n2d-highmem-32` | 32 | 256 GB | $1.85 | ~$0.55 | similar shape |
| Vast.ai (CPU-only listings) | varies | 192-512 GB | $0.40-1.50 | n/a | marketplace, variable, faster spin-up |
| Lambda Labs CPU-only | 16-64 | 128-256 GB | $0.50-1.50 | n/a | only some skus carry enough RAM |

Recommendation: **AWS `r7a.8xlarge` spot** if you have AWS already
configured (~$1.30 for a 2-hour Llama-3.1-8B export end-to-end), or
**Vast.ai** if you want the cheapest possible thing and don't mind a
manual setup. RunPod's catalog is mostly GPU instances; their CPU-only
options are not consistently cheaper than the cloud big-three at this
RAM tier.

**Don't use less than 128 GB RAM** for an 8B-class export. The
qai-hub-models warning of "150 GB RAM + swap recommended" is real;
128 GB without swap occasionally OOMs during the AIMET trace. 192-256 GB
is comfortable headroom.

### Setup (~10-15 min)

```bash
# Pick a Linux x86_64 instance (Ubuntu 22.04 / 24.04 LTS works).

# Install uv (or use pip directly with python 3.10).
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repo (or just rsync / scp the parts you need —
# we only need pyproject.toml + the export command, not the
# whole repo).
git clone https://github.com/<your>/specula.git
cd specula

# Create the export venv.
uv venv .venv-cloud-export --python 3.10
VIRTUAL_ENV=.venv-cloud-export uv pip install \
    "qai-hub-models[llama-v3-1-8b-instruct]" \
    "qai-hub-models[qwen2-5-7b-instruct]"

# Configure HF auth — required for the gated Llama weights.
.venv-cloud-export/bin/hf auth login   # interactive; paste a read-scope token

# Configure AI Hub.
.venv-cloud-export/bin/qai-hub configure --api_token YOUR_TOKEN
```

### Run the export

```bash
mkdir -p out

PYTHONIOENCODING=utf-8 PYTHONUTF8=1 \
.venv-cloud-export/bin/python -m qai_hub_models.models.llama_v3_1_8b_instruct.export \
    --chipset qualcomm-snapdragon-x2-elite \
    --device-os 11 \
    --context-length 4096 \
    --skip-profiling \
    --skip-inferencing \
    --synchronous \
    --output-dir out/llama_v3_1_8b-genie-w4a16-qualcomm_snapdragon_x2_elite

# Watch RAM with `htop` in a second shell.
# Expected peak: 100-140 GB during the AIMET-ONNX trace + insertion step.
# If it OOMs, the instance was undersized — bump RAM, redo.
```

`--synchronous` keeps the local process alive until the AI Hub cloud
compile finishes (~4-6 hr for an 8B model) and the bundle is
downloaded. Run inside `tmux` / `screen` so an ssh dropout doesn't
kill the job.

### Transfer the bundle home + tear down

```bash
# Tar + scp back to the X2E (the bundle is ~5-6 GB for an 8B model).
cd out
tar czf llama_v3_1_8b-genie-w4a16-x2-elite.tar.gz \
    llama_v3_1_8b-genie-w4a16-qualcomm_snapdragon_x2_elite/

# From your local laptop:
scp <user>@<cloud-host>:/path/to/out/llama_v3_1_8b-genie-w4a16-x2-elite.tar.gz \
    models/qualcomm-llama-v3-1-8b-ref/

# Then on the cloud box: shut it down.
sudo poweroff
# AWS: also stop or terminate the instance from the console — `poweroff`
#      stops the OS, not the billing.
```

The bundle is now bench-able locally with the same runner as the 4B
baseline (just point `--bundle-dir` at the new path).

### Gotchas

- **Linux Genie bundle ≠ Windows Genie bundle?** No — the bundle is
  chipset-tied, not OS-tied. The `.bin` partitions are HTP bytecode
  consumed by `genie-t2t-run.exe` on Windows-on-ARM at runtime. The
  Linux export just produces those bytes.
- **HF token leakage**: never pass the token as a CLI arg; it lands in
  shell history. Always use `hf auth login` interactively.
- **AI Hub credit usage**: a single 8B compile costs ~1-2 hours of cloud
  compile credits (free tier on AI Hub Workbench is generous as of
  2026-04). Check your credit balance at workbench.aihub.qualcomm.com
  before starting if you're not sure.
- **Don't leave the box running**. CPU-only instances are cheap by GPU
  standards but $1.50/hr × 24 hr = $36 wasted overnight. Set a phone
  alarm or use AWS auto-shutdown.

## Scenario B: CUDA GPU box for AIMET SEQ_MSE / AdaScale

This is the original use case for this doc — the Qwen3-4B Phase 5p
quality work. Use this scenario when you already have a working basic-
PTQ export and want to close the cos-divergence gap to Qualcomm's
shipping bundle.

Expected spend: **$2-4 one-time**.
Expected wall time: **3-6 hours** (renting + install + quantize + transfer).

### Why we need this (the short version)

At Phase 5o our bundle:
- Matches Qualcomm's shipping bundle **structurally** (uint8 KV with
  exact per-layer scales, half-dim cos/sin, correct IO naming).
- Generates the **first 8 decoded tokens verbatim** vs Qualcomm's
  recorded oracle (`\n Okay , the user is asking`).
- Matches **30/46 (65%)** of decoded argmax tokens.
- Is **~50% oversized** (4.8 GB vs Qualcomm's 3.1 GB) because we use
  w8 for the transformer layers where Qualcomm uses w4. At w4 our
  calibration quality drops too far to be useful.

Phases 5j, 5q, and 5r established that:
- Qualcomm's shipping w4 bundle reaches `cos 0.9998` on Part 2 L11 —
  the HTP backend itself is fully capable of w4 fp32-equivalent
  magnitude activations.
- Our local qairt-quantizer w4 with `--use_per_channel_quantization
  --use_per_row_quantization --apply_algorithms cle` achieves `cos
  0.9996` — essentially at the **basic PTQ quality ceiling**.
- AI Hub's `submit_compile_job` w4a16: `cos 0.9976`. Also basic PTQ.
  Worse than ours.
- AI Hub's `submit_quantize_job` w4a16: `cos 0.9926`. Also basic PTQ.
  Also worse than ours.

The gap to Qualcomm's shipping quality comes from **Sequential MSE
(SEQ_MSE)** and **AdaScale** — two iterative weight-search techniques
Qualcomm applies *on top of* basic PTQ. These techniques are published
only in the `qai_hub_models/models/qwen3_4b/quantize.py` script that
runs on Linux x86_64 + CUDA and requires **40 GB of VRAM** for a 4B
model. The actual source:

```python
# qai_hub_models/models/_shared/llm/quantize.py
if device.type != "cuda":
    if not allow_cpu_to_quantize:
        raise ValueError("...requires CUDA GPU (V100/A100)...")
    if use_seq_mse or use_ada_scale:
        raise ValueError("This technique requires V100/A100.")
```

None of this runs on AI Hub's cloud compute, and none of this runs on
our X2E Windows-on-ARM box. So we need to rent a Linux + CUDA box for
a few hours.

### Why not WSL / WSL2 on the X2E

- WSL2 on Windows-on-ARM only supports **ARM64 Linux** distros.
- Qualcomm's `aimet_onnx` wheel is published for
  `cu121-cp310-cp310-manylinux_2_34_x86_64.whl` — **x86_64 only**.
  There is no ARM build.
- Running the x86_64 wheel under QEMU user-mode emulation would work
  numerically but be unusably slow for a 4B model (tens of hours).
- No practical way to run AIMET on this machine.

### Why not a consumer GPU like RTX 2080/2080 Ti

- RTX 2080: 8 GB VRAM. A 4B model in fp16 alone is 8 GB — activations
  and scratch don't fit.
- RTX 2080 Ti: 11 GB. Might fit *basic* AIMET PTQ, but basic PTQ is
  same tier as AI Hub's cloud paths which we already measured at
  worse quality than our local qairt-quantizer. No value add.
- SEQ_MSE specifically needs 24-40+ GB per Qualcomm's own docs and
  the code's `V100/A100` error message. A 2080 will OOM.
- If you have a 24 GB+ card (3090 / 4090 / 5090 / any A-series), yes
  — skip the cloud and run locally. The instructions below still
  apply, just s/cloud/`your-local-linux-box`/.

### Picking a provider (CUDA)

Any of these work. Ranked by typical cost (cheapest first):

| provider | GPU | VRAM | $/hr (spot/on-demand) | notes |
|---|---|---:|---:|---|
| **RunPod** (Community Cloud) | A100 40GB | 40 GB | ~$1.00-1.50 | cheapest, good for one-shot jobs |
| **Lambda Labs** | A100 40GB | 40 GB | ~$1.29 | on-demand; reliable |
| **RunPod** (Community Cloud) | A100 80GB | 80 GB | ~$1.90 | overkill but safest |
| **Vast.ai** | A100 40GB | 40 GB | ~$0.80-1.50 | marketplace, variable |
| **AWS** p4d.24xlarge | A100 40GB × 8 | 320 GB | $32.77 on-demand ($10 spot) | too expensive for one-shot |
| **GCP** a2-highgpu-1g | A100 40GB | 40 GB | $3.67 on-demand | easy if you already have GCP |

Recommendation: **RunPod** — cheapest, pay-per-minute billing, no
long commitment, pre-built PyTorch containers that already have CUDA
drivers + common deps.

**Not sufficient** (documented to fail or OOM for 4B SEQ_MSE):
- A10G / L4 (24 GB) — may fit basic PTQ, not SEQ_MSE reliably
- RTX 4090 (24 GB) — same
- V100 16 GB — below target
- T4 (16 GB) — below target

### Step-by-step: RunPod path

### 1. Sign up + add $5 credit

https://www.runpod.io → sign up, add $5-10 credit. Should last for
all of Phase 5p + future repeats.

### 2. Start a pod

- Choose template: **"PyTorch 2.4.1"** (or later 2.x) — the image
  ships with CUDA 12.1 drivers and matches the `aimet_onnx+cu121`
  wheel's target.
- Choose GPU: **1× A100 40GB** (or 80GB if available cheaply).
- Storage: **100 GB** persistent volume (the fp32 Qwen3-4B model is
  ~17 GB; with AIMET intermediates budget 50+ GB).
- Region: doesn't matter much; pick one close to you for ssh latency.
- "Deploy" → note the ssh command RunPod gives you. Save it.

### 3. ssh in, install deps

```bash
ssh root@<runpod-host> -p <port>  # from the pod's connect tab

# Qualcomm's published install line for qwen3_4b with AIMET:
pip install "qai-hub-models[qwen3-4b]" onnxruntime-gpu==1.23.2 \
  https://github.com/quic/aimet/releases/download/2.26.0/aimet_onnx-2.26.0+cu121-cp310-cp310-manylinux_2_34_x86_64.whl \
  -f https://download.pytorch.org/whl/torch_stable.html

# Sanity checks
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True, 'NVIDIA A100-SXM4-40GB' (or similar)

python -c "import aimet_onnx; print(aimet_onnx.__version__)"
# Expected: 2.26.0

nvidia-smi  # confirm 40 GB total VRAM
```

### 4. Configure AI Hub credentials

AIMET's `llm_quantize` uses the Hugging Face `Qwen/Qwen3-4B` checkpoint
directly — no AI Hub call is required for the quantization step itself.
But `qai-hub-models` imports assume the token is configured, and you
can reuse the same one from the X2E:

```bash
qai-hub configure --api_token YOUR_TOKEN
```

(Find your token at
https://workbench.aihub.qualcomm.com/ → Account → Settings → API Token.
Same token we already use on the X2E.)

### 5. Run the quantize job

```bash
mkdir -p /workspace/out

# Basic PTQ first, to sanity check the pipeline (~15 min):
python -m qai_hub_models.models.qwen3_4b.quantize \
  --output-dir /workspace/out/qwen3_4b_basic_ptq \
  --precision w4a16 \
  --num-samples 128

# Then the real run with SEQ_MSE + AdaScale (~2-5 hrs):
python -m qai_hub_models.models.qwen3_4b.quantize \
  --output-dir /workspace/out/qwen3_4b_seqmse \
  --precision w4a16 \
  --use-seq-mse \
  --use-ada-scale \
  --num-samples 128 \
  --seq-mse-num-samples 128 \
  --ada-scale-num-samples 128
```

`--num-samples 128` is a reasonable default per Qualcomm's tutorial;
you can try 512 if you want to match their typical setup. Larger
sample counts = better encodings but linearly more time.

`nvidia-smi` in a second shell should show ~30-38 GB VRAM usage during
SEQ_MSE on a 40 GB card. If it OOMs (likely on 24 GB cards), drop to
basic PTQ or bump to 80 GB.

### 6. Inspect output

```bash
ls -la /workspace/out/qwen3_4b_seqmse/
# Expected files:
#   model.onnx              — quantized model with QDQ nodes (AIMET encodings inlined)
#   model.data              — external weight data (quantized, ~2-3 GB)
#   encodings.json          — per-tensor AIMET encodings (scale / offset / bitwidth)
#   aimet_config.json       — the quant config that was applied
#   config.json             — HF config copy (for downstream tools)
```

The key output is `encodings.json` (AIMET format). This is what we
import into qairt-converter via `--quantization_overrides` to get the
same quality on our pathb split parts.

### 7. Transfer to the X2E

From the pod:

```bash
# Tar up just what we need (encodings + quantized onnx for reference).
# Skip model.data if the X2E already has the fp32 weights (~4 GB saving).
cd /workspace/out
tar czf qwen3_4b_seqmse_encodings.tgz qwen3_4b_seqmse/encodings.json \
                                      qwen3_4b_seqmse/aimet_config.json \
                                      qwen3_4b_seqmse/config.json

# Upload somewhere you can pull from (or scp back to your home machine).
# Easiest: install rclone + point at your own google drive / dropbox / S3.
# Or: use RunPod's built-in file manager in the web UI.
```

On the X2E, pull the tgz into:

```
results/phase5_qwen3_4b_bundle/aimet_seqmse/
  encodings.json
  aimet_config.json
  config.json
```

### 8. SHUT DOWN THE POD

RunPod keeps billing by the minute. Once you've got the files locally,
immediately stop (or terminate) the pod. A forgotten A100 for 24 hours
is ~$30 you didn't mean to spend.

### Applying the encodings on the X2E

The `encodings.json` is in AIMET format, which QAIRT's
`qairt-converter` accepts directly via `--quantization_overrides`:

```bash
.venv-qairt/Scripts/python.exe -c "
import os, sys, subprocess
os.environ['PYTHONPATH'] = r'C:\Qualcomm\AIStack\QAIRT\2.45.40.260406\lib\python' + os.pathsep + os.environ.get('PYTHONPATH', '')
tool = r'C:\Qualcomm\AIStack\QAIRT\2.45.40.260406\bin\x86_64-windows-msvc\qairt-converter'
for part in (2, 3, 4):
    subprocess.run([sys.executable, tool,
        '--input_network', f'models/qwen3-4b-arm-pathb-ctx512-part{part}/model_halfdim.onnx',
        '--output_path', f'results/phase5_qwen3_4b_bundle/qwen3_4b_part{part}.fp32.dlc',
        '--preserve_onnx_output_order', '--remove_unused_inputs',
        '--quantization_overrides', 'results/phase5_qwen3_4b_bundle/aimet_seqmse/encodings.json'])
"
```

Then continue the existing pipeline — `qairt_quantize_4b_parts.py`
(keeping `--weights-bitwidth 4` this time; AIMET's wider internal
encodings should let us stay at w4 without magnitude compression),
`compile_4b_bundle_ctx_bin_gen.py`, rebuild wrappers, probe, oracle.

Expected result (if SEQ_MSE works as advertised):
- Part 2 bin size drops from our current 1221 MB (w8+CLE) to ~615 MB
  (w4+AIMET) — matching Qualcomm's 669 MB.
- Bundle total drops from 4.8 GB to ~3.1 GB (matching Qualcomm's).
- First-decode cos climbs from +0.611 into the +0.9 range.
- Decoded argmax agreement climbs from 30/46 toward 46/46.

If SEQ_MSE doesn't close the full gap, we'll still have a measurable
improvement over our basic-PTQ qairt-quantizer w4 build, and we'll
know the remainder is either calibration-set differences (Qualcomm's
calibration set is proprietary) or compile-side differences
(different SoC model version, different QAIRT version, etc.).

### Gotchas (Scenario B)

- **Token input shape**: the `quantize.py` script uses its OWN pathb
  ONNX with Qualcomm's internal structure, NOT our `model_halfdim.onnx`
  from the X2E. The encodings.json it outputs has AIMET tensor names
  that may not match our graph's tensor names 1:1. Sanity check the
  names before passing the JSON to qairt-converter; if they differ,
  script a mapping (most names should match since both flows start
  from the same HF Qwen3-4B checkpoint with hoisted rotary).
- **Precision string**: `--precision w4a16` assumes that's a supported
  enum. Run `python -m qai_hub_models.models.qwen3_4b.quantize --help`
  first to see the exact choice list.
- **Disk I/O**: AIMET writes a LOT of intermediate tensors during
  SEQ_MSE. 100 GB disk is a minimum; if the pod doesn't offer enough,
  quantization will silently fail partway through with ENOSPC.
- **Rate limit on HF checkpoint download**: first run fetches the
  ~17 GB Qwen3-4B checkpoint from Hugging Face. If you're behind a
  slow link or HF is rate-limiting, pre-download it via `huggingface-cli
  login && huggingface-cli download Qwen/Qwen3-4B` before running the
  quantize script.
- **Do not leave the pod running overnight**. An A100 at $1.50/hr
  costs $36 in 24 hrs. Set a phone alarm.

### Budget alternative: basic PTQ on cheaper GPU ($0.50 / ~30 min)

Basic AIMET PTQ without SEQ_MSE can run on a 24 GB card. Swap:
- GPU: A10G 24GB (RunPod ~$0.50/hr)
- Command: drop `--use-seq-mse --use-ada-scale`

This gives the AIMET basic-PTQ encoding. Based on our Phase 5q/5r
measurements this is **probably not better** than our local
qairt-quantizer w4+CLE+per-channel. Only worth doing if you want to
measure the delta rigorously.

### CPU-only sanity check ($0 / local Linux box with ≥32 GB RAM)

Per the source, `allow_cpu_to_quantize=True` is set for Qwen3-4B, so
basic PTQ runs on CPU. Locally on a machine with ≥32 GB RAM:

```bash
# Same install line as above, but on CPU-only torch:
pip install "qai-hub-models[qwen3-4b]" onnxruntime==1.23.2 \
  https://github.com/quic/aimet/releases/download/2.26.0/aimet_onnx-2.26.0+cpu-cp310-cp310-manylinux_2_34_x86_64.whl

# Run WITHOUT --use-seq-mse --use-ada-scale (those require GPU):
python -m qai_hub_models.models.qwen3_4b.quantize \
  --output-dir ./out --precision w4a16 --num-samples 128
```

~10 hours on a fast desktop CPU. Basic PTQ only → same quality tier as
AI Hub cloud APIs → probably not better than our qairt-quantizer.
Useful only as a reference point, not as a pipeline-fix.
