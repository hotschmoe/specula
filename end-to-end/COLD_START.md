# Cold start — empty `/workspace` → producing & comparing bundles

**Status:** runbook, 2026-05-09. This is the cold-start companion to
`end-to-end/WARM_START.md`. Use this when network storage was nuked,
the pod is in a new datacenter, or you are bootstrapping
`/workspace` for the first time.

If `/workspace` already has the venvs, SDK, and models populated
from a prior session, **stop reading and use `WARM_START.md`** — it
takes about 30 seconds vs. 1-2 hours for the full bootstrap below.

Strategic context (why this pipeline shape exists at all) lives in
`docs/one_pipeline_cloud_gpu.md`. This doc is execution only.

---

## Goal

End the session with three artifacts on `/workspace` plus a
comparison report, all on persistent network storage that survives
the next pod teardown:

1. `qwen3_0p6b_pathb_w8a16.bin` — full-quality 0.6B baseline
2. `qwen3_0p6b_pathb_w4a16.bin` — Lever C closure attempt
   (does SEQ_MSE + AdaScale + V/O-w8 pin recover from the cos 0.33
   collapse basic PTQ left? `docs/w4a16_investigation.md` session 17)
3. `qwen3_4b_pathb_w4a16.bin` — 4B production target, multi-part
   bundle, compared file-by-file to Qualcomm's blessed shipping
   bundle via `end-to-end/compare_to_qualcomm.py`

---

## Prereqs (off-pod, do once)

| What | Where | Why |
|---|---|---|
| RunPod account + ≥$30 credit | runpod.io | Pay-per-minute billing |
| Hugging Face account + read token | huggingface.co/settings/tokens | Qwen3 weight download (no gate, but rate-limited anonymous) |
| AI Hub token | workbench.aihub.qualcomm.com → Account → Settings | `qai-hub-models` package imports require it configured even though we don't call AI Hub itself |
| `gh auth login` on a personal box | local | Used to mirror auth into the pod's `dev` user (per `SETUP_DEV_USER.md` step 4) |

---

## Pod specification

**Primary:** RunPod **RTX Pro 6000** (Blackwell, 96 GB VRAM) +
**80 GB+ system RAM**.

**Fallback:** A100 80GB if no Pro 6000 in the chosen DC. Avoid
A100 40GB — it pairs with smaller host RAM tiers and the SQ4 M2
OOM was system-RAM, not VRAM.

| Spec | Value | Why |
|---|---|---|
| GPU | RTX Pro 6000 96GB (or A100 80GB) | AIMET QSM build + AdaScale fits with headroom |
| Host RAM | **80 GB minimum**, 128 GB preferred | SQ4 M2 OOM'd at QSM build below 80 GB; AdaScale's `copy.deepcopy(sim.model.model)` + FP32 sampler stack high |
| Persistent storage | **150 GB network volume**, mounted at `/workspace` | Qwen3-4B FP weights ~17 GB, AIMET intermediates ~30 GB, venvs ~16 GB, SDK ~3 GB, run workdirs ~30 GB |
| Pod template | "PyTorch 2.4.1" (or later 2.x with CUDA 12.1 drivers) | Matches `aimet_onnx-2.26.0+cu121` wheel target |
| Region | Pick by availability of the GPU sku, **not** by latency | We've already been forced to redeploy once because the chosen DC didn't have the sku |

When deploying: tick **persistent volume**, not the ephemeral
container disk. The whole point of this doc is that step (4)–(7)
below survive the next teardown.

---

## Bootstrap sequence

### 1. Spin the pod, ssh in as root

```bash
ssh root@<runpod-host> -p <port>   # from RunPod's connect tab
```

Confirm the volume mounted:

```bash
ls /workspace          # should be empty (or have prior dirs if not nuked)
df -h /workspace       # should show 150 GB
nvidia-smi             # confirm RTX Pro 6000 96GB or A100 80GB
free -h                # confirm 80+ GB RAM
```

If `/workspace` already has `venvs/`, `sdks/`, `models/` —
**you're not on a cold start. Use `WARM_START.md` instead.**

### 2. Create the `dev` user

Follow `end-to-end/SETUP_DEV_USER.md` exactly — recreates `dev`
(uid 1000), passwordless sudo, mirrored ssh keys + gh auth, and
appends env vars to `~dev/.bashrc` that point at the venvs and
SDK paths we are about to create.

The recipe references `/workspace/venvs/aimet-2.26-cu121-py310`
and `/workspace/sdks/qairt-2.45.40.260406`. **Don't change those
paths** unless you also update `SETUP_DEV_USER.md` step 6 in the
same edit — too many other scripts hard-code them.

After it runs, drop into the dev shell:

```bash
su - dev      # interactive, env loaded
```

Everything below runs as `dev`.

### 3. Clone the repo onto `/workspace`

```bash
cd /workspace
git clone https://github.com/hotschmoe/specula.git
cd specula
git config --global --add safe.directory /workspace/specula
```

Putting the repo on `/workspace` (not `~/dev/`) is deliberate: it
survives pod teardown alongside the venvs and models. Re-cloning
on every cold start is fine but the network volume avoids it
entirely.

### 4. Install `uv` and create the AIMET venv

```bash
# uv installer (site-local, doesn't need root)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc       # picks up uv on PATH

# Create the venv on the persistent volume so it survives teardown.
uv venv /workspace/venvs/aimet-2.26-cu121-py310 --python 3.10

# Activate + install Qualcomm's exact published line
source /workspace/venvs/aimet-2.26-cu121-py310/bin/activate

uv pip install "qai-hub-models[qwen3-4b]" \
               "qai-hub-models[qwen3-0_6b]" \
               onnxruntime-gpu==1.23.2 \
               https://github.com/quic/aimet/releases/download/2.26.0/aimet_onnx-2.26.0+cu121-cp310-cp310-manylinux_2_34_x86_64.whl \
               -f https://download.pytorch.org/whl/torch_stable.html
```

This is the slow step (~10-15 min, ~6 GB of wheels). It's the
one-time cost; future pods just `source` the venv.

Sanity:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True NVIDIA RTX Pro 6000 (or NVIDIA A100-SXM4-80GB)

python -c "import aimet_onnx, onnxruntime; print(aimet_onnx.__version__, onnxruntime.__version__)"
# Expected: 2.26.0+cu121 1.23.2
```

### 5. Configure the AI Hub token

```bash
qai-hub configure --api_token $AIHUB_TOKEN
```

Required for `qai-hub-models` package imports. AIMET itself pulls
weights from HF, not AI Hub.

### 6. Download QAIRT 2.45.40.260406 SDK to `/workspace/sdks/`

```bash
mkdir -p /workspace/sdks && cd /workspace/sdks

curl -sL -A "Mozilla/5.0" \
  https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.45.40.260406/v2.45.40.260406.zip \
  -o qairt-2.45.40.260406.zip

unzip -q qairt-2.45.40.260406.zip
rm qairt-2.45.40.260406.zip

# Verify the binaries are where dev's .bashrc expects them
ls /workspace/sdks/qairt-2.45.40.260406/bin/x86_64-linux-clang/qairt-converter
ls /workspace/sdks/qairt-2.45.40.260406/bin/x86_64-linux-clang/qnn-context-binary-generator
```

The `LD_LIBRARY_PATH` and `PATH` exports in `~dev/.bashrc` (set in
step 2) point at this exact path — do not relocate.

Re-source so PATH picks up the SDK:

```bash
source ~/.bashrc
qairt-converter --version       # expect 2.45.40.260406
```

### 7. Pre-fetch the HF model weights

```bash
huggingface-cli login            # paste read-scope token

# 0.6B (~1.5 GB)
huggingface-cli download Qwen/Qwen3-0.6B \
    --local-dir /workspace/models/Qwen3-0.6B \
    --local-dir-use-symlinks False

# 4B (~8 GB)
huggingface-cli download Qwen/Qwen3-4B \
    --local-dir /workspace/models/Qwen3-4B \
    --local-dir-use-symlinks False
```

`--local-dir-use-symlinks False` writes real files to the
persistent volume; symlinks would point into the ephemeral HF
cache and break on next pod boot.

### 8. Pre-fetch Qualcomm's blessed Qwen3-4B bundle (for comparison)

```bash
mkdir -p /workspace/reference && cd /workspace/reference

# The X2 Elite-keyed bundle is one of several published in
# qualcomm/Qwen3-4B; release_assets.json lists S3 URLs per chipset.
# Look up the X2 Elite asset URL and download it; ~3.1 GB.
curl -fsSL https://huggingface.co/qualcomm/Qwen3-4B/raw/main/release_assets.json \
    | python -c "import json,sys; d=json.load(sys.stdin); [print(a['url']) for a in d['chipset_assets'] if 'x2-elite' in a.get('chipset','').lower()]"

# Download the resulting URL and unpack:
# curl -fsSL <URL> -o qwen3_4b_qualcomm.zip && unzip -q qwen3_4b_qualcomm.zip -d qwen3_4b_qualcomm/
```

Lands at `/workspace/reference/qwen3_4b_qualcomm/` and stays
there for `compare_to_qualcomm.py` (step 12).

---

## Validation runs

All three runs use `end-to-end/quantize_to_npu.py`. Each is
backgrounded with `nohup setsid ... < /dev/null &` so an SSH
disconnect doesn't kill the job (per `RESUME.md`).

### 9. Qwen3-0.6B w8a16 (cheap pipeline validation, ~2 hr)

```bash
cd /workspace/specula

nohup setsid python end-to-end/quantize_to_npu.py \
    --model-id Qwen/Qwen3-0.6B \
    --model-path /workspace/models/Qwen3-0.6B \
    --workdir /workspace/runs/qwen3_0p6b_w8a16 \
    --precision w8a16 \
    --ctx 512 \
    > /workspace/runs/qwen3_0p6b_w8a16.log 2>&1 < /dev/null &
disown
```

Watch progress:

```bash
tail -F /workspace/runs/qwen3_0p6b_w8a16.log
grep -c "Optimizing block" /workspace/runs/qwen3_0p6b_w8a16.log   # / 28 layers
free -h | head -2                                                  # RAM headroom
```

**Gate:** the AIMET probe in stage 6 should report **cos ≥ 0.99**
vs FP32 reference. If it does, the pipeline + venv + SDK are
known-good and you can proceed to the 4B run with confidence.

Output bundle:
`/workspace/runs/qwen3_0p6b_w8a16/09_bundle_w8a16/<bundle>.tar`.

### 10. Qwen3-0.6B w4a16 (Lever C closure attempt, ~2 hr)

```bash
nohup setsid python end-to-end/quantize_to_npu.py \
    --model-id Qwen/Qwen3-0.6B \
    --model-path /workspace/models/Qwen3-0.6B \
    --workdir /workspace/runs/qwen3_0p6b_w4a16 \
    --precision w4a16 \
    --ctx 512 \
    > /workspace/runs/qwen3_0p6b_w4a16.log 2>&1 < /dev/null &
disown
```

`--vo-pin-w8` is on by default for `--precision w4a16` (per
`end-to-end/README.md` quality knobs table) — the V/O w8 pin is
the SQ2/m1d-derived mitigation for the V-projection collapse
diagnosed in `docs/w4a16_investigation.md` session 17.

**Gate:** AIMET cos ≥ 0.95 closes Lever C positive. If the cos
lands in the 0.3-0.5 range like our basic-PTQ baseline, then
SEQ_MSE + AdaScale + V/O-pin together are insufficient on
0.6B-class models and the V/O collapse is a fundamental property
— captured with full evidence, then graduate to Qwen3-4B+ as the
real production target.

### 11. Qwen3-4B w4a16 (production target, ~6-8 hr)

**Run this only after step 9 (or 9+10) passes.** A100/Pro 6000
hours are the bulk of the rental cost — burning them on a busted
pipeline is the easy mistake.

```bash
nohup setsid python end-to-end/quantize_to_npu.py \
    --model-id Qwen/Qwen3-4B \
    --model-path /workspace/models/Qwen3-4B \
    --workdir /workspace/runs/qwen3_4b_w4a16 \
    --precision w4a16 \
    --ctx 512 \
    > /workspace/runs/qwen3_4b_w4a16.log 2>&1 < /dev/null &
disown
```

Watch RAM aggressively — 4B AdaScale on 36 layers is the OOM
risk (per `WARM_START.md` §"RAM gotcha"):

```bash
watch -n 30 'free -h | head -2; echo; nvidia-smi --query-gpu=memory.used --format=csv'
```

Output bundle:
`/workspace/runs/qwen3_4b_w4a16/09_bundle_w4a16/<bundle>.tar` —
expected to be a multi-part bundle matching Qualcomm's 4-part
shape.

### 12. Compare 4B output to Qualcomm's bundle

```bash
python end-to-end/compare_to_qualcomm.py \
    --ours /workspace/runs/qwen3_4b_w4a16/09_bundle_w4a16/<bundle>/ \
    --theirs /workspace/reference/qwen3_4b_qualcomm/ \
    > /workspace/runs/qwen3_4b_w4a16/comparison.md

cat /workspace/runs/qwen3_4b_w4a16/comparison.md
```

`compare_to_qualcomm.py` is a file-system + ONNX-level audit (no
HTP runtime needed — that part has to happen on the X2E). It
reports: per-part `.bin` sizes, `genie_config.json` field
agreement, `htp_backend_ext_config.json` agreement, tokenizer
sha256, per-part `bin_info.json` IO shapes/dtypes, per-part
AIMET-encoding entry counts and KV bitwidth coverage.

**Gate:** every section reports ✓ except possibly the bin sizes
(small drift acceptable, large drift = different quant scheme
applied).

---

## Teardown

```bash
# 1. tar the artifacts you actually want long-term + push somewhere
#    independent of /workspace (rclone to NAS, scp home, etc.)
tar czf /workspace/handoff_$(date +%Y%m%d).tgz \
    /workspace/runs/*/09_bundle_*/*.tar \
    /workspace/runs/*/comparison.md

# 2. Optional: prune intermediates to keep network volume lean
rm -rf /workspace/runs/*/{01,02,03,04,05}_*  # pre-AIMET stages, regeneratable
sudo rm -rf /tmp/tmp*                         # AdaScale tempdir leftovers

# 3. STOP THE POD from RunPod's web UI.
#    `sudo poweroff` stops the OS, not the billing.
```

A Pro 6000 at ~$2-3/hr × 24 hr = $48-72 wasted overnight if you
forget. Set a phone alarm. Non-negotiable.

---

## Pitfalls

The full catalog lives in `docs/one_pipeline_cloud_gpu.md` §Pitfalls
(P1-P12) and `end-to-end/README.md` §Troubleshooting. Cold-start-
specific gotchas only:

### C1. Datacenter doesn't have the GPU sku you picked

What we hit on the previous attempt. RunPod's deploy UI shows
availability by sku, but **only after** you've already chosen a
template. If the chosen DC + sku combination is empty, the only
fix is to redeploy in a different DC — and **the network volume
is DC-keyed**. You will lose the volume.

Mitigation: when deploying, expand "Availability per region" and
verify the sku is shown in your target DC before clicking deploy.
If you have to redeploy in a different DC, this whole doc is the
playbook for re-bootstrapping there.

### C2. Network volume claims to be 150 GB but pod sees less

RunPod's per-tier disk limits are independent of the volume size.
If the chosen pod template has a 50 GB container disk and you
write to `/root/...` instead of `/workspace/...`, you'll fill the
container disk while the network volume sits empty. Always write
to `/workspace`.

### C3. cgroup memory limit is lower than `free -h` reports

Per `RESUME.md`: RunPod containers cap memory at the tier limit
(commonly 44 GB) regardless of host RAM. Pick a tier with at
least 80 GB allocated to the container — not just to the host.
Verify with:

```bash
cat /sys/fs/cgroup/memory.max     # cgroupv2; expect ~80GB+ in bytes
# or
cat /sys/fs/cgroup/memory/memory.limit_in_bytes   # cgroupv1
```

### C4. uv install on the venv path persists; uv install in `~` doesn't

`uv venv /workspace/venvs/...` writes to network storage and
survives. `uv pip install` *into* that venv also writes there.
But `pip install --user` or anything writing to `~/.local` writes
to ephemeral home and is lost on teardown. Activate the venv
before `pip install`-ing anything you want to keep.

### C5. HF download to `~/.cache/huggingface` is ephemeral

The `--local-dir /workspace/models/...` flag in step 7 is what
keeps the weights persistent. If you skip it, HF downloads to
`~/.cache/huggingface/hub/` which is on the ephemeral container
disk and re-fetches every cold start. ~10 GB and ~10 min wasted
each time.

---

## Next session

If `/workspace` survived the teardown (the common case — only
DC-change kills it), the next session is **not a cold start**:
follow `end-to-end/WARM_START.md` instead. That's a 30-second path:
ssh in, `su - dev`, `git pull`, kick off the run.

If the volume is gone (DC-change forced redeploy, or RunPod
maintenance event), you are back here at step 1.

---

## Update log

- **2026-05-09** — Doc created after SQ4 M2 forced redeploy in a
  DC with no Pro 6000. Replaces ad-hoc cold-start steps that were
  scattered across `WARM_START.md` (formerly `RESUME.md`),
  `SETUP_DEV_USER.md`, `docs/one_pipeline_cloud_gpu.md` §Step 0-6,
  and `docs/rent_cloud_compute.md` Scenario B. Those docs stay as
  references; this is the single canonical bootstrap path.
