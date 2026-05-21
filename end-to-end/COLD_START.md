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

## If you are an LLM agent following this solo

You are most likely Claude Code (or similar) running either ON the
pod or via SSH from the operator's local box. Before you start,
**verify the human operator has provided** these three secrets as
environment variables:

```bash
echo "HF_TOKEN     length: ${#HF_TOKEN}"        # expect non-zero
echo "AIHUB_TOKEN  length: ${#AIHUB_TOKEN}"     # expect non-zero
echo "GH_TOKEN     length: ${#GH_TOKEN}"        # optional, only for git push
```

If any are empty, **stop and ask the operator** before continuing —
the bootstrap depends on them at steps 5 and 7. Do not attempt
interactive `huggingface-cli login` or `gh auth login` flows; they
will hang the session.

Where state goes:

- **Repo state** (this doc, `current_status.md`, `docs/`) lives in
  `/workspace/specula/` once cloned. Edits should be committed and
  pushed; this is the durable record.
- **Run state** (logs, intermediate ONNX, AIMET workdirs) lives in
  `/workspace/runs/<run-name>/`. Not committed; tarred at teardown.
- **Project state of record** is `current_status.md` at repo root.
  Read it before starting (it's huge, read in chunks if needed) so
  you understand what session you are continuing. Append a session
  block at the end when you finish, before tearing down the pod.

Polling cadence: AIMET runs are 2-8 hours. Don't poll the log every
30 seconds — once every 5-10 minutes is plenty. Use the `grep -c
"Optimizing block"` pattern in the watch commands below to track
progress without flooding context.

If you hit a failure not covered by this doc, check
`end-to-end/README.md` §Troubleshooting and
`docs/one_pipeline_cloud_gpu.md` §Pitfalls (P1-P12) before improvising.

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
| Hugging Face read token (`HF_TOKEN`) | huggingface.co/settings/tokens | Qwen3 weight download (rate-limited anonymous) |
| AI Hub token (`AIHUB_TOKEN`) | workbench.aihub.qualcomm.com → Account → Settings | `qai-hub-models` package imports require it configured even though we don't call AI Hub itself |
| GitHub token (`GH_TOKEN`, optional) | github.com/settings/tokens | For `git push` of `current_status.md` updates from the pod |

**Pod env-var injection:** at RunPod deploy time, set the three
tokens above as **pod environment variables** (RunPod's deploy UI
has a "Environment Variables" section). They will be available to
both root and `dev` shells, and are the contract this doc assumes
in step 5 (AI Hub) and step 7 (HF). Do not paste tokens into
shell history; do not commit them to the repo.

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

After it runs, drop into the dev shell — **preserve the pod env
vars** so `$HF_TOKEN`, `$AIHUB_TOKEN`, `$GH_TOKEN` carry over from
the root context:

```bash
sudo -u dev --preserve-env=HF_TOKEN,AIHUB_TOKEN,GH_TOKEN -i
```

Plain `su - dev` would strip those (login shell resets env). Verify
inside the dev shell:

```bash
echo "HF=${#HF_TOKEN} AIHUB=${#AIHUB_TOKEN} GH=${#GH_TOKEN}"   # all non-zero
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

# Activate the venv
source /workspace/venvs/aimet-2.26-cu121-py310/bin/activate

# Core line. NOTE (verified 2026-05-21, qai-hub-models 0.54.0):
#  - there is NO `qwen3-0_6b` extra (only `qwen3-4b`); 0.6B uses the
#    same toolchain, it just isn't a qai-hub-models target.
#  - qai-hub-models does NOT pull `optimum`; install it + `optimum-onnx`
#    (optimum 2.x moved the `export onnx` subcommand into optimum-onnx)
#    and `accelerate` (transformers 4.51 references init_empty_weights
#    unconditionally — without accelerate the export dies NameError).
uv pip install "qai-hub-models[qwen3-4b]" \
               onnxruntime-gpu==1.23.2 \
               https://github.com/quic/aimet/releases/download/2.26.0/aimet_onnx-2.26.0+cu121-cp310-cp310-manylinux_2_34_x86_64.whl \
               optimum optimum-onnx accelerate \
               -f https://download.pytorch.org/whl/torch_stable.html
```

This is the slow step (~10-15 min, ~6 GB of wheels). It's the
one-time cost; future pods just `source` the venv.

**Blackwell pods only (RTX Pro 6000, sm_120):** the torch pulled in
above is a cu121 build with no sm_120 kernels — a real GPU matmul
fails `no kernel image is available`. Swap the whole torch family to
a cu128 build. `aimet_onnx 2.26`'s CUDA custom op JIT-compiles fine
onto sm_120 from its `compute_90` PTX, and `onnxruntime-gpu 1.23.2`'s
CUDA EP also runs on Blackwell — so **only torch** needs changing.
(On A100/H100, sm_80-90, skip this — the cu121 stack runs native.)

```bash
uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
               --index-url https://download.pytorch.org/whl/cu128
```

Sanity:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_arch_list()[-1])"
# Blackwell: True NVIDIA RTX PRO 6000 Blackwell ... compute_120
# A100:      True NVIDIA A100-SXM4-80GB sm_90

python -c "import aimet_onnx, onnxruntime; print(aimet_onnx.__version__, onnxruntime.__version__)"
# Expected: 2.26.0+cu121 1.23.2
```

### 4b. Second venv for `qairt-converter` (numpy 1.x)

QAIRT 2.45's compiled bindings (`ir_graph`/`libDlModelToolsPy`) are
built against the **numpy 1.x C ABI**. Run `qairt-converter` (stage 7,
a Python script) under the numpy-2.x AIMET venv and every static-tensor
handoff (Reduce axes, Reshape shapes) reads garbage in C++ — the
"Reduce param axis ... get:<garbage>" abort. So stage 7 needs its own
numpy-1.x venv (`lib/qairt.py` looks for it at this exact path):

```bash
uv venv /workspace/venvs/qairt-py310 --python 3.10
source /workspace/venvs/qairt-py310/bin/activate
uv pip install "numpy==1.26.4" onnx protobuf pyyaml packaging
deactivate
```

The AIMET stages (1-6) keep using the numpy-2.x `aimet-2.26-cu121-py310`
venv; only `qairt-converter` uses `qairt-py310`. `qnn-context-binary-
generator` (stage 8) is a compiled binary — numpy-agnostic.

### 5. Configure the AI Hub token

Uses `$AIHUB_TOKEN` from the preamble:

```bash
test -n "$AIHUB_TOKEN" || { echo "AIHUB_TOKEN unset; ask operator"; exit 1; }
qai-hub configure --api_token "$AIHUB_TOKEN"
qai-hub configure --api_token_query        # confirms it took
```

Required only if you `import qai_hub_models` directly. **The
`quantize_to_npu.py` pipeline does NOT** — it drives optimum +
aimet_onnx + qairt-converter, none of which touch AI Hub. Verified
2026-05-21: a full run reaches AIMET stage 6 with no AI Hub config.
**This step is skippable** for the end-to-end pipeline. AIMET pulls
weights from HF (or a local `--model-path`), not AI Hub.

### 6. Download QAIRT 2.45.40.260406 SDK to `/workspace/sdks/`

```bash
mkdir -p /workspace/sdks && cd /workspace/sdks

curl -sL -A "Mozilla/5.0" \
  https://softwarecenter.qualcomm.com/api/download/software/sdks/Qualcomm_AI_Runtime_Community/All/2.45.40.260406/v2.45.40.260406.zip \
  -o qairt-2.45.40.260406.zip

unzip -q qairt-2.45.40.260406.zip
rm qairt-2.45.40.260406.zip

# NOTE (2026-05-21): the zip extracts to `qairt/<version>/`, not
# `qairt-<version>/`. The scripts + ~dev/.bashrc hard-code the
# `qairt-<version>` form — relocate it:
mv /workspace/sdks/qairt/2.45.40.260406 /workspace/sdks/qairt-2.45.40.260406
rmdir /workspace/sdks/qairt

# qairt-converter's native bindings (libDlModelToolsPy.so) link
# libc++, which is absent on the Ubuntu base image. Install it now
# or `qairt-converter` fails with `ImportError: libc++.so.1`.
sudo apt-get update && sudo apt-get install -y libc++1 libc++abi1

# Verify the binaries are where dev's .bashrc expects them
ls /workspace/sdks/qairt-2.45.40.260406/bin/x86_64-linux-clang/qairt-converter
ls /workspace/sdks/qairt-2.45.40.260406/bin/x86_64-linux-clang/qnn-context-binary-generator
```

The `LD_LIBRARY_PATH` and `PATH` exports in `~dev/.bashrc` (set in
step 2) point at this exact path — do not relocate.

Re-source so PATH picks up the SDK:

```bash
source ~/.bashrc
qairt-converter --help | head -1   # smoke test — should print usage,
                                   # not an ImportError traceback
```

### 7. Pre-fetch the HF model weights

Non-interactive login using the env var from the preamble:

```bash
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
```

```bash
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

Verify both landed:

```bash
ls -la /workspace/models/Qwen3-0.6B/model.safetensors    # ~1.5 GB
ls -la /workspace/models/Qwen3-4B/model-*-of-*.safetensors  # ~8 GB total
```

### 8. Pre-fetch Qualcomm's blessed Qwen3-4B bundle (for comparison)

The X2 Elite-keyed bundle is one of several published in
`huggingface.co/qualcomm/Qwen3-4B`; `release_assets.json` lists S3
URLs per chipset. Resolve and download in one step:

```bash
mkdir -p /workspace/reference && cd /workspace/reference

# Resolve the X2 Elite asset URL.
URL=$(curl -fsSL https://huggingface.co/qualcomm/Qwen3-4B/raw/main/release_assets.json \
      | python -c "import json,sys; d=json.load(sys.stdin); \
          print(next(a['url'] for a in d['chipset_assets'] if 'x2-elite' in a.get('chipset','').lower()))")
echo "Resolved bundle URL: $URL"
test -n "$URL" || { echo "could not resolve X2 Elite bundle URL"; exit 1; }

# Download (~3.1 GB) and unpack.
curl -fsSL "$URL" -o qwen3_4b_qualcomm.zip
unzip -q qwen3_4b_qualcomm.zip -d qwen3_4b_qualcomm/
rm qwen3_4b_qualcomm.zip

# Verify
ls /workspace/reference/qwen3_4b_qualcomm/*.bin | wc -l    # expect 4 parts
ls /workspace/reference/qwen3_4b_qualcomm/genie_config.json
```

Lands at `/workspace/reference/qwen3_4b_qualcomm/` and stays
there for `compare_to_qualcomm.py` (step 12). If the resolver
fails (Qualcomm reorganised `release_assets.json`), browse to
`https://huggingface.co/qualcomm/Qwen3-4B/tree/main` in a browser
and grab the X2 Elite zip URL manually.

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
vs FP32 reference. Find it with:

```bash
grep -E "cos.*FP32|cosine" /workspace/runs/qwen3_0p6b_w8a16.log | tail -5
```

- **cos ≥ 0.99 →** pipeline + venv + SDK are known-good. Proceed
  to step 10.
- **0.95 ≤ cos < 0.99 →** marginal. Inspect the per-layer
  breakdown in the log; usually a calibration-set issue. Document
  in `current_status.md` and proceed cautiously.
- **cos < 0.95 →** pipeline regression. Stop. Compare the venv +
  SDK versions against `WARM_START.md` "What's persistent on
  /workspace" expectations; check `end-to-end/README.md`
  Troubleshooting. Do not proceed to 4B.

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

**Gate:** this run is itself the experiment, not a pass/fail
check. Both outcomes are publishable.

- **cos ≥ 0.95 →** Lever C closed positive. SEQ_MSE + AdaScale +
  V/O-pin together recover 0.6B w4a16 to ship-quality. Document
  in `current_status.md`, append finding to
  `docs/w4a16_investigation_continued.md`, proceed to step 11.
- **cos in 0.3-0.6 range →** Lever C closed negative. V/O
  collapse is a fundamental 0.6B-class property even with the
  best calibration we have. This is itself a valuable result;
  document with full per-layer evidence, append to
  `docs/w4a16_investigation_continued.md`, then proceed to step 11
  anyway — 4B is the real production target and may not exhibit
  the same collapse.
- **anything else (NaN, hang, OOM) →** investigate before
  proceeding to 4B. Check `end-to-end/README.md`
  Troubleshooting; the AdaScale ReduceMean v18 issue is a known
  failure mode there.

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

**Gate interpretation:**

- **All sections ✓ →** structural reproduction achieved. Numerical
  parity (cos vs. Qualcomm's first-decode logits) still requires
  X2E hardware to verify; that step lives in
  `npu_engine/qualcomm_qwen3_4b_oracle.py` on the laptop. Bundle
  is shippable for X2E testing.
- **Bin-size drift > ~10% →** different quant scheme was applied
  (e.g. our V/O-pin produced larger bins than Qualcomm's
  uniform-w4). Note in `current_status.md`; not a failure but a
  known difference.
- **Tokenizer sha mismatch →** stop. Means we exported from a
  different HF revision than Qualcomm. Re-pin and re-run.
- **`bin_info.json` IO shapes/dtypes mismatch →** stop. Means our
  pathb rewrites diverged from Qualcomm's reference graph. Debug
  via `scripts/rewrite_qwen3_pathb.py` against the differing
  shape; this is the highest-signal failure mode.
- **AIMET-encoding entry-count mismatch →** AIMET inserted/removed
  QDQ nodes vs. Qualcomm's reference. Probably tolerable; note
  the diff and continue.

### 13. Push past Qualcomm — cl=8192 / 16384 / 32768

**Run this only after step 12 passes** (i.e., we've verified our
pipeline can structurally reproduce Qualcomm's blessed 4-part
bundle at cl ∈ {512, 1024, 2048, 3072, 4096}). Qualcomm shipped
their bundle capped at cl=4096; **we are not capped at what they
shipped**. On 2026-05-13 we measured the existing cl=4096 graph at
pp=3584 / tg=256 and confirmed TG holds flat at ~20 t/s across all
buffer fill levels (`results/csv/qwen3_4b_ortqnn_cl4096_pp3584_tg256_2026-05-13.csv`),
so the only thing stopping us from real long-context NPU inference
is the absence of compiled larger-cl graphs.

The Qwen3-4B model itself supports it: `config.json` has
`max_position_embeddings: 40960` and `rope_theta: 1e6`. No YaRN /
rope-scaling required for cl ≤ 32k — `theta=1e6` covers it natively.

Build three additional bundles after the cl=4096 reproduction
lands:

```bash
# cl=8192 — safe, fits comfortably in HTP context budget
nohup setsid python end-to-end/quantize_to_npu.py \
    --model-id Qwen/Qwen3-4B \
    --model-path /workspace/models/Qwen3-4B \
    --workdir /workspace/runs/qwen3_4b_w4a16_cl8k \
    --precision w4a16 \
    --ctx 8192 \
    > /workspace/runs/qwen3_4b_w4a16_cl8k.log 2>&1 < /dev/null &
disown

# cl=16384 — KV ~9 GB, near HTP ceiling; verify it loads on X2E
# before assuming it's usable
nohup setsid python end-to-end/quantize_to_npu.py \
    --model-id Qwen/Qwen3-4B \
    --model-path /workspace/models/Qwen3-4B \
    --workdir /workspace/runs/qwen3_4b_w4a16_cl16k \
    --precision w4a16 \
    --ctx 16384 \
    > /workspace/runs/qwen3_4b_w4a16_cl16k.log 2>&1 < /dev/null &
disown

# cl=32768 — research push. KV ~18 GB; likely needs weight-sharing
# or won't load whole. Run anyway and report what the HTP says.
nohup setsid python end-to-end/quantize_to_npu.py \
    --model-id Qwen/Qwen3-4B \
    --model-path /workspace/models/Qwen3-4B \
    --workdir /workspace/runs/qwen3_4b_w4a16_cl32k \
    --precision w4a16 \
    --ctx 32768 \
    > /workspace/runs/qwen3_4b_w4a16_cl32k.log 2>&1 < /dev/null &
disown
```

Expected per-bundle build time: ~6-8 hr each (same shape as step
11). Each bundle is independent and can run on a fresh pod;
serialize them on one pod or fan out across pods depending on cost
budget.

**Expected HTP-side outcomes** (verify on X2E in `npu_engine/`,
not on cloud):

| tier   | KV memory | TG projection | Risk |
|--------|-----------|---------------|------|
| 8192   | ~4.6 GB   | ~15 t/s       | low — fits HTP budget |
| 16384  | ~9.2 GB   | ~10-12 t/s    | medium — at/over single-session HTP ceiling; may need session/weight-sharing |
| 32768  | ~18 GB    | unknown       | high — likely exceeds X2E HTP context budget; may not load |

Bench commands on the X2E (after copying the bundle to
`models/specula-qwen3-4b-cl{N}k/`):

```powershell
.venv/Scripts/python.exe npu_engine/bench_qwen3_4b_ortqnn.py `
    --power-state ac --ctx-tier 8192 `
    --prompt results/qwen3_4b_baseline/pp8k_prompt.txt `
    --pp-tokens 6144 --tg-tokens 512 `
    --tag cl8192_pp6144_<date>
```

Note: `bench_qwen3_4b_ortqnn.py` currently hardcodes `CTX_TIERS =
(512, 1024, 2048, 3072, 4096)` (the Qualcomm tier set). Extend
that tuple to include the new tiers before running, and confirm
the bundle's `metadata.yaml` exposes matching `ar1_cl{N}_*_of_4`
and `ar128_cl{N}_*_of_4` keys.

**Why this matters.** Every backend in our matrix can do >4k via
RoPE / sliding window / YaRN — except the NPU, which is capped by
the compiled cl tier. Closing that gap is what unlocks the NPU for
real long-context workloads (multi-doc RAG, agentic transcripts,
codebase Q&A). Plus we get to find out where the X2E HTP context
budget actually lives — Qualcomm only told us where their *bundle*
stops, not where their *silicon* stops. We are the researchers;
that's the data we don't have yet.

---

## Before teardown — write the session entry

Per `CLAUDE.md`, `current_status.md` is the project's session-level
source of truth and is appended each session. **Do not tear down
the pod until this is committed and pushed**, otherwise the next
session has no record of what this run produced.

Append a session block to `/workspace/specula/current_status.md`
covering:

- Date + which steps of this doc ran (e.g. "9, 10, 11, 12 all
  green").
- The three cos numbers (0.6B w8a16, 0.6B w4a16, 4B w4a16 vs FP32).
- The comparison-report verdict for 4B vs. Qualcomm.
- Any deviation from the doc (flag changes, gate failures,
  workarounds).
- Where the bundle tarballs were uploaded (NAS path, S3 URL, etc.)
  so the X2E side can fetch them.

Then commit + push:

```bash
cd /workspace/specula
git add current_status.md
git commit -m "current_status: cold-start session $(date +%Y-%m-%d) — <summary>"
git push origin master      # uses gh credential helper from SETUP_DEV_USER step 5
```

If `GH_TOKEN` is unset and the push fails with auth errors, ask
the operator before improvising — do not commit credentials to
the repo.

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
