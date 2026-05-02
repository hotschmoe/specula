# Resume — fresh VM / new shell quickstart

If you just mounted a clean VM with `/workspace` attached, this is
the fastest path back to where the M-series work left off.

## What's persistent on `/workspace`

```
/workspace/specula/                     repo (git pull from origin to refresh)
/workspace/models/Qwen3-0.6B/           HF FP weights (1.5 GB)
/workspace/models/Qwen3-4B/             HF FP weights (8 GB) — staged for M2
/workspace/venvs/aimet-2.26-cu121-py310/  working AIMET venv (16 GB)
/workspace/venvs/aimet-2.29-cu126-py312/  parked (broken at sim-build, see findings)
/workspace/sdks/qairt-2.45.40.260406/   QAIRT SDK matching X2E NPU runtime
/workspace/runs/                        (per-run workdirs land here)
```

Anything else (`/root`, `/home/dev`, `/tmp`) is ephemeral.

## Pulling the latest repo

```bash
cd /workspace/specula
git pull origin master
```

## Switching from root → dev user (recommended)

`root` has been the historical entry point on this RunPod
container, but for cleaner file ownership and fewer privilege
quirks we use a `dev` user (uid 1000, passwordless sudo, env
pre-wired for the venv + QAIRT).

If `dev` doesn't exist on a fresh VM, recreate from the recipe in
`end-to-end/SETUP_DEV_USER.md` (~10 lines of `useradd` + bashrc
append + gh auth mirror).

To switch:

```bash
# from root
su - dev          # interactive shell as dev, env loaded from .bashrc
# or
sudo -u dev -i    # equivalent
```

You can verify the env is loaded:

```bash
echo $AIMET_VENV  # /workspace/venvs/aimet-2.26-cu121-py310
which python      # .../aimet-2.26-cu121-py310/bin/python
which qairt-converter qnn-context-binary-generator
python -c "import aimet_onnx, onnxruntime; print(aimet_onnx.__version__, onnxruntime.__version__)"
```

## Re-running the e2e (M2: Qwen3-4B w4a16)

```bash
cd /workspace/specula
nohup setsid python end-to-end/quantize_to_npu.py \
    --model-id Qwen/Qwen3-4B \
    --workdir /workspace/runs/qwen3_4b_w4a16 \
    --precision w4a16 \
    --ctx 512 \
    > /workspace/runs/qwen3_4b_w4a16.log 2>&1 < /dev/null &
disown
```

Defaults: `--num-cal-samples 128 --use-seq-mse --use-ada-scale
--ada-scale-iters 1500 --vo-pin-w8 (auto for w4a16)`.

`nohup setsid ... < /dev/null &` is **important** — survives shell
teardowns and parent reparents to init. Also keeps the process
alive across SSH disconnects.

## Resuming a partial run

If a run aborts (OOM, network hiccup, manual cancel), re-running
with the same `--workdir` skips completed stages via `done.json`
markers. To force re-run from a specific stage onward:

```bash
... --force-stage 6  # re-runs aimet + qairt + qnn + bundle
```

## Checking progress

```bash
# tail the live log
tail -F /workspace/runs/qwen3_4b_w4a16.log

# count AdaScale blocks done (28 layers for 0.6B, 36 for 4B)
grep -c "Optimizing block" /workspace/runs/qwen3_4b_w4a16.log

# see SEQ_MSE pace
grep "Computed optimal" /workspace/runs/qwen3_4b_w4a16.log | wc -l

# RAM watch (44 GB cap on this RunPod tier)
free -h | head -2
```

## RAM gotcha (44 GB cap, not host's 503 GB)

The container has a cgroup memory limit of ~44 GB independent of
what `free -h` reports. AdaScale's `copy.deepcopy(sim.model.model)`
+ FP32 sampler + cal samples can push close to the cap on 4B+.
**Don't run two AIMET processes concurrently** — sequential or
nothing.

## Disk hygiene

`/tmp` accumulates AdaScale's `tempfile.TemporaryDirectory`
remnants when a process exits abnormally. Each ~3 GB. Clean with:

```bash
sudo rm -rf /tmp/tmp*  # only the random-named dirs; leaves /tmp/* logs
```

## What's in this branch

Most recent state:
- `end-to-end/` — single ONE-script pipeline with `lib/model_config.py`
  for model-agnostic family handling, four AdaScale monkey-patches,
  pathb chain, qairt + qnn-context, bundle assembly.
- `last_side_quest/sq4_cloud_adventure/findings.md` — the M1 fight
  log + closeout decision (ship not, cos ceiling 0.6558 on aimet_onnx
  + pathb + Qwen3-0.6B) + the upstream-fix watch list for the
  monkey-patches.
- M1 artifacts have been cleaned (not ship-quality); only the
  pipeline + docs remain.

## What changes for new model families

`end-to-end/lib/model_config.py` has a `FAMILY_CONFIGS` dict.
Adding qwen3.5/qwen3.6/llama-3 etc. is one entry — but be aware:

1. The pathb scripts (`scripts/rewrite_qwen3_*.py`) are
   Qwen3-specific. New families with the same graph shape (Qwen2.5,
   future Qwen3.5) may work if `architectures[0]` matches; verify
   the IsNaN guard count and rotary subgraph layout first.
2. Models with `rope_scaling` (Llama3+, Qwen3.5+) need the rotary
   hoist updated — `rewrite_qwen3_pathb.py:88` asserts identity
   scaling. The fix is folding the scale factor into the externally-
   computed cos/sin in `lib/rope.py`.
3. AIMET's `AdaScaleModelConfig.model_type` only knows
   `{qwen2,qwen3,llama,mistral,phi3}` — for anything else, you'll
   either pick the closest match or extend AIMET upstream.
