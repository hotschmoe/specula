# Threadripper build server — remote NPU-bundle builds over SSH

Created 2026-06-15 (autonomous session). The Snapdragon X2E laptop is RAM-
and ISA-limited for the heavy NPU-bundle build steps (48 GB, ARM64 + Prism
emulation). The **unRAID Threadripper box drives those steps natively**, and
the X2E stays the orchestrator + the Hexagon deploy target.

```
Snapdragon X2E (orchestrator + NPU runtime)
   │  ssh (key-based) + rsync over ~100 MB/s LAN
   ▼
Threadripper 1950X / unRAID  (192.168.10.5)
   32 threads · 125 GB RAM · 8 TB SSD
   export → rewrite → split → qairt-convert → qairt-quantize (w8a16) → ctx-bin
   ▼  rsync the small .bin bundle back
Snapdragon X2E → ORT-QNN → Hexagon NPU
```

Why it dissolves every X2E wall: **125 GB RAM** (no OOM/thrash on the 59 GB
ONNX loads), **native x86_64 Linux** (QAIRT runs without Prism; `aimet_onnx`
installable), **8 TB SSD** (no disk-copy pressure), and only the small final
`.bin` crosses the slow LAN.

## Box facts (recon)

- `Linux 6.18.29-Unraid x86_64`, unRAID 7.3.0, 192.168.10.5
- 32 threads (1950X), **125 GB RAM**, has `git` / `docker` / `rsync`, **no
  system python** (so all python lives in Docker — zero host clobber risk)
- Storage (use the SSD, never the HDD array or USB boot):
  - `/boot` = 28 GB USB — **avoid**
  - `/mnt/disk1,2` = WD 10 TB **HDD** array (slow, ~38 MB/s writes)
  - `/mnt/cache` = SPCC NVMe (954 GB, ~522 GB free) — active cache, leave alone
  - **`/mnt/vm_8tb` = Samsung 870 QVO 8 TB SSD (7.3 TB free) → workspace here**
    (`175–200 MB/s` sustained writes observed)
- Network laptop↔box: **~100 MB/s** both ways over SSH (1 GbE-class hop, not
  the WiFi-7 1.5 Gbps — a 1 GbE link caps it; fine for the small artifact).
  WAN download on the box ~330 Mbps.

## One-time setup

1. **SSH key** (passwordless, from the X2E): generate `~/.ssh/id_ed25519`,
   install the pubkey in the box's `~/.ssh/authorized_keys`. (unRAID is
   RAM-based — for persistence across reboots, add it via Management Access
   UI / `/boot/config/ssh/`; the session-only append lives in `/root/.ssh`.)
2. **Workspace** on the SSD: `/mnt/vm_8tb/specula-build`. Code synced from the
   X2E via `tar | ssh ... tar x` (or `git pull` — the repo is on GitHub).
3. **Export/rewrite env** — `python:3.11` container + venv `/workspace/.venv-box`
   on the SSD: `torch(cpu) onnx onnxruntime transformers==4.57.6 optimum==2.1.0
   optimum-onnx==0.1.0`. (The `optimum-onnx` split package is required for
   `optimum-cli export onnx` in optimum 2.x — without it you get
   "unrecognized arguments: onnx".)
4. **QAIRT SDK** — extracted from `Z:/exposed/junk/Qualcomm.zip` (already on
   the box at `/mnt/user/StrongSync/...`), which contains the full
   **Linux x86_64 QAIRT 2.45.40.260406** (exact version match to the X2E's
   ORT-QNN, see [[reference_ort_qnn_qairt_match]]) → `/mnt/vm_8tb/specula-build/
   qairt/Qualcomm/AIStack/QAIRT/2.45.40.260406`. `chmod +x bin/x86_64-linux-clang/*`.
5. **QAIRT runtime image** — `specula-qairt:2.45` (`Dockerfile.qairt`):
   `python:3.10` + `libc++1 libc++abi1` (the clang-built QAIRT binaries need
   `libc++.so.1`) + numpy **1.26.4** (QAIRT's C ABI; not numpy 2.x) + onnx /
   onnxruntime / protobuf<5. QAIRT env inside the image:
   ```
   Q=/workspace/qairt/Qualcomm/AIStack/QAIRT/2.45.40.260406
   export PATH=$Q/bin/x86_64-linux-clang:$PATH
   export LD_LIBRARY_PATH=$Q/lib/x86_64-linux-clang
   export PYTHONPATH=$Q/lib/python
   ```
   All three tools verified working: `qairt-converter`, `qairt-quantizer`,
   `qnn-context-binary-generator`.

## Pipeline (Qwen3-14B w8a16, on-box)

`run_stages_1_5.sh` (resumable; skips any stage whose `model.onnx` exists),
run inside the `.venv-box` container:
1. optimum export (patched `optimum_export_4b.py` for the 2 GiB protobuf cap)
2. `rewrite_qwen3_htp.py --mode stage`
3. `rewrite_qwen3_htp.py --mode fold-pathbmask`
4. `rewrite_qwen3_pathb.py` (rotary hoist)
5. `pin_shapes_qwen3_4b.py --ctx 512 --num-kv-heads 8 --head-dim 128 --vocab-size 151936`

Then the no-AIMET QAIRT half (in `specula-qairt:2.45`), per part:
- split (`lib/split.py`) → `qairt-converter` (no `--quantization_overrides`)
  → `qairt-quantizer --weights_bitwidth 8 --act_bitwidth 16 --input_list <calib>`
  → `qnn-context-binary-generator` → `lib/bundle.py` assemble → rsync back.

## Findings (14B-scale, surfaced on the box)

- **`optimum-onnx 0.1.0` pin** required (optimum 2.x split the onnx exporter).
- **14B fp32 export peaks ~114 GB RAM** (model + JIT graph + proto). On 125 GB
  it just fits; added a **64 GB SSD swapfile** (`/mnt/vm_8tb/swapfile`,
  `swapon`) as a backstop — it stayed untapped (no thrash). *For the 27B,
  ~2× this won't fit even 125 GB → need fp16 or a true streaming export.*
- **protobuf 2 GiB cap in the rewrites** — `prune_unused_initializers` (htp)
  and the pathb prune did `del + extend`; `extend` deep-copies each
  initializer and protobuf can't serialize a single TensorProto > 2 GiB
  (14B's embed/lm_head are 3.1 GB each). Fixed both to **in-place deletion**
  (commit on master). 4B never hit this (1.55 GB embed).
- **QAIRT needs `libc++.so.1`** + **numpy 1.x** → the `specula-qairt` image.

## QAIRT phase — no-AIMET w8a16, proven on the box

The e2e pipeline normally feeds AIMET encodings into `qairt-converter`. We
skip AIMET and do **native `qairt-quantizer` PTQ** instead. Per part:

```
qairt-converter --input_network part/model.onnx --output_path part.dlc \
    --preserve_io_datatype                    # no --quantization_overrides
qairt-quantizer  --input_dlc part.dlc --output_dlc part_q.dlc \
    --weights_bitwidth 8 --act_bitwidth 16 --bias_bitwidth 8 \
    --use_per_channel_quantization            # ⚠️ SEE WARNING BELOW — NOT loadable as-is
qnn-context-binary-generator --backend libQnnHtp.so --dlc_path part_q.dlc \
    --binary_file part --output_dir 09_bin --config_file qnn_v81_box.json
```

> **⚠️ CORRECTION (2026-06-16, runtime session).** This no-calibration chain
> *builds* but the resulting decoder parts **do not load on the X2E HTP** —
> they fail context-create with **QNN 1002**. Root cause: with **no
> `--input_list`**, `qairt-quantizer` never computes activation encodings, so
> the HTP compiles a **float (fp16) graph** and the int8 weights are stored
> back as **fp16** → each 5-layer decoder context is **3.30 GB** (= 2 B/param)
> and exceeds the **~2 GB X2E runtime per-context ceiling**. (The embed/head
> parts are 1.56 GB and load; Qualcomm's loadable 7B parts are ≤1.09 GB with
> **uint16** IO.) **For a loadable w8a16 you MUST calibrate:** pass
> `--input_list <cal.txt>` (per-part activations from
> `end-to-end/lib/cal.py::cal_iter`) and **drop `--preserve_io_datatype`** so
> KV/activation IO quantizes to uint16/uint8 — that forces the real int8-weight
> HTP path (~1.65 GB parts). Full analysis + fix spec:
> `docs/npu_engine_14b_runtime.md` §4–5.

**Split** (`lib/split.py`) needs two adaptations, both done:
- `extract_part` streams part external data (the protobuf-2 GiB fix, like the
  rewrites) instead of materializing the 3.1 GB embed/lm_head inline.
- transformers 4.57's fold-pathbmask emits a **live additive `attention_bias`
  input** (not the old internal folded mask) → declare it as a direct input
  on every decoder part (`split_14b.py`), not the `shared_mask` thread path.

**`specula-qairt` image deps (all required, all found the hard way):**
- `libc++1 libc++abi1` — the clang-built QAIRT binaries need `libc++.so.1`.
- `numpy==1.26.4` — QAIRT's C ABI.
- **`onnx==1.18.0`** — `qairt-converter` calls `onnx.version`, which onnx
  ≥1.22 removed. protobuf version is irrelevant (X2E runs 6.31.1).
- **`onnxsim`** — without it `qairt-converter` skips simplification and
  **mis-infers the rotary `rotate_half` head_dim as 127** (dynamic
  `Shape→Div→Slice` unfolded) → `getBroadcastedTensorShape [1,40,1,127] vs
  [1,1,1,128]` on every decoder part. With onnxsim it folds to 128 and
  converts. (part1/embed has no attention so it built without onnxsim — the
  tell.)
- **`QAIRT_TMP_DIR=/workspace/tmp`** — the converter spills >2 GB simplified
  models to `$TMPDIR`; the container's `/tmp` is tiny → part8 (16 GB) failed
  "Failed to copy external data". Point it at the SSD.

### Status — COMPLETE ✅

**Full Qwen3-14B w8a16 NPU bundle built end-to-end on the box** — export → 4
rewrites → split → convert → quantize → context-bin → assemble → 28 GB
genie-shaped bundle, **no AIMET, no cloud**, pulled to the X2E
(`models/qwen3_14b-w8a16-specula-x2e/`).

**Split balancing (HTP per-context limits — learned by failing):**
- A **10-layer part (~13 GB)** fails `qnn-context-binary-generator` with
  **QNN 1002 (graph finalize)** — the HTP per-context ceiling is **~5 layers
  / ~3.3 GB `.bin`**. Split decoder layers ≤5 per part.
- The **3.1 GB lm_head must be its own part** — mixed with attention layers
  it breaks the converter's symbolic shape inference (`TensorProto exceeded
  2GB` on `/lm_head/Transpose`), which then leaves the rotary 127 unfolded.
- Working layout for 14B (40 layers): **embed + 8×(5-layer) + lm_head = 10
  parts**. (`build_part_specs` default puts extra layers + lm_head in the
  last part — override it; for the 27B, plan ~5-layer decoder parts + a
  standalone lm_head part from the start.)

**Deploy caveat:** 10 parts > the **~7 ORT-QNN HTP session ceiling**
([[reference_ortqnn_session_limit]]) — loading needs the combined-wrapper /
sidecar. The build is done; on-device run is the next runtime step.

Box drivers (in `end-to-end/build_server/`): `run_stages_1_5.sh`,
`split_14b.py` (+ `split_tail2.py` for the balanced tail), `build_qairt_14b.sh`
(+ `build_part8_9.sh`), `bundle_14b.py`, `Dockerfile.qairt`, `qnn_v81_box.json`.

## Next / TODO

- Finish the 8-part build → assemble bundle (`lib/bundle.py`) → rsync to X2E
  → load via ORT-QNN on the Hexagon.
- ⚠️ **Calibration is REQUIRED, not optional** (corrected 2026-06-16). The
  no-`--input_list` build produces fp16-weight decoder contexts (3.30 GB) that
  fail to load on the X2E HTP (QNN 1002, exceed ~2 GB/context). Add the calib
  set + drop `--preserve_io_datatype` for the *next* build (adapt
  `capture_calibration_qwen3_4b.py` / `lib/cal.py::cal_iter`: 40 layers,
  head_dim 128, ctx 512). See `docs/npu_engine_14b_runtime.md` §4–5.
- v2: a FastAPI job service in the `specula-qairt` image (POST a build, poll,
  download the bundle) instead of ad-hoc ssh.
- Fold the box-side drivers into the repo (`split_14b.py`, `build_qairt_14b.sh`,
  `Dockerfile.qairt`, `qnn_v81_box.json`) once the bundle lands.
