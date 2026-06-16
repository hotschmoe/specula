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

## Next / TODO

- Calibration capture for 14B (`capture_calibration_qwen3_4b.py` /
  `qairt_prep_calibration_4b.py` — adapt the `_4b` shapes: 40 layers, 8 kv
  heads, head_dim 128, ctx 512).
- split → convert → quantize w8a16 → ctx-bin → bundle → rsync to X2E.
- v2: a FastAPI job service in the `specula-qairt` image (POST a build, poll,
  download the bundle) instead of ad-hoc ssh.
