"""aihub_compile_probe.py — Stage 2: does the HTP compiler accept the SSM ONNX?

Submits the Option-A exported gated-delta-net ONNX (`out/qwen3_next_tiny.onnx`,
produced by op_compilability_probe.py) to Qualcomm AI Hub `submit_compile_job`
targeting the real Snapdragon X2 Elite (CRD), QNN context binary. Free; the
first real HTP op-validation signal for the gated-delta-net op set (esp.
Where / ScatterElements / IsNaN / Softplus).

Run in the base `.venv` (qai_hub 0.48.0, token in ~/.qai_hub/client.ini):
  .venv/Scripts/python.exe end-to-end/probes/aihub_compile_probe.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import qai_hub as hub

ONNX = Path(__file__).resolve().parent / "out" / "qwen3_next_tiny.onnx"
SEQ = 8
DEVICE = "Snapdragon X2 Elite CRD"


def main() -> int:
    if not ONNX.exists():
        print(f"missing ONNX: {ONNX} — run op_compilability_probe.py first")
        return 2

    print(f"submitting compile job: {ONNX.name} -> {DEVICE}")
    job = hub.submit_compile_job(
        model=str(ONNX),
        device=hub.Device(DEVICE),
        name="qwen3next-ssm-opprobe",
        input_specs={"input_ids": ((1, SEQ), "int64")},
        options="--target_runtime qnn_context_binary --truncate_64bit_io",
    )
    print(f"job_id : {job.job_id}")
    print(f"url    : {job.url}")

    # Poll up to ~3 min; AI Hub compile is server-side + async.
    for _ in range(9):
        status = job.get_status()
        print(f"status : {status.code}"
              + (f" — {status.message}" if getattr(status, 'message', None) else ""))
        if status.finished:
            break
        time.sleep(18)

    status = job.get_status()
    print()
    if status.success:
        print("HTP COMPILE OK — the gated-delta-net op set is accepted by the "
              "X2 Elite QNN compiler. Stage 2 PASS.")
        return 0
    if status.finished:
        print("HTP COMPILE FAILED — read the op-validation error at the url "
              "above; that names the op(s) needing a rewrite (snag 2).")
        return 1
    print(f"still running — check {job.url} (or re-run get_status). job {job.job_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
