"""aihub_inference_probe.py — Stage 3: HTP numerics vs eager.

Runs the compiled X2 Elite binary (the compile job from aihub_compile_probe.py)
on the dumped reference input via AI Hub `submit_inference_job`, and compares
the returned logits to the eager torch reference. Op-support is proven
(Stage 2); this checks the gated-delta-net computes the *right answer* on
real HTP silicon.

Two-venv handoff (torch and qai_hub don't co-habit):
  1. .venv-arm-export: op_compilability_probe.py --dump-ref  (writes ref_*.npy)
  2. .venv (this script): submit_inference_job + compare

Usage: .venv/Scripts/python.exe end-to-end/probes/aihub_inference_probe.py [compile_job_id]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import qai_hub as hub

OUT = Path(__file__).resolve().parent / "out"
DEVICE = "Snapdragon X2 Elite CRD"


def main(compile_job_id: str) -> int:
    ref_ids = OUT / "ref_input_ids.npy"
    ref_logits = OUT / "ref_logits.npy"
    if not ref_ids.exists() or not ref_logits.exists():
        print(f"missing reference npy in {OUT} — run "
              "`op_compilability_probe.py --dump-ref` in .venv-arm-export first")
        return 2

    input_ids = np.load(ref_ids)               # int64 (1, seq)
    eager = np.load(ref_logits).astype(np.float32)
    print(f"ref input_ids {input_ids.shape}  eager logits {eager.shape}")

    cjob = hub.get_job(compile_job_id)
    target = cjob.get_target_model()
    if target is None:
        print(f"compile job {compile_job_id} has no target model (not SUCCESS?)")
        return 1

    ijob = hub.submit_inference_job(
        model=target,
        device=hub.Device(DEVICE),
        inputs={"input_ids": [input_ids.astype(np.int32)]},
        name="qwen3next-ssm-numerics",
    )
    print(f"inference job: {ijob.job_id}  {ijob.url}")
    ijob.wait()
    status = ijob.get_status()
    if not status.success:
        print(f"inference FAILED: {status.code} — {getattr(status, 'message', '')}")
        return 1

    data = ijob.download_output_data()
    key = "logits" if "logits" in data else list(data)[0]
    htp = np.asarray(data[key][0]).astype(np.float32).reshape(eager.shape)

    e, h = eager.reshape(-1), htp.reshape(-1)
    cos = float(np.dot(e, h) / (np.linalg.norm(e) * np.linalg.norm(h) + 1e-9))
    max_abs = float(np.max(np.abs(e - h)))
    e_last, h_last = eager[0, -1], htp[0, -1]
    argmax_match = bool(np.argmax(e_last) == np.argmax(h_last))
    # top-5 overlap on the last token
    top5_e = set(np.argsort(e_last)[-5:].tolist())
    top5_h = set(np.argsort(h_last)[-5:].tolist())

    print()
    print("STAGE 3 RESULT — HTP (X2 Elite) vs eager torch")
    print(f"  cosine sim         : {cos:.5f}")
    print(f"  max abs diff       : {max_abs:.4f}")
    print(f"  last-token argmax  : {'MATCH' if argmax_match else 'MISMATCH'}")
    print(f"  last-token top5 ovl: {len(top5_e & top5_h)}/5")
    if cos > 0.99 and argmax_match:
        print("  VERDICT: numerics match — gated-delta-net computes correctly on HTP.")
    elif cos > 0.95:
        print("  VERDICT: close (likely quant/precision) — inspect before trusting.")
    else:
        print("  VERDICT: diverged — op accepted but math wrong; needs investigation.")
    return 0


if __name__ == "__main__":
    jid = sys.argv[1] if len(sys.argv) > 1 else "j5qw8d6m5"
    sys.exit(main(jid))
