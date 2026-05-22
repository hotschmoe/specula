"""Qualcomm AI Hub cloud path for Gemma 4 -> Snapdragon.

The local pipeline (`quantize_to_npu.py`) needs a CUDA box for the
AIMET step and a Snapdragon for on-device validation. Qualcomm AI Hub
sidesteps both: it compiles, quantizes, profiles and runs inference on
Qualcomm's *own* cloud devices. All it needs from us is an API token —
no local CUDA, no local NPU. That makes it the one full conversion
path runnable from the current x86 dev box.

As of 2026-05-22 (see AI_HUB.md) AI Hub has NO Gemma recipe — the
pre-built LLM recipes are Llama/Qwen/Phi/Mistral only, and the
bring-your-own path is architecture-locked. So the only usable AI Hub
route is raw job submission with an ONNX we produced ourselves:

  B. Raw `qai-hub` job submission — take our own ONNX (pipeline stage-5
     output) and submit:
       - submit_quantize_job  (server-side PTQ; int8 weights ONLY, no
         int4 -> w8a16 reachable, w4a16 NOT) [optional]
       - submit_compile_job   (ONNX -> QNN context binary, X Elite)
       - submit_inference_job (validate on a real cloud Snapdragon)
     All run on Qualcomm's cloud — no local CUDA, no local NPU.

  A. `qai-hub-models` recipe — kept as a runtime check only, in case
     Qualcomm adds a `gemma4` recipe later.

This script is a PRE-FLIGHT + LAUNCHER: it verifies the token, lists
what Gemma support exists, and prints the exact command to run. It does
not silently burn cloud credits.

Setup (one-time):
    pip install qai-hub qai-hub-models
    qai-hub configure --api_token <YOUR_TOKEN>     # from app.aihub.qualcomm.com

Usage:
    python submit_ai_hub.py --check                # preflight only
    python submit_ai_hub.py --device "Snapdragon X Elite CRD"
"""
from __future__ import annotations

import argparse
import configparser
import importlib
import os
import sys
from pathlib import Path

QAI_HUB_INI = Path.home() / ".qai_hub" / "client.ini"


def check_token() -> bool:
    """Report whether a qai-hub API token is configured."""
    if QAI_HUB_INI.exists():
        cp = configparser.ConfigParser()
        cp.read(QAI_HUB_INI)
        for sect in cp.sections():
            if cp[sect].get("api_token"):
                print(f"[token] found in {QAI_HUB_INI} (section [{sect}])")
                return True
    if os.environ.get("QAI_HUB_API_TOKEN"):
        print("[token] found in $QAI_HUB_API_TOKEN")
        return True
    print(f"[token] NONE — no {QAI_HUB_INI} and no $QAI_HUB_API_TOKEN")
    print("        Get one at https://app.aihub.qualcomm.com/ then run:")
    print("        qai-hub configure --api_token <TOKEN>")
    return False


def check_package(name: str) -> bool:
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "?")
        print(f"[pkg] {name}  {ver}")
        return True
    except ImportError:
        print(f"[pkg] {name}  NOT INSTALLED  (pip install {name})")
        return False


def find_gemma_recipes() -> list[str]:
    """List qai_hub_models model recipes whose name mentions gemma."""
    try:
        import qai_hub_models.models as m
    except ImportError:
        return []
    root = Path(m.__file__).parent
    return sorted(p.name for p in root.iterdir()
                  if p.is_dir() and "gemma" in p.name.lower())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="Preflight only: token + packages + Gemma recipes.")
    ap.add_argument("--device", default="Snapdragon X Elite CRD",
                    help="AI Hub target device name.")
    ap.add_argument("--onnx", type=Path, default=None,
                    help="Path B: submit a compile job for this ONNX "
                         "(pipeline stage-5 output).")
    ap.add_argument("--quantize", action="store_true",
                    help="Before compiling, submit a server-side quantize "
                         "job. NOTE: AI Hub quantize is int8-weights only — "
                         "this yields w8a16, NOT w4a16 (see AI_HUB.md).")
    args = ap.parse_args()

    print("=== Qualcomm AI Hub preflight ===\n")
    have_token = check_token()
    have_hub = check_package("qai_hub")
    have_models = check_package("qai_hub_models")

    recipes = find_gemma_recipes() if have_models else []
    print()
    if recipes:
        print(f"[recipes] Gemma recipes in qai_hub_models: {recipes}")
        print("          Path A available — e.g.:")
        print(f"            python -m qai_hub_models.models.{recipes[0]}.export "
              f"--device \"{args.device}\"")
    elif have_models:
        print("[recipes] no Gemma recipe in this qai_hub_models version.")
        print("          Upgrade (`pip install -U qai-hub-models`) or use "
              "Path B (--onnx).")
    else:
        print("[recipes] qai_hub_models not installed — cannot check.")

    print()
    if not (have_token and have_hub):
        print("BLOCKED: configure a token and install qai-hub before "
              "submitting jobs.")
        return 1

    if args.check:
        print("Preflight OK. Re-run without --check to submit.")
        return 0

    if args.onnx is not None:
        if not args.onnx.exists():
            print(f"FATAL: --onnx not found: {args.onnx}", file=sys.stderr)
            return 2
        import qai_hub as hub

        compile_input = str(args.onnx)
        if args.quantize:
            print(f"\n[path B] submitting quantize job for {args.onnx}")
            print("  NOTE: int8 weights only -> w8a16. For w4a16 quantize "
                  "locally with AIMET instead (see AI_HUB.md Route 2).")
            print("  NOTE: needs calibration data — wire a calibration "
                  "Dataset here (500-1000 samples) before this will run.")
            print("FATAL: quantize-job path needs a calibration Dataset "
                  "uploaded first;\n       build that step before using "
                  "--quantize.", file=sys.stderr)
            return 2

        print(f"\n[path B] submitting compile job for {compile_input} "
              f"-> device {args.device!r}")
        job = hub.submit_compile_job(
            model=compile_input,
            device=hub.Device(args.device),
            options="--target_runtime qnn_context_binary",
        )
        print(f"  compile job: {job.url}")
        print("  follow with submit_profile_job / submit_inference_job once "
              "it completes.")
        return 0

    print("\nNothing to do: pass --onnx <stage5.onnx> for Path B, or run a "
          "Path A\nrecipe export directly (see [recipes] above).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
