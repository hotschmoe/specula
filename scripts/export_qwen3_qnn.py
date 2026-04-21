"""DEPRECATED: this path does not actually decompose ORT fused ops.

Kept for reference. Tested against onnxruntime-genai 0.13.1 (current PyPI
latest, 2026-04-20): the `execution_provider="qnn"` flag is silently
accepted but there is no `qnn` entry in the builder's `ep_attrs` dict
(only cpu/cuda/dml/webgpu/trt-rtx), and `grep -ri qnn` across the
installed package returns zero matches. The produced ONNX still contains
`com.microsoft::RotaryEmbedding`, `SkipSimplifiedLayerNormalization`,
and `MultiHeadAttention` — the exact ops QAIRT's QNN converter rejects.
The call then crashes at `make_genai_config` with `KeyError: 'qnn'`.

Use the optimum path instead — see docs/phase5_export_on_x86.md.
"""

import argparse
import platform
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "models" / "qwen3-0.6b-qnn-source"
DEFAULT_CACHE = REPO_ROOT / "models" / ".hf_cache"

MODEL_NAME = "Qwen/Qwen3-0.6B"
EXECUTION_PROVIDER = "qnn"


def warn_if_woa() -> None:
    """Guard rail: refuse to run on Windows-on-ARM64 where torch is unavailable."""
    machine = platform.machine().lower()
    system = platform.system().lower()
    if system == "windows" and machine in ("arm64", "aarch64"):
        print("ERROR: this script is meant to run on x86_64, not Windows-on-ARM64.")
        print("  torch has no cp312 win_arm64 wheel; the HF PyTorch trace would fail.")
        print("  Run this on a Linux x86_64 or Windows x86_64 machine, then copy the")
        print("  output directory back to the ARM64 target.")
        print("  See docs/phase5_export_on_x86.md for the full procedure.")
        sys.exit(2)


def main() -> int:
    warn_if_woa()

    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=["fp16", "int4", "fp32"], default="fp16")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
        help=f"HF model id (default: {MODEL_NAME})",
    )
    args = parser.parse_args()

    try:
        from onnxruntime_genai.models.builder import create_model
    except ImportError:
        print("ERROR: onnxruntime-genai not installed in this environment.")
        print("  pip install onnxruntime-genai torch transformers huggingface_hub")
        print("  (see docs/phase5_export_on_x86.md section 'Install')")
        return 2

    args.output.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)

    print(f"model            : {args.model_name}")
    print(f"precision        : {args.precision}")
    print(f"execution_provider: {EXECUTION_PROVIDER}")
    print(f"output dir       : {args.output}")
    print(f"HF cache dir     : {args.cache}")
    print()

    t0 = time.perf_counter()
    create_model(
        model_name=args.model_name,
        input_path="",  # empty -> download from HF
        output_dir=str(args.output),
        precision=args.precision,
        execution_provider=EXECUTION_PROVIDER,
        cache_dir=str(args.cache),
        hf_token="false",  # Qwen3-0.6B is public; skip local-token lookup
    )
    print(f"\nexport complete in {time.perf_counter() - t0:.1f} s")

    print("\n--- output files ---")
    total_mb = 0
    for p in sorted(args.output.rglob("*")):
        if p.is_file():
            size_mb = p.stat().st_size / (1024 * 1024)
            total_mb += size_mb
            rel = p.relative_to(args.output)
            print(f"  {str(rel):40s} {size_mb:8.1f} MB")
    print(f"  {'TOTAL':40s} {total_mb:8.1f} MB")

    print("\n--- next steps ---")
    print("1. Verify no com.microsoft ops remain (local to this machine):")
    onnx_files = list(args.output.rglob("*.onnx"))
    for f in onnx_files:
        print(f"   python scripts/inspect_onnx_ops.py --model {f}")
    print("2. Transfer the output dir to the Snapdragon X2E machine, e.g.:")
    print(f"   scp -r {args.output}/ <user>@<x2e-host>:<specula-path>/models/")
    print("   (or rsync / git-lfs / USB / your transfer tool of choice)")
    print("3. On the X2E, re-run the compile pipeline pointing at the new source:")
    print("   python scripts/prep_onnx_for_ai_hub.py --source qwen3-0.6b-qnn-source")
    print("   python scripts/compile_qwen3_ai_hub.py --submit")

    return 0


if __name__ == "__main__":
    sys.exit(main())
