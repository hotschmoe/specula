"""Run ORT's graph optimizer over the ONNX and save the result.

AI Hub's compile pipeline tries to run onnxsim but their environment
doesn't ship it (visible in the job log: 'Onnx model simplification
failed due to: No module named onnxsim'). Without simplification, the
optimum --no-post-process export keeps a dynamic attention-mask
subgraph (Range/Flatten/Gather) with mixed int+float dtypes that HTP's
Gather op rejects.

onnxsim itself has no Windows-on-ARM wheel, so we can't run it locally
on this machine. ORT's built-in graph optimizer (ORT_ENABLE_ALL) does
a substantial subset of the same constant-folding and fusion passes
and ships natively with onnxruntime-qnn.

Inputs:  models/<source_dir>/model.onnx (+ model.onnx_data or model.data)
Outputs: models/<source_dir>-ortopt/model.onnx (+ model.data)
         Ready for prep_onnx_for_ai_hub.py to stage directly.

Run:
    .venv\\Scripts\\python.exe scripts\\ort_optimize_onnx.py
"""

import sys
import time
from pathlib import Path

import onnxruntime as ort


REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE = REPO_ROOT / "models" / "qwen3-0.6b-optimum" / "model.onnx"
DEST_DIR = REPO_ROOT / "models" / "qwen3-0.6b-optimum-ortopt"
DEST_ONNX = DEST_DIR / "model.onnx"


def main() -> int:
    if not SOURCE.exists():
        print(f"ERROR: source missing at {SOURCE}")
        return 2

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    # BASIC = constant folding + graph cleanup, NO op fusion. ENABLE_ALL
    # (tried first) fused ops into com.microsoft::FusedMatMul and
    # ::QuickGelu, which is the exact failure mode we escaped by using
    # optimum --no-post-process. Stick to BASIC so we keep a clean
    # default-domain graph but still get constant folding of the
    # Range/Shape-based mask subgraph that tripped HTP's Gather_5.
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess_opts.optimized_model_filepath = str(DEST_ONNX)
    sess_opts.log_severity_level = 3

    print(f"source        : {SOURCE}")
    print(f"dest          : {DEST_ONNX}")
    print(f"optimization  : ORT_ENABLE_BASIC (no op fusion)")
    print("loading + optimizing (CPU provider) ...")

    t0 = time.perf_counter()
    sess = ort.InferenceSession(
        str(SOURCE),
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"],
    )
    del sess  # force save
    elapsed = time.perf_counter() - t0
    print(f"done in {elapsed:.1f} s")

    # Summarise output
    onnx_mb = DEST_ONNX.stat().st_size / (1024 * 1024)
    print(f"\noutput files:")
    for p in sorted(DEST_DIR.iterdir()):
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  {p.name:30s} {size_mb:8.1f} MB")

    print(f"\n--- next steps ---")
    print(f"1. Verify node counts + ops:")
    print(f"   python scripts/inspect_onnx_ops.py --model {DEST_ONNX}")
    print(f"2. Re-point prep + compile scripts at {DEST_DIR.name}/ (or update paths).")
    print(f"3. python scripts/prep_onnx_for_ai_hub.py")
    print(f"4. python scripts/compile_qwen3_ai_hub.py --submit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
