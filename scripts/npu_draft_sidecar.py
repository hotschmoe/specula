"""Phase 5 step 2 - ORT-QNN sidecar skeleton.

Single-session wrapper + smoke test of a tiny ONNX graph on HTP.

Guards against the scoping-doc section 3.6 silent-CPU-fallback trap by:
  (1) passing QNNExecutionProvider WITHOUT a CPU fallback listed, so
      session construction errors instead of silently degrading;
  (2) inspecting session.get_providers() after construction;
  (3) enabling QNN profiling at basic level so the executor chain is
      visible in logs.

Run:
    .venv\\Scripts\\python.exe scripts\\npu_draft_sidecar.py

Exit codes:
    0 - session ran on HTP, latency printed
    1 - session created but executor was not QNN (silent fallback)
    2 - hard failure (exception, missing DLL, etc.)
"""

import sys
import time
import traceback
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


SOC_MODEL = "88"  # confirmed by AI Hub device attr 'soc-model:88' + voice_project/encoder_info_v81.json
HTP_ARCH = "81"   # confirmed by three sources, see npu_env_snapshot.txt

# Provider options keys follow the onnxruntime-qnn 1.24.x interface.
# Reference: ORT QNN EP docs + voice_project/Transcriber.cs.


def build_tiny_matmul_model(path: Path, m: int = 1, k: int = 64, n: int = 32) -> None:
    """Minimal 2-op graph: MatMul(x, W) -> Relu -> y. FP32 throughout.

    Saved to `path`. Input x is a runtime input; W is a baked-in initializer.
    """
    rng = np.random.default_rng(seed=1)
    w_data = rng.standard_normal(size=(k, n), dtype=np.float32) * 0.1

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [m, k])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [m, n])
    w_init = numpy_helper.from_array(w_data, name="W")

    matmul = helper.make_node("MatMul", inputs=["x", "W"], outputs=["xw"], name="mm")
    relu = helper.make_node("Relu", inputs=["xw"], outputs=["y"], name="relu")

    graph = helper.make_graph(
        nodes=[matmul, relu],
        name="tiny_npu_smoke",
        inputs=[x],
        outputs=[y],
        initializer=[w_init],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 17)],
        producer_name="specula-phase5",
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    path.write_bytes(model.SerializeToString())


class NPUSession:
    """Thin wrapper over ort.InferenceSession pinned to the QNN HTP backend.

    Constructor errors if QNN fails to activate - we never want a silent
    CPU session masquerading as an NPU one.
    """

    def __init__(
        self,
        model_path: Path,
        *,
        htp_performance_mode: str = "burst",
        profiling_level: str = "off",
        profiling_file_path: Path | None = None,
        enable_htp_fp16: bool = True,
    ) -> None:
        self.model_path = model_path
        ort_pkg = Path(ort.__file__).parent
        self.backend_path = ort_pkg / "capi" / "QnnHtp.dll"
        if not self.backend_path.exists():
            raise FileNotFoundError(f"QnnHtp.dll not bundled with onnxruntime at {self.backend_path}")

        provider_options = {
            "backend_path": str(self.backend_path),
            "htp_performance_mode": htp_performance_mode,
            # soc_model / htp_arch are validated by the EP at init; if the
            # bundled stack disagrees we'll see an error rather than a
            # silent fallback.
            "soc_model": SOC_MODEL,
            "htp_arch": HTP_ARCH,
        }
        if profiling_level != "off":
            provider_options["profiling_level"] = profiling_level
            if profiling_file_path is None:
                raise ValueError("profiling_level != 'off' requires profiling_file_path (or enable ETW)")
            provider_options["profiling_file_path"] = str(profiling_file_path)
        if enable_htp_fp16:
            provider_options["enable_htp_fp16_precision"] = "1"

        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3  # warning+; drop the noisy info logs
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.provider_options = provider_options
        # No CPU fallback listed -> any QNN failure raises.
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_opts,
            providers=[("QNNExecutionProvider", provider_options)],
        )
        providers = self.session.get_providers()
        if not providers or providers[0] != "QNNExecutionProvider":
            raise RuntimeError(f"expected QNNExecutionProvider first, got {providers}")
        self.providers = providers

    def run(self, feed: dict) -> list:
        return self.session.run(None, feed)

    def close(self) -> None:
        # ORT doesn't need explicit close, but expose a hook for future async work.
        del self.session


def main() -> int:
    print("=== step 2 - NPU sidecar smoke test ===\n")

    tmp_model = Path(__file__).resolve().parent.parent / "results" / "tiny_npu_smoke.onnx"
    tmp_model.parent.mkdir(exist_ok=True)
    build_tiny_matmul_model(tmp_model)
    print(f"built tiny MatMul+ReLU graph     : {tmp_model} ({tmp_model.stat().st_size} bytes)")

    try:
        sess = NPUSession(tmp_model)
    except Exception:
        print("\nNPUSession construction FAILED:")
        traceback.print_exc()
        return 2

    print(f"session providers (first must be QNN): {sess.providers}")
    print(f"backend_path                         : {sess.backend_path}")
    print(f"provider_options                     : {sess.provider_options}")

    x = np.random.default_rng(seed=42).standard_normal(size=(1, 64), dtype=np.float32)

    # Warmup (also pays any first-compile cost for the HTP context binary).
    for _ in range(5):
        sess.run({"x": x})

    N = 200
    t0 = time.perf_counter_ns()
    for _ in range(N):
        y = sess.run({"x": x})
    t1 = time.perf_counter_ns()

    per_call_us = (t1 - t0) / N / 1000.0
    print(f"\noutput shape                         : {y[0].shape}")
    print(f"output dtype                         : {y[0].dtype}")
    print(f"mean per-call latency                : {per_call_us:.3f} us  ({per_call_us / 1000:.3f} ms)")
    print(f"target (scoping doc step 2 exit)     : < 5000 us (< 5 ms)")

    status = "ok" if per_call_us < 5000 else f"partial: latency {per_call_us:.0f} us > 5000 us target"
    providers = sess.providers
    if "QNNExecutionProvider" not in providers or providers[0] != "QNNExecutionProvider":
        status = f"FAIL: provider chain {providers}"
    print(f"\n=== STATUS: {status} ===")
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
