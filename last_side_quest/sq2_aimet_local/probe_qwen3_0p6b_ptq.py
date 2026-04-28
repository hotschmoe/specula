"""SQ2 — basic PTQ on Qwen3-0.6B with AIMET v2 PyTorch.

Goal: verify that AIMET v2 quantsim + compute_encodings (no SEQ_MSE)
runs on a real Qwen3-class LLM on the local CPU, end to end. Output:
- emit aimet_basic_ptq_qwen3_0p6b/encodings.json
- print cos(fp32 logits, quantized logits) on a small probe prompt
- timing per stage

Run from the SQ2 venv:
  last_side_quest/sq2_aimet_local/.venv-aimet-x86/Scripts/python.exe \
    last_side_quest/sq2_aimet_local/probe_qwen3_0p6b_ptq.py

This is intentionally minimal — no SEQ_MSE / AdaScale on the first
pass, just to confirm the pipeline. SEQ_MSE follow-up only if basic
PTQ goes through clean.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Use the project's local HF cache so downloads land where we expect.
REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(REPO / "models" / ".hf_cache"))

OUT_DIR = REPO / "last_side_quest" / "sq2_aimet_local" / "out_qwen3_0p6b_basic_ptq"


def _logits(out):
    """Pull logits out of (logits,) tuple or (logits, ...) tuple, or .logits."""
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


class LogitsOnly(__import__("torch").nn.Module):
    """Wrap an HF causal-LM so forward returns just the logits tensor.

    AIMET v2 quantsim requires the traced module to return a tensor
    (or tuple/list/dict of tensors), not a transformers ModelOutput.
    """

    def __init__(self, hf_model):
        super().__init__()
        self.inner = hf_model

    def forward(self, input_ids, attention_mask=None):
        out = self.inner(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return out.logits


def main() -> int:
    import torch
    import torch.nn.functional as F

    print(f"[setup] HF_HOME = {os.environ['HF_HOME']}")
    print(f"[setup] out_dir = {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load Qwen3-0.6B from HF
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "Qwen/Qwen3-0.6B"
    t0 = time.time()
    print(f"\n[step 1] loading {model_id} (FP32, CPU)…")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, attn_implementation="eager",
    ).eval()
    print(f"[step 1] loaded in {time.time()-t0:.1f}s. "
          f"layers={model.config.num_hidden_layers}, "
          f"hidden={model.config.hidden_size}, "
          f"heads={model.config.num_attention_heads}")

    # Wrap so forward returns just logits — AIMET's tracer wants a tensor.
    model.config.use_cache = False
    model.generation_config = None
    model = LogitsOnly(model).eval()

    # ---- 2. Build calibration tensors
    cal_prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "def fibonacci(n):\n    if n < 2:\n        return n",
        "User: Hello!\nAssistant: Hi there, how can I help?",
        "import json\ndata = json.loads('{\"a\": 1}')",
    ]
    # Pad to a uniform shape so AIMET's traced graph sees stable shapes.
    cal_pairs = []
    for p in cal_prompts:
        enc = tok(p, return_tensors="pt", truncation=True,
                  padding="max_length", max_length=64)
        cal_pairs.append((enc["input_ids"], enc["attention_mask"]))
    print(f"\n[step 2] calibration set: {len(cal_pairs)} prompts, "
          f"max len 64 tokens.")

    # ---- 3. Float reference logits on the probe prompt (also padded to 64)
    probe = tok(
        "The capital of France is",
        return_tensors="pt", padding="max_length", max_length=64,
    )
    probe_ids = probe["input_ids"]
    probe_mask = probe["attention_mask"]
    last_real_pos = int(probe_mask[0].sum().item()) - 1
    with torch.no_grad():
        fp_logits = model(probe_ids, probe_mask)
    print(f"[step 3] fp32 probe logits: shape={tuple(fp_logits.shape)}, "
          f"last_real_pos={last_real_pos}, "
          f"argmax_last_real={tok.decode(fp_logits[0, last_real_pos].argmax().item())!r}")

    # ---- 4. Build the v2 quantsim
    print("\n[step 4] building AIMET v2 QuantizationSimModel "
          "(w4 sym weights / a16 asym acts)…")
    t0 = time.time()
    from aimet_torch.common.defs import QuantScheme
    from aimet_torch.v2.quantsim import QuantizationSimModel

    dummy_in = (probe_ids, probe_mask)
    sim = QuantizationSimModel(
        model,
        dummy_input=dummy_in,
        quant_scheme=QuantScheme.post_training_tf_enhanced,
        default_param_bw=4,
        default_output_bw=16,
    )
    print(f"[step 4] sim built in {time.time()-t0:.1f}s.")

    # Print one sample QuantizedLinear so we can see what landed
    sample = None
    for name, mod in sim.model.named_modules():
        if type(mod).__name__ == "QuantizedLinear":
            sample = (name, mod)
            break
    if sample is not None:
        n, mod = sample
        print(f"[step 4] sample QuantizedLinear: {n}")
        print(f"         {mod}")

    # ---- 5. compute_encodings (basic PTQ calibration)
    print("\n[step 5] running compute_encodings on calibration set…")
    t0 = time.time()

    def fwd_fn(m, _):
        with torch.no_grad():
            for ids, mask in cal_pairs:
                m(ids, mask)

    sim.compute_encodings(fwd_fn, None)
    print(f"[step 5] compute_encodings done in {time.time()-t0:.1f}s.")

    # ---- 6. Quantized probe pass + cos vs fp32 (only at real-token positions)
    with torch.no_grad():
        q_logits = sim.model(probe_ids, probe_mask)
    fp_real = fp_logits[0, : last_real_pos + 1, :].flatten().float()
    q_real = q_logits[0, : last_real_pos + 1, :].flatten().float()
    cos = float(F.cosine_similarity(fp_real, q_real, dim=0))
    argmax_q = q_logits[0, last_real_pos].argmax().item()
    print(f"\n[step 6] quantized probe logits: "
          f"shape={tuple(q_logits.shape)}, "
          f"argmax_last_real={tok.decode(argmax_q)!r}")
    print(f"[step 6] cos(fp32, quant) over real positions = {cos:.6f}")

    # ---- 7. Save encodings.json only (sim.export emits 5+ GB of weight
    #         files for a 0.6B model and trips a protobuf size cap when
    #         it tries to serialize the wrapping ONNX). save_encodings_to_json
    #         is the v2.20+ recommended way to extract just the per-tensor
    #         scales/offsets without the giant ONNX dump.
    print("\n[step 7] saving encodings.json (skipping ONNX export)…")
    t0 = time.time()
    enc_path = OUT_DIR / "qwen3_0p6b_basic_ptq.encodings.json"
    try:
        sim.save_encodings_to_json(str(OUT_DIR),
                                    "qwen3_0p6b_basic_ptq.encodings")
    except Exception as e:
        import traceback
        print(f"[step 7] save_encodings_to_json FAILED: "
              f"{type(e).__name__}: {e}")
        traceback.print_exc()
    print(f"[step 7] encodings save done in {time.time()-t0:.1f}s.")

    print("\n--- emitted files ---")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size} B)")

    print("\nSQ2 PTQ probe complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
