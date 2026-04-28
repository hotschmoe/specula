"""SQ3 — basic PTQ on Granite-3.0-1B-A400M-Instruct (MoE) with AIMET v2.

Goal: answer the open question from SQ2 — does AIMET v2's generic
auto-quantization handle a non-Qwen MoE architecture, given that AIMET
2.29 ships a `qwen3_moe` adapter but **no `granitemoe` adapter**?

Run from the SQ2 venv (reused — same pin works for any HF causal LM):
  last_side_quest/sq2_aimet_local/.venv-aimet-x86/Scripts/python.exe \
    last_side_quest/sq3_small_moe/probe_granite_1b_a400m_ptq.py

This is intentionally minimal:
- HF-load granite-3.0-1b-a400m-instruct (FP32, eager attn, no cache)
- Wrap to logits-only so AIMET tracer sees a tensor return
- Build QuantizationSimModel w4 sym weights / a16 asym acts
- Inspect: did AIMET wrap per-expert linears? what about the router?
- compute_encodings on 4 short calibration prompts
- Quantized probe forward + cos vs FP32 + argmax probe
- save_encodings_to_json

Three failure modes to watch for:
1. quantsim *construct* fails — tracer chokes on MoE router top-k op
   ⇒ verdict "AIMET MoE support is qwen3_moe-specific"
2. construct OK, compute_encodings fails — per-expert routing breaks
   PTQ calibration data flow
3. Both run, cos catastrophic — matches SQ2's V/O-collapse story; need
   SEQ_MSE / AdaScale escalation
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(REPO / "models" / ".hf_cache"))

OUT_DIR = REPO / "last_side_quest" / "sq3_small_moe" / "out_granite_1b_a400m_basic_ptq"

MODEL_ID = "ibm-granite/granite-3.0-1b-a400m-instruct"


class LogitsOnly(__import__("torch").nn.Module):
    """Wrap an HF causal-LM so forward returns just the logits tensor.

    Identical to SQ2 — AIMET v2 quantsim requires tensor return, not
    transformers ModelOutput.
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


def _count_module_types(model):
    counts = {}
    for _, mod in model.named_modules():
        t = type(mod).__name__
        counts[t] = counts.get(t, 0) + 1
    return counts


def main() -> int:
    import torch
    import torch.nn.functional as F

    print(f"[setup] HF_HOME = {os.environ['HF_HOME']}")
    print(f"[setup] out_dir = {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    print(f"\n[step 1] loading {MODEL_ID} (FP32, eager attn, CPU)…")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, attn_implementation="eager",
    ).eval()
    cfg = model.config
    print(f"[step 1] loaded in {time.time()-t0:.1f}s. "
          f"arch={type(model).__name__}, "
          f"layers={cfg.num_hidden_layers}, "
          f"hidden={cfg.hidden_size}, "
          f"experts/layer={getattr(cfg, 'num_local_experts', '?')}, "
          f"top_k={getattr(cfg, 'num_experts_per_tok', '?')}")

    pre_counts = _count_module_types(model)
    moe_class = next((k for k in pre_counts if "MoE" in k or "moe" in k.lower()),
                     None)
    print(f"[step 1] MoE module class detected: {moe_class!r} "
          f"(count={pre_counts.get(moe_class, 0) if moe_class else 0})")
    print(f"[step 1] nn.Linear count = {pre_counts.get('Linear', 0)} "
          "(includes per-expert FFN linears)")

    model.config.use_cache = False
    model.generation_config = None
    model = LogitsOnly(model).eval()

    cal_prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "def fibonacci(n):\n    if n < 2:\n        return n",
        "User: Hello!\nAssistant: Hi there, how can I help?",
        "import json\ndata = json.loads('{\"a\": 1}')",
    ]
    cal_pairs = []
    for p in cal_prompts:
        enc = tok(p, return_tensors="pt", truncation=True,
                  padding="max_length", max_length=64)
        cal_pairs.append((enc["input_ids"], enc["attention_mask"]))
    print(f"\n[step 2] calibration set: {len(cal_pairs)} prompts × 64 tokens.")

    probe = tok("The capital of France is", return_tensors="pt",
                padding="max_length", max_length=64)
    probe_ids = probe["input_ids"]
    probe_mask = probe["attention_mask"]
    last_real_pos = int(probe_mask[0].sum().item()) - 1
    with torch.no_grad():
        fp_logits = model(probe_ids, probe_mask)
    fp_argmax = tok.decode(fp_logits[0, last_real_pos].argmax().item())
    print(f"[step 3] fp32 probe: shape={tuple(fp_logits.shape)}, "
          f"last_real_pos={last_real_pos}, argmax={fp_argmax!r}")

    print("\n[step 4] building AIMET v2 QuantizationSimModel "
          "(w4 sym weights / a16 asym acts) — this is the critical step…")
    t0 = time.time()
    from aimet_torch.common.defs import QuantScheme
    from aimet_torch.v2.quantsim import QuantizationSimModel
    # Register Granite-MoE adapters (AIMET 2.29 has no built-in granitemoe
    # adapter; without these three @QuantizationMixin.implements registrations
    # the sim builder errors out with "Quantized module definitions of the
    # following modules are not registered: GraniteMoeParallelExperts, ...").
    import granite_moe_adapters  # noqa: F401  (registration side effect)

    dummy_in = (probe_ids, probe_mask)
    try:
        sim = QuantizationSimModel(
            model,
            dummy_input=dummy_in,
            quant_scheme=QuantScheme.post_training_tf_enhanced,
            default_param_bw=4,
            default_output_bw=16,
        )
    except Exception as e:
        import traceback
        print(f"\n[step 4] FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        print("\n=== verdict: AIMET v2 quantsim builder cannot trace "
              "GraniteMoe forward — MoE-adapter coverage is "
              "qwen3_moe-only at AIMET 2.29 ===")
        return 1
    print(f"[step 4] sim built in {time.time()-t0:.1f}s.")

    post_counts = _count_module_types(sim.model)
    qlinear = post_counts.get("QuantizedLinear", 0)
    print(f"[step 4] QuantizedLinear modules in sim: {qlinear} "
          f"(was {pre_counts.get('Linear', 0)} nn.Linear pre-sim)")

    # Show one router-gate quantizer if we can find it.
    for name, mod in sim.model.named_modules():
        if type(mod).__name__ == "QuantizedLinear" and "router" in name.lower():
            print(f"[step 4] sample router QuantizedLinear: {name}\n         {mod}")
            break

    print("\n[step 5] running compute_encodings on calibration set…")
    t0 = time.time()

    def fwd_fn(m, _):
        with torch.no_grad():
            for ids, mask in cal_pairs:
                m(ids, mask)

    try:
        sim.compute_encodings(fwd_fn, None)
    except Exception as e:
        import traceback
        print(f"\n[step 5] FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        print("\n=== verdict: AIMET sim builds against GraniteMoe but "
              "compute_encodings can't run; likely per-expert routing "
              "breaks AIMET's calibration data flow ===")
        return 2
    print(f"[step 5] compute_encodings done in {time.time()-t0:.1f}s.")

    with torch.no_grad():
        q_logits = sim.model(probe_ids, probe_mask)
    fp_real = fp_logits[0, : last_real_pos + 1, :].flatten().float()
    q_real = q_logits[0, : last_real_pos + 1, :].flatten().float()
    cos = float(F.cosine_similarity(fp_real, q_real, dim=0))
    argmax_q = q_logits[0, last_real_pos].argmax().item()
    print(f"\n[step 6] quantized probe: argmax={tok.decode(argmax_q)!r}")
    print(f"[step 6] cos(fp32, quant) over real positions = {cos:.6f}")

    print("\n[step 7] saving encodings.json…")
    t0 = time.time()
    try:
        sim.save_encodings_to_json(
            str(OUT_DIR), "granite_1b_a400m_basic_ptq.encodings"
        )
    except Exception as e:
        import traceback
        print(f"[step 7] save FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
    print(f"[step 7] encodings save done in {time.time()-t0:.1f}s.")

    print("\n--- emitted files ---")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size} B)")

    print("\nSQ3 Granite-MoE PTQ probe complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
