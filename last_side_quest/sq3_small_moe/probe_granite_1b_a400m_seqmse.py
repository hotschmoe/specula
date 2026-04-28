"""SQ3 follow-up B — SEQ_MSE on Granite-3.0-1B-A400M-Instruct.

Sequential MSE iterates per-linear-layer, evaluating num_candidates
weight scale clip values per channel and picking the one that
minimizes MSE between the layer's quantized output and FP32 output.
This is the AIMET technique designed to close the V/O-projection
collapse story (`docs/w4a16_investigation.md` Sessions 17-18) — the
exact failure mode SQ2 reproduced on Qwen3-0.6B (cos -0.065) and that
we observed on Granite-1B-A400M (cos +0.656 baseline, +0.712 with
per-channel weights).

SEQ_MSE works on `nn.Linear`-class modules. For Granite-MoE that's
the 121 attention projections + router gates + lm_head, but NOT the
48 GraniteMoeParallelExperts (custom 3D-weight fused experts — those
are weight-encoded by sim.compute_encodings as before).

Flow:
1. Build sim
2. Override per-(expert, out-channel) on ParallelExperts (carryover from A)
3. apply_seq_mse(sim, cal_loader, num_candidates=20) — sets weight
   encodings for the 121 supported modules
4. sim.compute_encodings(...) — sets activation encodings (and weight
   encodings for the 48 non-supported ParallelExperts modules)
5. Quantized probe forward + cos vs FP32 + argmax

Estimated wall-time: 121 modules × 20 candidates × 4 batches of
sub-layer forward = several minutes on Prism CPU. Documented in
findings if it diverges.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Match probe_granite_1b_a400m_ptq.py — same per-channel override on
# fused experts. SEQ_MSE doesn't touch ParallelExperts, so the override
# applies to the same 48 modules independent of SEQ_MSE.
PER_EXPERT_PER_CHANNEL_WEIGHTS = True

# SEQ_MSE search width per channel. Default is 20; smaller is faster
# but lower quality. Qualcomm's published recipes use 20.
NUM_CANDIDATES = 20

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(REPO / "models" / ".hf_cache"))

OUT_DIR = REPO / "last_side_quest" / "sq3_small_moe" / "out_granite_1b_a400m_seqmse"

MODEL_ID = "ibm-granite/granite-3.0-1b-a400m-instruct"


class LogitsOnly(__import__("torch").nn.Module):
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

    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    print(f"\n[step 1] loading {MODEL_ID} (FP32, eager attn, CPU)…")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32, attn_implementation="eager",
    ).eval()
    cfg = model.config
    print(f"[step 1] loaded in {time.time()-t0:.1f}s. "
          f"layers={cfg.num_hidden_layers}, "
          f"experts/layer={cfg.num_local_experts}, top_k={cfg.num_experts_per_tok}")

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
    print(f"\n[step 2] calibration set: {len(cal_pairs)} prompts x 64 tokens.")

    probe = tok("The capital of France is", return_tensors="pt",
                padding="max_length", max_length=64)
    probe_ids = probe["input_ids"]
    probe_mask = probe["attention_mask"]
    last_real_pos = int(probe_mask[0].sum().item()) - 1
    with torch.no_grad():
        fp_logits = model(probe_ids, probe_mask)
    fp_argmax = tok.decode(fp_logits[0, last_real_pos].argmax().item())
    print(f"[step 3] fp32 probe argmax = {fp_argmax!r}")

    print(f"\n[step 4] building AIMET v2 QuantizationSimModel "
          f"(w4 sym / a16 asym)…")
    t0 = time.time()
    from aimet_torch.common.defs import QuantScheme
    from aimet_torch.v2.quantsim import QuantizationSimModel
    import granite_moe_adapters  # noqa: F401

    sim = QuantizationSimModel(
        model,
        dummy_input=(probe_ids, probe_mask),
        quant_scheme=QuantScheme.post_training_tf_enhanced,
        default_param_bw=4,
        default_output_bw=16,
    )
    print(f"[step 4] sim built in {time.time()-t0:.1f}s.")

    if PER_EXPERT_PER_CHANNEL_WEIGHTS:
        from aimet_torch.v2.quantization.affine import QuantizeDequantize
        n_overridden = 0
        for name, mod in sim.model.named_modules():
            if type(mod).__name__ == "QuantizedGraniteMoeParallelExperts":
                ne, out_size, _ = mod.weight.shape
                mod.param_quantizers["weight"] = QuantizeDequantize(
                    shape=(ne, out_size, 1), qmin=-8, qmax=7, symmetric=True
                )
                n_overridden += 1
        print(f"[step 4b] overrode per-(expert, out-channel) weight "
              f"quantizer on {n_overridden} ParallelExperts modules")

    # ---- 5. SEQ_MSE: data loader yields (input_ids, attention_mask) tuples
    # default_forward_fn does `model(*inputs)` so a tuple of tensors is right.
    print(f"\n[step 5] running apply_seq_mse "
          f"(num_candidates={NUM_CANDIDATES}, num_batches={len(cal_pairs)})…")
    from aimet_torch.v2.seq_mse import apply_seq_mse

    class CalibIterable:
        def __init__(self, pairs):
            self.pairs = pairs
        def __iter__(self):
            for p in self.pairs:
                yield p
        def __len__(self):
            return len(self.pairs)

    cal_loader = CalibIterable(cal_pairs)
    t0 = time.time()
    try:
        apply_seq_mse(sim, cal_loader, num_candidates=NUM_CANDIDATES)
    except Exception as e:
        import traceback
        print(f"[step 5] SEQ_MSE FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1
    print(f"[step 5] SEQ_MSE done in {time.time()-t0:.1f}s.")

    # ---- 6. compute_encodings for activations + non-supported (ParallelExperts) weights
    print(f"\n[step 6] running sim.compute_encodings (activations + "
          f"non-SEQ_MSE-supported weights)…")
    t0 = time.time()

    def fwd_fn(m, _):
        with torch.no_grad():
            for ids, mask in cal_pairs:
                m(ids, mask)

    sim.compute_encodings(fwd_fn, None)
    print(f"[step 6] compute_encodings done in {time.time()-t0:.1f}s.")

    # ---- 7. Eval
    with torch.no_grad():
        q_logits = sim.model(probe_ids, probe_mask)
    fp_real = fp_logits[0, : last_real_pos + 1, :].flatten().float()
    q_real = q_logits[0, : last_real_pos + 1, :].flatten().float()
    cos = float(F.cosine_similarity(fp_real, q_real, dim=0))
    argmax_q = q_logits[0, last_real_pos].argmax().item()
    print(f"\n[step 7] quantized probe argmax = {tok.decode(argmax_q)!r}")
    print(f"[step 7] cos(fp32, q4a16+SEQ_MSE) = {cos:.6f}")

    print(f"\n[step 8] saving encodings.json…")
    t0 = time.time()
    try:
        sim.save_encodings_to_json(
            str(OUT_DIR), "granite_1b_a400m_seqmse.encodings"
        )
    except Exception as e:
        import traceback
        print(f"[step 8] save FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
    print(f"[step 8] encodings save done in {time.time()-t0:.1f}s.")

    print(f"\n--- emitted files ---")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size} B)")

    print("\nSQ3 SEQ_MSE probe complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
