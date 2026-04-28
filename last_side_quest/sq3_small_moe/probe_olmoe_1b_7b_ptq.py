"""SQ3 follow-up C — basic PTQ on OLMoE-1B-7B-Instruct (MoE) with AIMET v2.

Tests whether the AIMET adapter pattern from granitemoe generalizes to
a second non-blessed MoE arch with different structural choices:
- OLMoE: 7B / 1B active, 16 layers × 2048 hidden × 64 experts top-8
- Per-expert FFNs are stdlib nn.Linear (vs granitemoe's fused 3D
  ParallelExperts) — adapter surface is smaller (~50 LOC vs ~80)
- BF16 native; loaded as FP32 (~28 GB) fits 48 GB DRAM with headroom

Same recipe as the Granite probe: w4 sym weights / a16 asym acts,
basic PTQ via compute_encodings, 16-prompt × 64-token calibration set
(matches the SEQ_MSE retest). Two key questions:
1. Does the adapter pattern work?
2. Does OLMoE quantize *better* or *worse* than Granite-MoE at the
   same recipe? (Hypothesis from SQ3-Granite: MoE quantizes better
   than dense; this is a second MoE data point at a different scale.)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(REPO / "models" / ".hf_cache"))

OUT_DIR = REPO / "last_side_quest" / "sq3_small_moe" / "out_olmoe_1b_7b_basic_ptq"

MODEL_ID = "allenai/OLMoE-1B-7B-0125-Instruct"

CAL_BREADTH = 16


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
          f"experts/layer={cfg.num_experts}, top_k={cfg.num_experts_per_tok}")

    pre_counts = _count_module_types(model)
    print(f"[step 1] nn.Linear count = {pre_counts.get('Linear', 0)} "
          "(includes 64 experts × 3 linears × 16 layers + attn proj + lm_head)")

    model.config.use_cache = False
    model.generation_config = None
    model = LogitsOnly(model).eval()

    base_prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n-1)",
        "User: Hello!\nAssistant: Hi there, how can I help?",
        "import json\ndata = json.loads('{\"a\": 1, \"b\": [2, 3]}')",
        "Solve for x: 2x + 5 = 17. Subtracting 5 from both sides gives",
        "The capital of France is Paris, which is located in the",
        "function add(a, b) {\n    return a + b;\n}\nconsole.log(add(",
        "Once upon a time in a small village, there lived a young",
        "Q: What is the difference between TCP and UDP? A: TCP is connection-oriented,",
        "Le chat est sur la table. La maison est grande. Bonjour mon",
        "SELECT name, age FROM users WHERE age > 18 ORDER BY name;",
        "# Project README\n\n## Installation\nRun `pip install` to install",
        "Claude is an AI assistant created by Anthropic. It can help with",
        "<thought>Let me think about this step by step.</thought>\n<answer>",
        "The integral of x^2 dx from 0 to 1 equals 1/3. The proof uses",
        "{\"name\": \"Alice\", \"age\": 30, \"hobbies\": [\"reading\", \"chess\"]}",
    ]
    cal_prompts = base_prompts[:CAL_BREADTH]
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
    print(f"[step 3] fp32 probe argmax = {fp_argmax!r}")

    print(f"\n[step 4] building AIMET v2 QuantizationSimModel "
          f"(w4 sym weights / a16 asym acts)…")
    t0 = time.time()
    from aimet_torch.common.defs import QuantScheme
    from aimet_torch.v2.quantsim import QuantizationSimModel
    import olmoe_adapters  # noqa: F401

    sim = QuantizationSimModel(
        model,
        dummy_input=(probe_ids, probe_mask),
        quant_scheme=QuantScheme.post_training_tf_enhanced,
        default_param_bw=4,
        default_output_bw=16,
    )
    print(f"[step 4] sim built in {time.time()-t0:.1f}s.")

    post_counts = _count_module_types(sim.model)
    qlinear = post_counts.get("QuantizedLinear", 0)
    print(f"[step 4] QuantizedLinear modules in sim: {qlinear}")

    # ---- 4b. Disable output quantization on per-expert inner Linears.
    # AIMET v2's compute_encodings stats-mode patching apparently doesn't
    # survive sparse-MoE per-expert dispatch (where each expert's down_proj
    # may be called many times in a single forward pass, not once).
    # Empirically: with output_quantizers in place, compute_encodings
    # raises "QuantizeDequantize not initialized" on the non-empty path
    # of expert FFNs, even though AIMET should have set those to _no_op
    # mode for the duration of stats collection. Workaround: set those
    # output_quantizers to None — per-expert intermediate activations
    # stay FP32, but weights still get w4-quantized and the rest of the
    # model still gets a16. Routing/gate/attention activations also stay
    # at a16 (those are outside `experts.*`).
    n_disabled = 0
    for name, mod in sim.model.named_modules():
        if ".mlp.experts." in name and type(mod).__name__ == "QuantizedLinear":
            mod.output_quantizers = torch.nn.ModuleList([None])
            n_disabled += 1
    print(f"[step 4b] disabled output_quantizers on {n_disabled} per-expert "
          f"inner QuantizedLinears under .mlp.experts.* (FP32 intermediate "
          f"activations inside experts; weights still w4)")

    print(f"\n[step 5] running compute_encodings on calibration set…")
    t0 = time.time()

    def fwd_fn(m, _):
        with torch.no_grad():
            for ids, mask in cal_pairs:
                m(ids, mask)

    sim.compute_encodings(fwd_fn, None)
    print(f"[step 5] compute_encodings done in {time.time()-t0:.1f}s.")

    with torch.no_grad():
        q_logits = sim.model(probe_ids, probe_mask)
    fp_real = fp_logits[0, : last_real_pos + 1, :].flatten().float()
    q_real = q_logits[0, : last_real_pos + 1, :].flatten().float()
    cos = float(F.cosine_similarity(fp_real, q_real, dim=0))
    argmax_q = q_logits[0, last_real_pos].argmax().item()
    print(f"\n[step 6] quantized probe argmax = {tok.decode(argmax_q)!r}")
    print(f"[step 6] cos(fp32, quant) = {cos:.6f}")

    print(f"\n[step 7] saving encodings.json…")
    t0 = time.time()
    try:
        sim.save_encodings_to_json(
            str(OUT_DIR), "olmoe_1b_7b_basic_ptq.encodings"
        )
    except Exception as e:
        import traceback
        print(f"[step 7] save FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
    print(f"[step 7] encodings save done in {time.time()-t0:.1f}s.")

    print(f"\n--- emitted files ---")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size} B)")

    print("\nSQ3 OLMoE PTQ probe complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
