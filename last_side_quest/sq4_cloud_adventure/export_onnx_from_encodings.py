"""Load an existing AIMET encodings.json into a fresh sim and export
ONNX with QDQ — without re-running SEQ_MSE / AdaScale / compute_encodings.

Saves ~20 minutes vs a full aimet_quant.py rerun. Intended use:
  - You already have a `qsim.json` (encodings) from a prior run.
  - You want the QDQ ONNX (and a fresh per-tensor encodings.json) to
    feed into qairt-converter.

Uses `aimet_torch.onnx.export` (the v2-recommended path; sim.export()
is deprecated). The exporter traces the sim model with torch.onnx and
emits an ONNX with QDQ ops + a sibling encodings.json.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

# Module-level so torch.onnx tracer / pickle can find it.
import torch  # noqa: E402


class LogitsOnly(torch.nn.Module):
    """Wrap an HF causal-LM so forward returns just the logits tensor."""

    def __init__(self, hf_model):
        super().__init__()
        self.inner = hf_model

    def forward(self, input_ids, attention_mask=None):
        return self.inner(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).logits


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", required=True)
    p.add_argument("--model-path", default=None)
    p.add_argument("--encodings", required=True,
                   help="Path to qsim.json from a prior aimet_quant.py run")
    p.add_argument("--precision", choices=["w4a16", "w8a16", "w8a8"], required=True,
                   help="Must match the precision used in the original run")
    p.add_argument("--ctx", type=int, default=64)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--filename-prefix", default="qsim_export")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = p.parse_args(argv)

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[setup] output_dir = {out_dir}")
    print(f"[setup] encodings  = {args.encodings}")

    import torch

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[setup] device = {device}")

    # ---- Load HF model
    t0 = time.time()
    print(f"\n[1] loading {args.model_id} (FP32)…")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    src = args.model_path or args.model_id
    tok = AutoTokenizer.from_pretrained(src)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    hf_model = AutoModelForCausalLM.from_pretrained(
        src, torch_dtype=torch.float32, attn_implementation="eager",
    ).eval()
    print(f"[1] loaded in {time.time()-t0:.1f}s")
    hf_model.config.use_cache = False
    hf_model.generation_config = None

    model = LogitsOnly(hf_model).to(device).eval()

    # ---- Build dummy input matching the original run's shape
    print(f"\n[2] building dummy input (ctx={args.ctx})…")
    probe = tok("The capital of France is", return_tensors="pt",
                padding="max_length", max_length=args.ctx)
    dummy_input = (probe["input_ids"].to(device),
                   probe["attention_mask"].to(device))

    # ---- Build sim with same precision
    if args.precision == "w4a16":
        param_bw, output_bw = 4, 16
    elif args.precision == "w8a16":
        param_bw, output_bw = 8, 16
    elif args.precision == "w8a8":
        param_bw, output_bw = 8, 8
    else:
        raise ValueError(args.precision)

    print(f"\n[3] building QuantizationSimModel ({args.precision})…")
    t0 = time.time()
    from aimet_torch.common.defs import QuantScheme
    from aimet_torch.v2.quantsim import QuantizationSimModel

    sim = QuantizationSimModel(
        model,
        dummy_input=dummy_input,
        quant_scheme=QuantScheme.post_training_tf_enhanced,
        default_param_bw=param_bw,
        default_output_bw=output_bw,
    )
    print(f"[3] sim built in {time.time()-t0:.1f}s")

    # ---- Load existing encodings
    print(f"\n[4] loading encodings from {args.encodings}…")
    t0 = time.time()
    sim.load_encodings(args.encodings)
    print(f"[4] load_encodings done in {time.time()-t0:.1f}s")

    # ---- Export ONNX with QDQ via aimet_torch.onnx.export
    # (sim.export is deprecated in 2.26+; the new path goes via
    # torch.onnx.export with QDQ insertion handled by aimet's QuantizationMixin)
    print(f"\n[5] exporting ONNX with QDQ (aimet_torch.onnx.export)…")
    t0 = time.time()
    onnx_path = out_dir / f"{args.filename_prefix}.onnx"
    from aimet_torch import onnx as aimet_onnx_export

    aimet_onnx_export.export(
        sim,
        dummy_input,
        str(onnx_path),
        opset_version=21,  # AIMET requires opset>=21 for INT4/INT16 QDQ ops
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
    )
    print(f"[5] export done in {time.time()-t0:.1f}s")
    # Also save encodings.json next to the onnx (for qairt-converter)
    print(f"\n[5b] saving sibling encodings.json…")
    sim.save_encodings_to_json(str(out_dir), f"{args.filename_prefix}_encodings")

    print(f"\n--- emitted files in {out_dir} ---")
    total = 0
    for f in sorted(out_dir.iterdir()):
        sz = f.stat().st_size
        total += sz
        print(f"  {f.name}  ({sz/1e6:.1f} MB)")
    print(f"total: {total/1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
