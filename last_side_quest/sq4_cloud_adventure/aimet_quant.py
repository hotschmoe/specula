"""SQ4 — generic AIMET driver: HF model -> w{4,8}a16 quantized ONNX + encodings.

Goal: a model-agnostic AIMET driver we can rerun on Qwen3-0.6B / 4B /
Qwen3.6 / Gemma4 etc. without depending on `qai_hub_models.models.<name>.quantize`
recipes (those only ship for a few "blessed" models).

Stages:
  1. Load HF causal LM + tokenizer (FP32, GPU if available).
  2. Wrap so forward() returns logits tensor (AIMET tracer wants tensors).
  3. Build calibration tensors at a fixed (B, T) shape.
  4. Capture FP32 reference logits on a probe prompt.
  5. Build aimet_torch v2 QuantizationSimModel at the requested precision.
  6. Optional: apply_seq_mse (block-wise weight optimization).
  7. Optional: apply_adascale (activation-aware weight scale tuning).
     Qwen3 is in adascale's native supported_modules list as of 2.26.
  8. compute_encodings over the calibration set (basic PTQ).
  9. Run quantized probe pass; report cos(fp32, q) over real positions.
 10. Export: encodings.json (always); ONNX+QDQ via sim.export() if asked.

This script does NOT do the QAIRT compile — that is a separate stage
(qairt-converter / qairt-quantizer / qnn-context-binary-generator).
The encodings.json + ONNX it emits are the inputs for that stage.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

# Suppress AIMET's noisy deprecation warnings from holoviews/param.
warnings.filterwarnings("ignore", category=FutureWarning)


def _build_logits_only_wrapper():
    """Build the LogitsOnly nn.Module class — torch import deferred."""
    import torch

    class LogitsOnly(torch.nn.Module):
        """Wrap an HF causal-LM so forward returns just the logits tensor."""

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

    return LogitsOnly


def _calibration_prompts():
    """A small, diverse calibration set. Tune via --num-cal-samples (cycles)."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "User: Hello!\nAssistant: Hi there, how can I help?",
        "import json\ndata = json.loads('{\"a\": 1, \"b\": [2, 3]}')",
        "The capital of France is Paris. The capital of Germany is Berlin.",
        "Once upon a time, in a land far, far away, there lived a wise old wizard.",
        "Q: What is 17 times 23?\nA: Let me compute. 17 * 23 = 17 * 20 + 17 * 3 = 340 + 51 = 391.",
        "SELECT user_id, COUNT(*) FROM events WHERE timestamp > '2025-01-01' GROUP BY user_id ORDER BY 2 DESC LIMIT 10;",
        "Thermodynamics: the first law states that energy can neither be created nor destroyed.",
        "<html>\n<body>\n<h1>Hello</h1>\n<p>This is a test.</p>\n</body>\n</html>",
        "A neural network with one hidden layer can approximate any continuous function.",
        "git commit -m 'fix: handle null pointer in cache lookup' && git push origin main",
        "The Pythagorean theorem: a^2 + b^2 = c^2 for right triangles.",
        "She sells seashells by the seashore. The shells she sells are seashells, I'm sure.",
        "[INST] Summarize the following passage in one sentence. [/INST]",
        "Implementing speculative decoding requires a draft model and a target model.",
    ]


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description="Generic AIMET driver: HF -> quantized ONNX + encodings."
    )
    p.add_argument("--model-id", required=True,
                   help="HF model id, e.g. Qwen/Qwen3-0.6B")
    p.add_argument("--model-path", default=None,
                   help="Local path to HF snapshot (overrides --model-id download)")
    p.add_argument("--precision", choices=["w4a16", "w8a16", "w8a8"], default="w4a16")
    p.add_argument("--ctx", type=int, default=64,
                   help="Sequence length for AIMET tracing + calibration. "
                        "Keep small — calibration cost scales with this.")
    p.add_argument("--num-cal-samples", type=int, default=32,
                   help="How many calibration samples to feed compute_encodings. "
                        "Cycled through the built-in prompt list.")
    p.add_argument("--use-seq-mse", action="store_true")
    p.add_argument("--use-ada-scale", action="store_true")
    p.add_argument("--seq-mse-num-batches", type=int, default=4)
    p.add_argument("--ada-scale-iters", type=int, default=200)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--export-onnx", action="store_true",
                   help="Also call sim.export() to write ONNX with QDQ. "
                        "Slow + large; encodings.json alone is enough for QAIRT.")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--probe-prompt", default="The capital of France is")
    args = p.parse_args(argv)

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Manifest dict — written at the end whether we succeed or fail mid-way.
    manifest: dict = {
        "model_id": args.model_id,
        "model_path": args.model_path,
        "precision": args.precision,
        "ctx": args.ctx,
        "num_cal_samples": args.num_cal_samples,
        "use_seq_mse": args.use_seq_mse,
        "use_ada_scale": args.use_ada_scale,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stages": {},
    }

    def stamp(stage: str, **fields):
        manifest["stages"][stage] = {"t": time.strftime("%H:%M:%S"), **fields}
        with open(out_dir / "manifest.json", "w") as fh:
            json.dump(manifest, fh, indent=2, default=str)

    print(f"[setup] output_dir = {out_dir}")
    stamp("start")

    # ---- Imports (torch first, then aimet — order matters for some warnings)
    import torch
    import torch.nn.functional as F

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[setup] device = {device}, "
          f"{'gpu=' + torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}")
    manifest["device"] = str(device)
    if device.type == "cuda":
        manifest["gpu"] = torch.cuda.get_device_name(0)
        manifest["gpu_vram_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 2)

    # ---- 1. Load HF model
    print(f"\n[1] loading {args.model_id}{' (local: '+args.model_path+')' if args.model_path else ''} (FP32)…")
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    src = args.model_path or args.model_id
    tok = AutoTokenizer.from_pretrained(src)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    hf_model = AutoModelForCausalLM.from_pretrained(
        src,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    cfg = hf_model.config
    print(f"[1] loaded in {time.time()-t0:.1f}s. arch={cfg.architectures}, "
          f"layers={cfg.num_hidden_layers}, hidden={cfg.hidden_size}, "
          f"heads={cfg.num_attention_heads}, vocab={cfg.vocab_size}")
    manifest["model_arch"] = list(cfg.architectures or [])
    manifest["model_layers"] = cfg.num_hidden_layers
    manifest["model_hidden"] = cfg.hidden_size
    manifest["load_seconds"] = round(time.time() - t0, 2)
    stamp("load_hf", seconds=round(time.time() - t0, 2))

    hf_model.config.use_cache = False
    hf_model.generation_config = None
    LogitsOnly = _build_logits_only_wrapper()
    model = LogitsOnly(hf_model).to(device).eval()

    # ---- 2. Calibration tensors (kept on CPU; forward_fn moves them to
    # whatever device the model is currently on. This matters for AdaScale,
    # which moves the entire model to CPU at the start and shuttles blocks
    # to GPU one at a time — pre-loading inputs on cuda would mis-match.)
    print(f"\n[2] building calibration set (ctx={args.ctx}, n={args.num_cal_samples})…")
    base_prompts = _calibration_prompts()
    cal_pairs = []
    for i in range(args.num_cal_samples):
        prompt = base_prompts[i % len(base_prompts)]
        enc = tok(prompt, return_tensors="pt", truncation=True,
                  padding="max_length", max_length=args.ctx)
        cal_pairs.append((enc["input_ids"], enc["attention_mask"]))  # CPU
    print(f"[2] cal_pairs: {len(cal_pairs)} @ shape "
          f"{tuple(cal_pairs[0][0].shape)} (kept on CPU)")
    stamp("calibration_built", n=len(cal_pairs), shape=list(cal_pairs[0][0].shape))

    # ---- 3. FP32 reference on probe (probe tensors kept on CPU; moved
    # inside forward calls so they always land where the model is)
    print(f"\n[3] FP32 reference probe: {args.probe_prompt!r}")
    probe = tok(args.probe_prompt, return_tensors="pt",
                padding="max_length", max_length=args.ctx)
    probe_ids_cpu = probe["input_ids"]
    probe_mask_cpu = probe["attention_mask"]
    probe_ids = probe_ids_cpu.to(device)
    probe_mask = probe_mask_cpu.to(device)
    last_real_pos = int(probe_mask[0].sum().item()) - 1
    with torch.no_grad():
        fp_logits = model(probe_ids, probe_mask)
    fp_argmax = tok.decode(fp_logits[0, last_real_pos].argmax().item())
    print(f"[3] fp32 last-pos argmax: {fp_argmax!r}")
    manifest["fp32_argmax"] = fp_argmax
    stamp("fp32_probe", argmax=fp_argmax)

    # ---- 4. QuantizationSimModel
    print(f"\n[4] building QuantizationSimModel (precision={args.precision})…")
    t0 = time.time()
    from aimet_torch.common.defs import QuantScheme
    from aimet_torch.v2.quantsim import QuantizationSimModel

    if args.precision == "w4a16":
        param_bw, output_bw = 4, 16
    elif args.precision == "w8a16":
        param_bw, output_bw = 8, 16
    elif args.precision == "w8a8":
        param_bw, output_bw = 8, 8
    else:
        raise ValueError(args.precision)

    sim = QuantizationSimModel(
        model,
        dummy_input=(probe_ids, probe_mask),
        quant_scheme=QuantScheme.post_training_tf_enhanced,
        default_param_bw=param_bw,
        default_output_bw=output_bw,
    )
    print(f"[4] sim built in {time.time()-t0:.1f}s "
          f"(param_bw={param_bw}, output_bw={output_bw}).")
    stamp("sim_built", seconds=round(time.time() - t0, 2),
          param_bw=param_bw, output_bw=output_bw)

    def _model_device(m):
        # Find current device of the wrapped model. AdaScale moves entire
        # model to CPU during apply_adascale, so this is dynamic.
        try:
            return next(m.parameters()).device
        except StopIteration:
            return device

    # Per-batch forward function used by SEQ_MSE / AdaScale.
    # Both call this as fn(model, batch) where batch is one item from data_loader.
    def per_batch_fwd(m, batch):
        ids, mask = batch
        target = _model_device(m)
        with torch.no_grad():
            return m(ids.to(target), mask.to(target))

    # Whole-set forward used by compute_encodings (AIMET passes a single arg).
    def cal_forward(m, _=None):
        target = _model_device(m)
        with torch.no_grad():
            for ids, mask in cal_pairs:
                m(ids.to(target), mask.to(target))

    # ---- 5. SEQ_MSE (optional)
    if args.use_seq_mse:
        print(f"\n[5] apply_seq_mse (num_batches={args.seq_mse_num_batches})…")
        t0 = time.time()
        from aimet_torch.v2.seq_mse import apply_seq_mse

        cal_ds = list(cal_pairs[: args.seq_mse_num_batches])
        try:
            # New 2.26+ signature: (sim, data_loader, num_candidates, forward_fn, ...)
            apply_seq_mse(
                sim,
                cal_ds,
                forward_fn=per_batch_fwd,
            )
            print(f"[5] seq_mse done in {time.time()-t0:.1f}s.")
            stamp("seq_mse", seconds=round(time.time() - t0, 2),
                  num_batches=len(cal_ds))
        except Exception as e:
            import traceback
            print(f"[5] SEQ_MSE FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            stamp("seq_mse_failed", error=f"{type(e).__name__}: {e}")
            print("[5] continuing without SEQ_MSE")

        # Defensive: move model back to GPU if SEQ_MSE moved anything to CPU.
        sim.model.to(device)

    # ---- 6. AdaScale (optional)
    if args.use_ada_scale:
        print(f"\n[6] apply_adascale (iters={args.ada_scale_iters})…")
        t0 = time.time()
        from aimet_torch.experimental.adascale.adascale_optimizer import apply_adascale

        try:
            apply_adascale(
                qsim=sim,
                data_loader=cal_pairs,
                forward_fn=per_batch_fwd,
                num_iterations=args.ada_scale_iters,
            )
            print(f"[6] adascale done in {time.time()-t0:.1f}s.")
            stamp("ada_scale", seconds=round(time.time() - t0, 2),
                  iters=args.ada_scale_iters)
        except Exception as e:
            import traceback
            print(f"[6] AdaScale FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            stamp("ada_scale_failed", error=f"{type(e).__name__}: {e}")

        # Defensive: re-pin to device before compute_encodings.
        sim.model.to(device)

    # ---- 7. compute_encodings (basic PTQ over cal set)
    print(f"\n[7] compute_encodings…")
    sim.model.to(device)  # defensive re-pin
    t0 = time.time()
    sim.compute_encodings(cal_forward, None)
    print(f"[7] compute_encodings done in {time.time()-t0:.1f}s.")
    stamp("compute_encodings", seconds=round(time.time() - t0, 2))

    # ---- 8. Quantized probe pass + cos
    print(f"\n[8] quantized probe pass…")
    with torch.no_grad():
        q_logits = sim.model(probe_ids, probe_mask)
    fp_real = fp_logits[0, : last_real_pos + 1, :].flatten().float()
    q_real = q_logits[0, : last_real_pos + 1, :].flatten().float()
    cos = float(F.cosine_similarity(fp_real, q_real, dim=0))
    q_argmax = tok.decode(q_logits[0, last_real_pos].argmax().item())
    print(f"[8] q last-pos argmax: {q_argmax!r}  "
          f"(fp32 was {fp_argmax!r})")
    print(f"[8] cos(fp32, q) over real positions = {cos:.6f}")
    manifest["q_argmax"] = q_argmax
    manifest["cos_fp32_vs_q"] = round(cos, 6)
    manifest["argmax_match"] = (q_argmax == fp_argmax)
    stamp("probe", cos=round(cos, 6), q_argmax=q_argmax,
          argmax_match=q_argmax == fp_argmax)

    # ---- 9. Save encodings.json (always) + optional ONNX export
    print(f"\n[9] saving encodings…")
    t0 = time.time()
    enc_stem = "qsim"
    try:
        sim.save_encodings_to_json(str(out_dir), enc_stem)
        print(f"[9] save_encodings_to_json -> {out_dir/(enc_stem+'.encodings.json')}")
        stamp("save_encodings", seconds=round(time.time() - t0, 2))
    except Exception as e:
        import traceback
        print(f"[9] save_encodings_to_json FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        stamp("save_encodings_failed", error=f"{type(e).__name__}: {e}")

    if args.export_onnx:
        print(f"\n[9b] sim.export() → ONNX with QDQ (this is slow + large)…")
        t0 = time.time()
        try:
            sim.export(
                path=str(out_dir),
                filename_prefix="qsim_model",
                dummy_input=(probe_ids.cpu(), probe_mask.cpu()),
            )
            print(f"[9b] sim.export done in {time.time()-t0:.1f}s.")
            stamp("export_onnx", seconds=round(time.time() - t0, 2))
        except Exception as e:
            import traceback
            print(f"[9b] sim.export FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            stamp("export_onnx_failed", error=f"{type(e).__name__}: {e}")

    manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    stamp("done")
    print(f"\n--- emitted files in {out_dir} ---")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size} B)")
    print("\nAIMET quant run complete.")
    print(f"  cos(fp32,q) = {cos:.6f}    argmax_match = {q_argmax == fp_argmax}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
