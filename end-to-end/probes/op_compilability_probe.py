"""op_compilability_probe.py — can the gated-delta-net (SSM) op leave PyTorch?

The make-or-break unknown for the Qwen3.6-27B NPU port is whether its
`linear_attention` (gated-delta-net / Mamba2-style) layers can be exported
to ONNX and compiled to the Hexagon HTP at all. Quantization is moot if the
op can't leave PyTorch as a static graph.

Qwen3.6-27B is `model_type: qwen3_5`, which transformers 4.57.6 does NOT yet
ship (needs transformers-from-source). But `qwen3_next` IS available and its
`linear_attention` layers are the SAME gated-delta-net op family — same
`partial_rotary_factor=0.25`, same `head_dim=256`, same linear_* dims schema.
So we use a tiny random `qwen3_next` as a faithful proxy.

Stage 1 (this script): build tiny model -> eager forward sanity -> ONNX export
  (dynamo first, legacy fallback) -> tally ONNX ops, flag the HTP-hostile ones
  (Scan / Loop / If / NonZero / custom domains = data-dependent control flow).
Stage 2 (separate): feed the ONNX to qairt-converter / AI Hub submit_compile_job
  targeting Snapdragon X2 Elite.

Run in .venv-arm-export (torch 2.10, transformers 4.57.6):
  .venv-arm-export/Scripts/python.exe end-to-end/probes/op_compilability_probe.py
"""
from __future__ import annotations

import argparse
import sys
import traceback
from collections import Counter
from pathlib import Path

import torch

# Windows consoles default to cp1252; torch/onnx error strings carry unicode
# (e.g. the cross/check glyphs) that otherwise raise UnicodeEncodeError.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# HTP-hostile ONNX ops: data-dependent control flow / dynamic shape that the
# QNN HTP backend either rejects or cannot tile statically.
HOSTILE_OPS = {"Scan", "Loop", "If", "NonZero", "Where", "CumSum",
               "ScatterND", "GatherND", "TopK", "Range"}


def build_tiny_qwen3_next():
    """Tiny random qwen3_next with mixed linear/full attention layers."""
    from transformers import Qwen3NextConfig, Qwen3NextForCausalLM

    cfg = Qwen3NextConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=16,
        partial_rotary_factor=0.25,        # same as Qwen3.6-27B
        # gated-delta-net (linear_attention) dims — tiny but self-consistent
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        # Force DENSE FFN on every layer (mlp_only_layers) — the real
        # Qwen3.6-27B is dense-FFN, NOT MoE. qwen3_next defaults to a sparse
        # MoE block whose nonzero()-based expert dispatch is data-dependent
        # (export-hostile) and absent from the target, so we opt out of it to
        # keep the proxy's op-set faithful (SSM + attn + dense FFN).
        mlp_only_layers=[0, 1, 2, 3],
        num_experts=4,                  # required >0, but unused given mlp_only_layers
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        max_position_embeddings=512,
        tie_word_embeddings=True,
        # 0,1,2 = gated-delta-net; 3 = full attention (the 1-in-4 pattern)
        layer_types=["linear_attention", "linear_attention",
                     "linear_attention", "full_attention"],
        use_cache=False,
    )
    model = Qwen3NextForCausalLM(cfg).eval()
    return model, cfg


def apply_ssm_export_patches(model) -> list[str]:
    """Option A — make qwen3_next's gated-delta-net export-friendly.

    (1) `_update_linear_attn_mask` -> static None: drops the data-dependent
        `.item()` guard (`cache_position[0] > 0 or torch.all(mask == 1)`) that
        blocks torch.export. None == attend-all, correct for a no-pad prefill.
    (2) force every gated-delta-net to the per-step *recurrent* rule instead
        of the `chunk_size=64` chunked rule (in-place indexed writes + cumsum
        + triu). The two are math-equivalent.
    Returns applied-patch descriptions for the log.
    """
    import transformers.models.qwen3_next.modeling_qwen3_next as M

    def _static_linear_attn_mask(self, attention_mask, cache_position):
        return None

    M.Qwen3NextModel._update_linear_attn_mask = _static_linear_attn_mask
    applied = ["_update_linear_attn_mask -> static None (drop .item())"]
    n = 0
    for mod in model.modules():
        if isinstance(mod, M.Qwen3NextGatedDeltaNet):
            mod.chunk_gated_delta_rule = M.torch_recurrent_gated_delta_rule
            n += 1
    applied.append(f"chunk->recurrent gated-delta-net on {n} layer(s)")
    return applied


def tally_onnx(path: Path) -> Counter:
    import onnx
    m = onnx.load(str(path))
    ops = Counter()
    for n in m.graph.node:
        dom = n.domain or "ai.onnx"
        ops[(dom, n.op_type)] += 1
    return ops


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "out" / "qwen3_next_tiny.onnx")
    ap.add_argument("--seq", type=int, default=8)
    ap.add_argument("--no-patch", action="store_true",
                    help="skip the Option-A SSM export patches")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    print("=" * 70)
    print("STAGE 1 — gated-delta-net ONNX exportability probe (qwen3_next proxy)")
    print("=" * 70)

    # --- build ---
    try:
        model, cfg = build_tiny_qwen3_next()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[build]   OK — tiny qwen3_next, {n_params/1e6:.2f}M params, "
              f"layer_types={cfg.layer_types}")
    except Exception:
        print("[build]   FAIL")
        traceback.print_exc()
        return 2

    # --- Option A: patch the SSM to be export-friendly ---
    if not args.no_patch:
        for a in apply_ssm_export_patches(model):
            print(f"[patch]   {a}")
    else:
        print("[patch]   skipped (--no-patch)")

    # --- eager forward sanity ---
    input_ids = torch.randint(0, cfg.vocab_size, (1, args.seq))
    try:
        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=False)
        print(f"[eager]   OK — logits {tuple(out.logits.shape)}")
    except Exception:
        print("[eager]   FAIL (model can't even run in PyTorch on CPU)")
        traceback.print_exc()
        return 3

    # --- ONNX export: try dynamo (torch.export), then legacy (TorchScript) ---
    exported_by = None
    exc_summary: dict[str, str] = {}
    for mode in ("dynamo", "legacy"):
        try:
            print(f"[export]  trying {mode} exporter ...")
            with torch.no_grad():
                torch.onnx.export(
                    model, (input_ids,), str(args.out),
                    input_names=["input_ids"], output_names=["logits"],
                    opset_version=18, dynamo=(mode == "dynamo"),
                )
            exported_by = mode
            print(f"[export]  OK via {mode} -> {args.out}")
            # Inline external weights into one self-contained file — AI Hub
            # compile + many tools reject a bare .onnx whose weights live in a
            # sibling .data file.
            import onnx as _onnx
            _onnx.save_model(_onnx.load(str(args.out)), str(args.out),
                             save_as_external_data=False)
            print(f"[export]  inlined weights -> self-contained {args.out.name}")
            break
        except Exception as e:
            first = (str(e).strip().splitlines() or [""])[0]
            exc_summary[mode] = f"{type(e).__name__}: {first}"[:200]
            print(f"[export]  {mode} FAILED -> {exc_summary[mode]}")

    if exported_by is None:
        print("\nPROBE RESULT: gated-delta-net does NOT export to ONNX with "
              "stock exporters. Snag 2 is a real op-rewrite (or roll our own "
              "aarch64/HTP op). Failure modes:")
        for mode, summ in exc_summary.items():
            print(f"  - {mode:7s}: {summ}")
        return 4

    # --- tally ops, flag hostiles ---
    ops = tally_onnx(args.out)
    total = sum(ops.values())
    hostile = {k: v for k, v in ops.items() if k[1] in HOSTILE_OPS}
    custom = {k: v for k, v in ops.items() if k[0] not in ("ai.onnx", "")}

    print("\n--- ONNX op inventory ({} nodes, exporter={}) ---".format(total, exported_by))
    for (dom, op), c in sorted(ops.items(), key=lambda x: -x[1]):
        mark = "  <-- HTP-hostile" if op in HOSTILE_OPS else ""
        domtag = "" if dom in ("ai.onnx", "") else f"[{dom}] "
        print(f"  {c:4d}  {domtag}{op}{mark}")

    print("\nPROBE RESULT")
    print(f"  exported_by   : {exported_by}")
    print(f"  total nodes   : {total}")
    print(f"  custom domains: {sorted({k[0] for k in custom}) or 'none'}")
    print(f"  HTP-hostile   : {dict(hostile) or 'none'}")
    if hostile or custom:
        print("  VERDICT: exports, but contains ops the HTP likely rejects/can't "
              "tile -> expect a graph rewrite at snag 2 before compile.")
    else:
        print("  VERDICT: clean static op set -> good odds QAIRT/AI Hub compiles "
              "it. Proceed to stage 2 (qairt-converter / submit_compile_job).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
