"""Custom exporter for transformers models that hit the 2GiB protobuf
limit in optimum-cli's legacy torch.onnx.export path.

Used by stage_optimum_export when the model is too large for the legacy
exporter (Qwen3-4B and bigger). Mimics optimum's `text-generation-with-past`
output: a single `model.onnx` + external `model.onnx_data`, with the
same input/output naming so the downstream rewrite_qwen3_*.py scripts
work unchanged.

Strategy: drive `optimum.exporters.onnx.OnnxConfig` to produce the same
dummy inputs and dynamic-axis spec optimum would, then call
`torch.onnx.export` directly with `use_external_data_format=True`. This
skips the failing `_jit_pass_onnx_graph_shape_type_inference` path by
turning shape inference off at export time.

CLI:
  python export_dynamo.py --model-path /workspace/models/Qwen3-4B \
                          --out-dir /workspace/runs/.../01_optimum
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _build_dummy_inputs(model, batch=1, seq=2, past_seq=4):
    """Mimic optimum's `text-generation-with-past` dummies for a Qwen3-style
    causal LM: input_ids [B, T], attention_mask [B, T+past], past_key_values
    list of (key, value) tuples, position_ids [B, T]."""
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    input_ids = torch.zeros((batch, seq), dtype=torch.long, device=device)
    attention_mask = torch.ones((batch, seq + past_seq), dtype=torch.long, device=device)
    position_ids = torch.arange(past_seq, past_seq + seq, dtype=torch.long, device=device).unsqueeze(0).expand(batch, -1).contiguous()

    past_key_values = []
    for _ in range(n_layers):
        k = torch.zeros((batch, n_kv, past_seq, head_dim), dtype=dtype, device=device)
        v = torch.zeros((batch, n_kv, past_seq, head_dim), dtype=dtype, device=device)
        past_key_values.append((k, v))

    return input_ids, attention_mask, position_ids, past_key_values


def _names_and_axes(n_layers):
    input_names = ["input_ids", "attention_mask", "position_ids"]
    output_names = ["logits"]
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "past_sequence_length + 1"},
        "position_ids": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"},
    }
    for i in range(n_layers):
        input_names += [f"past_key_values.{i}.key", f"past_key_values.{i}.value"]
        output_names += [f"present.{i}.key", f"present.{i}.value"]
        dynamic_axes[f"past_key_values.{i}.key"] = {0: "batch_size", 2: "past_sequence_length"}
        dynamic_axes[f"past_key_values.{i}.value"] = {0: "batch_size", 2: "past_sequence_length"}
        dynamic_axes[f"present.{i}.key"] = {0: "batch_size", 2: "past_sequence_length + 1"}
        dynamic_axes[f"present.{i}.value"] = {0: "batch_size", 2: "past_sequence_length + 1"}
    return input_names, output_names, dynamic_axes


class _Wrapper(torch.nn.Module):
    """Forwards (input_ids, attention_mask, position_ids, *flat_past_kv)
    through the model.  Keeps the call signature flat so torch.onnx.export
    can name each tensor positionally."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, position_ids, past_kv_tuple):
        """past_kv_tuple is a flat tuple of 2*n_layers tensors: k0, v0, k1, v1, ..."""
        from transformers.cache_utils import DynamicCache
        n_layers = len(past_kv_tuple) // 2
        cache = DynamicCache()
        for i in range(n_layers):
            cache.update(past_kv_tuple[2 * i], past_kv_tuple[2 * i + 1], i)
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        flat_present = []
        for layer_kv in out.past_key_values:
            flat_present.append(layer_kv[0]); flat_present.append(layer_kv[1])
        return (out.logits, *flat_present)


def _patch_transformers_masking():
    """Replace the vmap-based causal mask construction with a direct
    broadcast-based one that's friendly to torch.export / torch.onnx.

    The default `_vmap_for_bhqkv(causal_mask_function)` builds a
    `kv_idx <= q_idx` triangle via vmap, which torch.export rejects
    ('functionalization on closure not supported'). The replacement
    below produces an equivalent boolean (B, 1, Q, K) mask via plain
    broadcasting — no vmap, fully traceable.
    """
    import transformers.masking_utils as mu
    if getattr(mu, "_specula_patched", False):
        return

    def _direct_causal_mask(mask_function, bh_indices=True):
        # Returns a function callable as
        #   fn(batch_arange, head_arange, cache_position, kv_arange)
        # producing a (B, H, Q, K) bool tensor for the causal mask
        # where mask_function(b, h, q, k) is *assumed to be*
        # `kv_idx <= q_idx` (the standard causal mask). We don't try
        # to support arbitrary mask_functions — just the common path.
        def fn(batch_arange, head_arange, cache_position, kv_arange):
            # cache_position: (Q,)  kv_arange: (K,)
            q = cache_position.unsqueeze(-1)  # (Q, 1)
            k = kv_arange.unsqueeze(0)        # (1, K)
            mask = (k <= q)                   # (Q, K)
            if bh_indices:
                B = batch_arange.shape[0]
                H = head_arange.shape[0]
                mask = mask.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)
            return mask
        return fn

    mu._original_vmap_for_bhqkv = mu._vmap_for_bhqkv
    mu._vmap_for_bhqkv = _direct_causal_mask
    mu._specula_patched = True
    print("[export-dynamo] patched transformers.masking_utils._vmap_for_bhqkv "
          "(direct broadcast, no vmap)")


def export(model_path: Path, out_dir: Path, dtype: str = "fp32",
           opset: int = 18) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[export-dynamo] model: {model_path} → {out_dir}")
    print(f"[export-dynamo] dtype: {dtype}, opset: {opset}")
    _patch_transformers_masking()

    cfg = AutoConfig.from_pretrained(str(model_path))
    n_layers = cfg.num_hidden_layers

    torch_dtype = {"fp32": torch.float32, "fp16": torch.float16,
                   "bf16": torch.bfloat16}[dtype]

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), dtype=torch_dtype, attn_implementation="eager",
    )
    model.eval()
    print(f"[export-dynamo] model loaded ({time.time() - t0:.1f}s); "
          f"{sum(p.numel() for p in model.parameters()) / 1e9:.2f} B params")

    wrapped = _Wrapper(model)

    input_ids, attn_mask, pos_ids, past_kv = _build_dummy_inputs(model)
    flat_past = tuple(t for kv in past_kv for t in kv)
    args = (input_ids, attn_mask, pos_ids, flat_past)

    input_names, output_names, dynamic_axes = _names_and_axes(n_layers)
    onnx_path = out_dir / "model.onnx"

    print(f"[export-dynamo] torch.onnx.export(dynamo=True) → {onnx_path}")
    t0 = time.time()
    with torch.no_grad():
        from torch.export import Dim
        batch = Dim("batch", min=1, max=64)
        seq = Dim("seq", min=1, max=8192)
        kv_len = Dim("kv_len", min=1, max=16382)   # past_seq + seq
        past_seq = Dim("past_seq", min=0, max=8191)
        # Match args structure: (input_ids, attn_mask, pos_ids, tuple_of_72_kv)
        # Inner tuple has 72 elements (36 layers × 2 each).
        kv_dyn = tuple({0: batch, 2: past_seq} for _ in range(2 * n_layers))
        dyn_shapes = (
            {0: batch, 1: seq},                         # input_ids
            {0: batch, 1: kv_len},                      # attention_mask
            {0: batch, 1: seq},                         # position_ids
            kv_dyn,                                     # the 72-tensor flat tuple
        )
        torch.onnx.export(
            wrapped, args, str(onnx_path),
            input_names=input_names, output_names=output_names,
            opset_version=opset,
            dynamo=True,
            external_data=True,
            dynamic_shapes=dyn_shapes,
            verbose=False,
        )
    print(f"[export-dynamo] export done ({time.time() - t0:.1f}s)")

    # Save tokenizer + configs alongside (mirrors optimum's output layout).
    tok = AutoTokenizer.from_pretrained(str(model_path))
    tok.save_pretrained(str(out_dir))
    cfg.save_pretrained(str(out_dir))

    # List output files.
    print(f"[export-dynamo] output files:")
    for f in sorted(out_dir.iterdir()):
        size_mb = f.stat().st_size / 1e6 if f.is_file() else 0
        print(f"  {f.name:40s} {size_mb:10.1f} MB")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    p.add_argument("--opset", type=int, default=18)
    args = p.parse_args()
    export(args.model_path, args.out_dir, dtype=args.dtype, opset=args.opset)
    return 0


if __name__ == "__main__":
    sys.exit(main())
