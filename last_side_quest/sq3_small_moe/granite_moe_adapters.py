"""AIMET v2 QuantizationMixin adapters for Granite-MoE.

AIMET 2.29 ships transformer-arch adapters for llama, mistral, phi3,
gemma3, qwen2/2_5/3/3.5/3_vl/3_moe — but **not** granitemoe. When you
call `QuantizationSimModel(GraniteMoeForCausalLM(...))`, AIMET's quantsim
builder enumerates every `nn.Module` subclass in the model and refuses
to proceed if any custom (non-stdlib) class lacks a registered
QuantizationMixin implementation. For Granite-MoE the missing classes
are:

  - GraniteMoeParallelExperts — the fused per-expert FFN (custom layer,
    not nn.Linear; weights shape [num_experts, output_size, input_size])
  - GraniteMoeRMSNorm — Granite's RMSNorm
  - GraniteMoeRotaryEmbedding — Granite's RoPE module

The patterns mirror AIMET's official Qwen3 adapter
(aimet_torch/v2/nn/transformers/models/qwen3/modeling_qwen3.py):
- RoPE is `QuantizationMixin.ignore`'d (cos/sin tables stay FP32 — also
  matches Qualcomm's published Qwen3 path which hoists rotary_emb out
  of the quantized graph entirely; see
  reference_rotary_emb_hoist.md).
- RMSNorm gets a custom adapter with `param_quantizers` declared
  explicitly (to keep the gain weight unquantized — RMSNorm gain is
  scalar-per-channel and high-precision-sensitive).
- ParallelExperts (the granitemoe-specific bit, no Qwen analog) gets
  the standard QuantizationMixin pattern — AIMET will pick a
  per-axis-0 (per-expert) weight quantizer scale for the fused
  [num_experts, out_features, in_features] weight tensor.

Importing this module triggers the @QuantizationMixin.implements
decorators and the .ignore call, registering everything globally with
AIMET.
"""
from __future__ import annotations

import torch

from aimet_torch.v2.nn import QuantizationMixin
from aimet_torch.onnx_utils import map_torch_types_to_onnx

from transformers.models.granitemoe.modeling_granitemoe import (
    GraniteMoeParallelExperts,
    GraniteMoeRMSNorm,
    GraniteMoeRotaryEmbedding,
)


# Tag granitemoe RMSNorm so that any RMSNormalization-specific quantsim
# config maps to it (mirrors AIMET's Qwen3 handling).
map_torch_types_to_onnx[GraniteMoeRMSNorm] = ["RMSNormalization"]

# RoPE outputs (cos, sin) are a tuple — AIMET's output quantizer can't
# wrap tuples, and quantizing position embeddings hurts accuracy.
# Match AIMET's Qwen3 recipe: skip quantsim on RoPE entirely.
QuantizationMixin.ignore(GraniteMoeRotaryEmbedding)


@QuantizationMixin.implements(GraniteMoeRMSNorm)
class QuantizedGraniteMoeRMSNorm(QuantizationMixin, GraniteMoeRMSNorm):
    def __quant_init__(self):
        super().__quant_init__()
        self.input_quantizers = torch.nn.ModuleList([None])
        self.output_quantizers = torch.nn.ModuleList([None])
        self.param_quantizers = torch.nn.ModuleDict({"weight": None})

    def forward(self, hidden_states):
        if self.input_quantizers[0]:
            hidden_states = self.input_quantizers[0](hidden_states)
        with self._patch_quantized_parameters():
            ret = super().forward(hidden_states)
        if self.output_quantizers[0]:
            ret = self.output_quantizers[0](ret)
        return ret


@QuantizationMixin.implements(GraniteMoeParallelExperts)
class QuantizedGraniteMoeParallelExperts(QuantizationMixin, GraniteMoeParallelExperts):
    def __quant_init__(self):
        super().__quant_init__()
        # forward(inputs: Tensor, expert_size: List[int]) → Tensor
        # expert_size is a Python list (see TracerWarning at modeling_granitemoe.py:299
        # "expert_size = expert_size.tolist()") — AIMET can't quantize a list,
        # so only declare 1 input quantizer slot for `inputs`.
        self.input_quantizers = torch.nn.ModuleList([None])
        self.output_quantizers = torch.nn.ModuleList([None])

    def forward(self, inputs, expert_size):
        if self.input_quantizers[0]:
            inputs = self.input_quantizers[0](inputs)
        with self._patch_quantized_parameters():
            ret = super().forward(inputs, expert_size)
        if self.output_quantizers[0]:
            ret = self.output_quantizers[0](ret)
        return ret
