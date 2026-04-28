"""AIMET v2 QuantizationMixin adapters for OLMoE.

AIMET 2.29's transformer-arch adapter list does NOT include `olmoe`.
Adapters needed:

  - OlmoeRotaryEmbedding — `(cos, sin)` tuple-output; ignore.
  - OlmoeRMSNorm — declare param_quantizers so gain stays FP32.

Plus a transformers-level monkey-patch to OlmoeMLP.forward (NOT a
QuantizationMixin adapter) — see _patch_olmoe_mlp_for_empty_dispatch()
below for why.

Importing this module triggers all three side effects: the two AIMET
class registrations and the OlmoeMLP.forward monkey-patch.
"""
from __future__ import annotations

import torch

from aimet_torch.v2.nn import QuantizationMixin
from aimet_torch.onnx_utils import map_torch_types_to_onnx

from transformers.models.olmoe import modeling_olmoe
from transformers.models.olmoe.modeling_olmoe import (
    OlmoeMLP,
    OlmoeRMSNorm,
    OlmoeRotaryEmbedding,
)


map_torch_types_to_onnx[OlmoeRMSNorm] = ["RMSNormalization"]

QuantizationMixin.ignore(OlmoeRotaryEmbedding)


@QuantizationMixin.implements(OlmoeRMSNorm)
class QuantizedOlmoeRMSNorm(QuantizationMixin, OlmoeRMSNorm):
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


def _patch_olmoe_mlp_for_empty_dispatch():
    """Class-level monkey-patch on `OlmoeMLP.forward` for sparse-MoE PTQ.

    Why a monkey-patch and NOT a `@QuantizationMixin.implements(OlmoeMLP)`
    adapter (initial attempt — committed in olmoe_adapters.py history):

    OLMoE's expert dispatch (modeling_olmoe.py:606
    `expert_layer(current_state) * routing_weights[top_x, idx, None]`)
    can pass an empty `current_state` of shape `[0, hidden]` when no
    tokens route to a given expert for a calibration batch. AIMET v2's
    QuantizedLinear chokes on empty input two ways:
      1. Encoding analyzer crashes on `torch.min()` of an empty tensor
         during stats collection (first-run failure).
      2. After we wrapped OlmoeMLP with QuantizationMixin to early-
         return on empty, the *non-empty* path still hit "QuantizeDequantize
         not initialized" — the QuantizationMixin wrapper apparently
         disrupted AIMET's per-quantizer compute_encodings stats-mode
         patching for inner QuantizedLinears (the `patch_attr(quantizer,
         'forward', _no_op)` mechanism didn't apply, possibly because
         the wrapper class re-bound the lookup).

    A monkey-patch sidesteps both. We modify the **stdlib** OlmoeMLP's
    forward at the class level BEFORE AIMET sim-builds. AIMET then
    sees a regular OlmoeMLP, doesn't add any MLP-level wrapper, and
    only quantizes the inner gate_proj/up_proj/down_proj as standard
    QuantizedLinears. The patched forward early-returns on empty
    input, so inner Linears never see [0, …] tensors and the encoding
    analyzer never sees empty stats. This is granitemoe-via-
    ParallelExperts behavior, retrofitted onto OLMoE's per-expert-MLP
    dispatch design.

    Granite didn't need this because GraniteMoeParallelExperts dispatches
    internally via `index_select` — there is no per-expert call with
    empty input. qwen3_moe presumably has a similar internal-dispatch
    structure (we haven't verified directly).
    """
    _orig_forward = OlmoeMLP.forward

    def _patched_forward(self, x):
        if x.numel() == 0:
            # Mirror the FP32 forward output shape: [..., hidden_size]
            # gate_proj is nn.Linear(hidden_size, intermediate_size),
            # so gate_proj.in_features = hidden_size.
            hidden_size = self.gate_proj.in_features
            return torch.zeros(*x.shape[:-1], hidden_size,
                               dtype=x.dtype, device=x.device)
        return _orig_forward(self, x)

    OlmoeMLP.forward = _patched_forward


_patch_olmoe_mlp_for_empty_dispatch()
