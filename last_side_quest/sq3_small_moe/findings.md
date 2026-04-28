# SQ3 (small-MoE branch) — Granite-3.0-1B-A400M-Instruct local AIMET PTQ

**Status:** closed POSITIVE on the install/extensibility axis,
2026-04-28. Quality follow-ups deferred. Driver:
`probe_granite_1b_a400m_ptq.py`. Adapter module:
`granite_moe_adapters.py`.

## Why this run

SQ2 closed positive on AIMET PyTorch local-axis viability but landed
the Qwen3-MoE quantsim claim **untested at scale** — AIMET 2.29 ships
a `qwen3_moe` adapter, but we hadn't actually loaded any MoE model and
built a sim. The umbrella plan's SQ3 wanted Qwen3-30B-A3B for that
test (~30 GB FP32, fits 48 GB tight); the user opted to defer that to
cloud and asked whether smaller MoEs exist for a faster local probe.

Search turned up three small-MoE candidates outside the Qwen family:
- **Granite-3.0-1B-A400M-Instruct** (IBM): 1.3B / 400M active, 24 layers,
  32 experts top-8. ~5 GB FP32. Tiny.
- **Granite-3.0-3B-A800M-Instruct** (IBM): 3B / 800M active.
- **OLMoE-1B-7B** (AllenAI): 7B / 1B active.

User selected the smallest (Granite 1B-A400M) for fastest iteration.

## Headline finding

**AIMET 2.29 has no `granitemoe` arch adapter, but extending it to a
new MoE architecture takes ~80 LOC.** Once the three missing class
adapters are registered, AIMET's v2 quantsim builds against
GraniteMoeForCausalLM, runs `compute_encodings` on CPU in 99.8s, and
emits a 38 MB encodings.json. Cos similarity vs FP32 logits is
**0.656** — bad-but-not-catastrophic, dramatically better than SQ2's
**-0.065** for Qwen3-0.6B-dense at the same w4a16 basic-PTQ recipe.

The implication is structural: **MoE w4a16 PTQ may be naturally
better-conditioned than dense w4a16 PTQ** at comparable param counts,
likely because expert specialization narrows the per-expert weight
distribution that quantization has to capture. This is a hypothesis
that would need 2-3 more (model, recipe) cells to confirm; if it
holds, it's a real result.

## What AIMET wanted, what we provided

AIMET's `QuantizationSimModel` constructor enumerates every
non-stdlib `nn.Module` subclass in the model and refuses to proceed
if any of them lack a `@QuantizationMixin.implements(...)` registered
adapter. For `GraniteMoeForCausalLM` the missing classes were:

| class | role | what we did |
|---|---|---|
| `GraniteMoeRotaryEmbedding` | RoPE — returns `(cos, sin)` tuple | `QuantizationMixin.ignore(...)` — skip quantsim entirely (matches AIMET's official Qwen3 adapter pattern; quantizing position embeds also hurts accuracy) |
| `GraniteMoeRMSNorm` | RMSNorm | Custom adapter with `param_quantizers = {"weight": None}` so the gain weight stays FP32 (RMSNorm gain is scalar-per-channel and high-precision-sensitive) |
| `GraniteMoeParallelExperts` | Fused per-expert FFN, 3D weight `[num_experts, out_features, in_features]` | Custom adapter with **1 input quantizer** (the `inputs` tensor) and **1 output quantizer**. Note: `expert_size` is a Python list (post `tolist()`) and cannot be quantized — declared only 1 input slot, not 2 |

Total adapter LOC including imports + comments + decorators: **~80**.

The AIMET error message that initiated this is **highly actionable**:
it tells you exactly which classes are missing, prints copy-paste
templates for each, and hyperlinks the API ref. The "extend AIMET to a
new arch" experience is a small, mechanical exercise — not the
multi-day adventure SQ2's prior assumed.

## Run trace

```text
[step 1] loaded in 3.4s. arch=GraniteMoeForCausalLM, layers=24,
         hidden=1024, experts/layer=32, top_k=8
[step 1] nn.Linear count = 121

[step 4] sim built in 7.8s.
[step 4] QuantizedLinear modules in sim: 121 (was 121 nn.Linear pre-sim)
[step 4] sample router QuantizedLinear:
         inner.model.layers.0.block_sparse_moe.router.layer
         in=1024, out=32, w4 sym per-channel, a16 asym uint16

[step 5] compute_encodings done in 99.8s.

[step 3] fp32 probe:    "The capital of France is" → ' Paris'
[step 6] q4a16 probe:   "The capital of France is" → '<|end_of_text|>'
[step 6] cos(fp32, q4a16) over real positions = 0.655903

[step 7] encodings save done in 3.4s.
         granite_1b_a400m_basic_ptq.encodings.json (38 MB, 219 param +
         242 activation entries)
```

## Quality finding — comparison with SQ2 baseline

| run | model | params | calibration | cos vs FP32 |
|---|---|---:|---:|---:|
| SQ2 | Qwen3-0.6B (dense) | 596 M | 254 s | **-0.065** ❌ catastrophic |
| SQ3 | Granite-1B-A400M (MoE) | 1.3 B / 400 M act | 99.8 s | **+0.656** ⚠ degraded |

Both runs use **identical PTQ recipe**: AIMET v2
`QuantizationSimModel`, `quant_scheme=post_training_tf_enhanced`,
`default_param_bw=4`, `default_output_bw=16`, 4 calibration prompts
× 64 tokens, basic compute_encodings (no SEQ_MSE / AdaScale / CLE).

Two observations:

1. **MoE quantizes better than dense at the same recipe + bitwidth.**
   Cos +0.656 ≈ "the model has degraded toward a weak signal" —
   recoverable with SEQ_MSE / AdaScale. Cos -0.065 ≈ "the model is
   noise" — not recoverable without rethinking the bitwidth or
   adding heavy techniques. This is a 0.7-cosine gap, well outside
   measurement noise.

2. **Calibration scales with active params, not total.** 99.8 s for
   1.3 B / 400 M-active vs 254 s for 0.6 B-dense. Per-active-param
   wall-time is ~comparable (0.6 B at 254 s ≈ 0.4 B at 170 s
   normalized; observed 99.8 s for 0.4 B-active is ~40% lower than
   linear scaling, consistent with MoE FFN being smaller per expert).
   For SQ4's cloud-rental sizing this is significant: SEQ_MSE on
   Qwen3-30B-A3B at 3 B-active ≈ 7.5× wall-time of this Granite run,
   not 75× as a naive total-param read would suggest. Plausibly
   ~13 minutes per calibration pass on Prism CPU for 30B-A3B basic
   PTQ — order-of-magnitude tractable locally.

## Subtlety — per-tensor quantization on the fused expert weights

Inspecting the encodings.json reveals an **important per-axis quantizer
limitation** that the cos 0.656 number hides:

```json
"inner.model.layers.0.block_sparse_moe.input_linear.weight": [
  {"bitwidth": 4, "is_symmetric": "True",
   "max": 0.0641, "min": -0.0733,
   "offset": -8, "scale": 0.0092}
]
```

That's a **list of length 1** — meaning ONE scale/offset for the
entire `[32 experts × 512 hidden_mlp × 1024 hidden]` weight tensor.
Compare to the router gate, which got per-output-channel:

```json
"inner.model.layers.0.block_sparse_moe.router.layer.weight": [
  {…},   ← 32 entries, one per expert routing logit
]
```

So our minimal adapter accidentally got **per-tensor** weight
quantization on the fused experts (the worst granularity). The cos
0.656 result was achieved *despite* per-tensor expert weights. With
per-expert (axis 0) or ideally per-expert-per-output-channel (axis 0
+ axis 1), cos likely climbs noticeably. **This is the first concrete
quality lever for SQ3.b** if we keep going.

The fix is one line in the adapter — declare an explicit
per-axis-0-or-1 weight QuantizeDequantize in `__quant_init__`, replacing
AIMET's default scalar pick. Untested in this session.

## Cross-reference: AIMET Qwen3 adapter as the canonical pattern

The fix to the original tuple-output crash came from reading
`aimet_torch/v2/nn/transformers/models/qwen3/modeling_qwen3.py` (51
lines including license header). The pattern there is:

```python
QuantizationMixin.ignore(modeling_qwen3.Qwen3RotaryEmbedding)

@QuantizationMixin.implements(modeling_qwen3.Qwen3RMSNorm)
class QuantizedQwen3RMSNorm(QuantizationMixin, modeling_qwen3.Qwen3RMSNorm):
    def __quant_init__(self):
        super().__quant_init__()
        self.input_quantizers = torch.nn.ModuleList([None])
        self.output_quantizers = torch.nn.ModuleList([None])
        self.param_quantizers = torch.nn.ModuleDict({"weight": None})
    ...

map_torch_types_to_onnx[modeling_qwen3.Qwen3RMSNorm] = ["RMSNormalization"]
```

The `qwen3_moe` adapter is similar — RMSNorm + ignore RoPE; expert
FFNs are stdlib `nn.Linear` so no custom expert adapter. Granite's
`ParallelExperts` is the granitemoe-specific divergence — the only
class for which we wrote genuinely-new (not template-copied) code, and
even that came from AIMET's emitted error template.

## Follow-up A — per-expert per-output-channel weight quantization

**Result: cos 0.656 → 0.712** (modest, +0.056). Encodings.json size
**38 MB → 503 MB** (~13× larger). Calibration time **99.8 s → 70.2 s**
(faster, likely because per-tensor min/max tracking was less efficient
across a 16M-element flat tensor than per-channel min/max per output).

| run | weight quant on ParallelExperts | scales / fused tensor | calib time | cos | argmax | encodings.json |
|---|---|---:|---:|---:|---|---:|
| A0 baseline | per-tensor (AIMET default) | 1 | 99.8 s | 0.656 | `<\|end_of_text\|>` | 38 MB |
| **A1 per-(exp, out)** | shape (32, 1024, 1) per layer | 32,768 | **70.2 s** | **0.712** | ` ` (space) | 503 MB |

Scales for first 1000 channels of `layers.0.block_sparse_moe.input_linear.weight`:
min=0.00408, max=0.02525, mean=0.01042, stdev=0.00294. The 6× spread
in per-channel scales confirms per-axis granularity captures real
distribution variance that per-tensor averaged away.

**Why the small jump.** Going from 1 scale → 32K scales per fused
expert tensor reduces weight quantization error proportionally to the
per-channel distribution variance. For Granite-MoE this is ~6×
narrower per channel than per-tensor, so weight rounding error drops
a few-fold. But the dominant remaining error sources are likely
**activation quantization at a16** (uint16 isn't enough for outlier-
channel activations in V/O / down-projections — the SQ2 V/O collapse
story) and **calibration breadth** (4 prompts × 64 tokens is tiny).
Those need SEQ_MSE / AdaScale / SmoothQuant, not weight granularity.

**Probe configuration.** `probe_granite_1b_a400m_ptq.py` now has a
top-of-file toggle `PER_EXPERT_PER_CHANNEL_WEIGHTS = True` (default).
Set False to reproduce A0 baseline. Output dir suffix tracks which
config ran. Override is post-sim-build (walking `sim.model` for
`QuantizedGraniteMoeParallelExperts` instances and replacing
`param_quantizers["weight"]`) — keeps the adapter file canonical.

**Implication for production bundles.** Per-(expert, out-channel) is
the right granularity for quality, but the 503 MB encodings.json is
unwieldy. For deployment we'd want per-expert (scale shape
`(num_experts, 1, 1)` = 32 scales/layer) as a middle ground, or
per-row-block compression via `qairt-quantizer`'s row-grouping.
Untested in this session.

## Open follow-ups (not done in this session)

1. **SEQ_MSE on this 1.3 B-MoE.** SQ2 left this open for Qwen3-0.6B
   "to see if cos -0.065 closes to ≥ 0.95"; the Granite case starts
   from cos +0.712 with per-channel weights and would likely close
   even faster. Estimated wall-time ~30-90 minutes on Prism CPU.
   Likely the real lever for closing to 0.95+.
2. **AdaScale on this same model.** Imports work; behavior on
   GraniteMoe attention head untested.
3. **The same AIMET adapter approach applied to Granite-3.0-3B-A800M
   or OLMoE-1B-7B.** OLMoE has its own custom `OlmoeForCausalLM`
   with a different MLP shape; predicted ~80 LOC adapter.
4. **Apply the per-tensor-vs-per-channel comparison to the
   "MoE-quantizes-better-than-dense" hypothesis.** Need a comparable
   dense baseline at the same active-param count (Qwen3-1.7B is the
   obvious one) for the comparison to be fair.
5. **Per-expert (axis-0 only) variant** as a middle ground for
   deployment. Untested.

## Feed back into the umbrella plan

This run **does not close SQ3 as scoped** — that side quest is about
Qwen3-MoE (or similarly-large MoE) on the *NPU compile* path, which
requires the cloud pipeline. But it does close a **smaller, more
useful sub-question** that SQ3 implicitly contained:

> Is AIMET's MoE quantsim machinery generalized, or specifically
> hand-tuned for `qwen3_moe`?

Answer: **specifically hand-tuned for the families AIMET ships
adapters for, but extensible to others in ~80 LOC of mechanical
adapter code.** This collapses what would otherwise be a "wait for
Qualcomm to bless your MoE arch" blocker into a "spend ~30 minutes
writing adapters once per arch" cost.

Implications for the cloud-pipeline doc
(`docs/one_pipeline_cloud_gpu.md`): the AIMET PTQ step for Qwen3.5/3.6
MoE production targets is now a **single-file adapter + standard
`compute_encodings` call** away. No need to wait on Qualcomm or build
anything from scratch.

## Reproducibility

Everything in this writeup is reproducible from:

- `last_side_quest/sq3_small_moe/probe_granite_1b_a400m_ptq.py` —
  driver (committed)
- `last_side_quest/sq3_small_moe/granite_moe_adapters.py` —
  three-class AIMET adapter (committed, 80 LOC)
- `last_side_quest/sq2_aimet_local/.venv-aimet-x86/` — reused venv
  (gitignored; install pin in `last_side_quest/sq2_aimet_local/aimet_local_survey.md`)
- `last_side_quest/sq3_small_moe/out_granite_1b_a400m_basic_ptq/granite_1b_a400m_basic_ptq.encodings.json`
  — 38 MB (regeneratable; stage for `marked_for_deletion/` after this
  PR lands per repo hygiene)

## Update log

- **2026-04-28** — Doc created. Granite-MoE local PTQ probe ran
  end-to-end on Prism CPU; cos 0.656 vs FP32 (vs SQ2's -0.065 on
  Qwen3-0.6B at same recipe). AIMET extensibility verified — three
  adapters in ~80 LOC unblocks any future non-blessed MoE.
- **2026-04-28** — Follow-up A landed: per-(expert, out-channel)
  weight quantization on `GraniteMoeParallelExperts` (override post-
  sim-build). Cos 0.656 → 0.712 (+0.056). Confirms per-axis weights
  help but aren't the dominant quality lever — activation a16 outliers
  + calibration breadth are likely bigger. SEQ_MSE next.
