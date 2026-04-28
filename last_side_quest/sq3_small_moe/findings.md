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

## Follow-up B — SEQ_MSE on the same 1.3 B-MoE (4-prompt calibration)

**Counter-intuitive result: cos 0.712 → 0.640.** SEQ_MSE made the
model **worse**, not better. 362 s wall-time for the SEQ_MSE search,
+15.4 s for activation `compute_encodings` afterward, total ~377 s.
Driver: `probe_granite_1b_a400m_seqmse.py`.

| run | technique | calib data | calib time | cos | argmax |
|---|---|---|---:|---:|---|
| A0 | basic PTQ, per-tensor experts | 4 × 64-tok | 99.8 s | 0.656 | `<\|end_of_text\|>` |
| **A1** | **basic PTQ, per-(exp, out) experts** | 4 × 64-tok | 70.2 s | **0.712** | ` ` |
| B | SEQ_MSE + per-(exp, out) experts | 4 × 64-tok | 377 s | 0.640 | ` ` |
| B.5 | SEQ_MSE + per-(exp, out) experts | 16 × 64-tok | **1333 s** | **0.682** | ` ` |

SEQ_MSE was applied to the 121 `nn.Linear`-class modules (attn proj
× 24 layers × 4 + router-gate × 24 + lm_head). The 48
`GraniteMoeParallelExperts` were *not* SEQ_MSE-optimized — SEQ_MSE
only handles `nn.Linear`-class modules; the fused 3D-weight experts
fell through to `compute_encodings` for their weight encodings.

**Why did SEQ_MSE regress.** The most likely explanation is
**calibration overfitting**:

- SEQ_MSE evaluates 20 candidate weight-clip values per channel and
  picks the one minimizing per-layer MSE *on the calibration set*.
- With only 4 prompts × 64 tokens (256 tokens of context total),
  per-layer MSE estimates are noisy. Picking the lowest-MSE candidate
  can overfit to that noise.
- Per-layer optimization doesn't account for cross-layer error
  compounding. A weight scale that's locally optimal can be globally
  worse if it amplifies a downstream-sensitive error mode.
- Basic PTQ uses a coarser TF-enhanced histogram approach that's
  more robust to small calibration sets. SEQ_MSE's iterative search
  needs more data to be reliable.

This **directly contradicts the SQ2 prior** for the Qwen3-0.6B case:
> "SEQ_MSE on Qwen3-0.6B, locally, to see if it closes the cos
> -0.065 catastrophic divergence into ≥ 0.95."

With 4 prompts, SEQ_MSE may not close it — and may make it worse
through overfitting. The technique needs **calibration breadth**,
not just the technique name.

**Wall-time per module:** 121 modules × 20 candidates × 4 batches ≈
9680 layer-forwards. 362 s / 9680 ≈ **37 ms per layer-forward**.
Linear in num_candidates × num_batches. For 32-prompt × 64-token
calibration: ~50 minutes. For 128-prompt × 64-token: ~3 hours.

**Encodings.json:** 503 MB (same shape as A1 since per-channel
overrides on experts dominate the file size).

## Follow-up B.5 — SEQ_MSE retest with 16 calibration prompts

**Result: cos 0.682.** Recovered partially from B's 0.640 but still
**below basic-PTQ-A1's 0.712**. Wall-time scaled linearly (4× cal
data ⇒ 4× time): 22 minutes for SEQ_MSE + 1 minute for
compute_encodings. Encodings.json: 503 MB (same shape as A1/B —
per-channel ParallelExperts override dominates).

**Refines the SEQ_MSE narrative.** It's not pure overfitting —
more data did help, going 4→16 prompts closed ~30% of the
distance back toward A1. But the trend (0.640 → 0.682, +0.042
absolute, vs A1's 0.712 = +0.030 above B.5) suggests **diminishing
returns**: a 32-prompt run would likely land somewhere in
0.69-0.70, **still below A1**. To beat per-channel basic PTQ on
this 1.3B MoE, we'd need either **much** more calibration (64+
prompts, ~hours wall-time) **or** a different technique.

**Probable explanation specific to MoE.** SEQ_MSE optimizes per-
layer reconstruction MSE on the **routed** activations. With 32
experts top-8, each expert sees ~1/4 of tokens during calibration.
Per-layer MSE estimates on attention proj weights (Q/K/V/O are
shared across experts) get full-token statistics, but per-router-
gate MSE on the routing layer is noisy because routing is sparse
and discrete-ish. SEQ_MSE may pick router-gate scales that fit
calibration tokens' routing patterns but generalize poorly. This
is consistent with the observation that SEQ_MSE on dense models
(qwen3-0.6b would be the apples-to-apples test) is reported to
work better than what we see here.

**Verdict for the SQ3 quality stack.** A1 (basic PTQ + per-channel
weights on fused experts) is the locally-achievable champion at
**cos 0.712** on Granite-1B-A400M w4a16. SEQ_MSE with this hardware
budget regresses; AdaScale untested. The next quality lever is
likely **AdaScale** (which scales activations rather than weights —
designed exactly for outlier-channel activation collapse, the SQ2
finding) or **expanding bit-width** (a16 → a32 on outlier-prone
projections like V/O / down_proj).

## Follow-up C — OLMoE-1B-7B (NEGATIVE — AIMET v2 incompatible at multiple layers)

**Status: did NOT complete an end-to-end PTQ run.** Three iterations,
each peeling back a deeper AIMET-v2-vs-OLMoE incompatibility. The
adapter/probe (`olmoe_adapters.py`, `probe_olmoe_1b_7b_ptq.py`)
land as a **reference for the failure modes**, not as a working pipeline.

### Why OLMoE is structurally harder than Granite

Compare expert-dispatch designs:

| arch | how experts get tokens | empty-expert calls? |
|---|---|---|
| **Granite-MoE** (`GraniteMoeParallelExperts`) | Fused 3D weight tensor `[num_experts, out, in]` + `expert_size = expert_size.tolist()` then a single index-based MatMul | No — internal index_select, never per-expert empty forward |
| **Qwen3-MoE** (`Qwen3MoeSparseMoeBlock`) | `nn.ModuleList([Qwen3MoeMLP(...) ...])` per-expert calls **but** filters `expert_hitted = (mask.sum() > 0).nonzero()` first | No — only iterates experts with ≥1 token |
| **OLMoE** (`OlmoeSparseMoeBlock`) | `nn.ModuleList([OlmoeMLP(...) ...])` per-expert calls, **iterates ALL `range(num_experts)`** regardless of token count | **Yes** — empty `[0, hidden]` forwards happen routinely |

OLMoE's choice to iterate all experts is the structural source of the
problems below.

### Iteration 1 — Empty-tensor crash in encoding analyzer

Fresh `compute_encodings` with adapters for RoPE (ignore) + RMSNorm:

```text
File "transformers/models/olmoe/modeling_olmoe.py:606
    current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
File ".../aimet_torch/v2/quantization/encoding_analyzer.py:210, _get_min_max
    min = torch.where(isfinite, hist_input, float("inf")).min()
RuntimeError: min(): Expected reduction dim to be specified for input.numel() == 0
```

Root cause: `expert_layer(empty_tensor)` flows through `gate_proj` → empty
output → AIMET's encoding analyzer chokes on `torch.min()` of empty input.

### Iteration 2 — `@QuantizationMixin.implements(OlmoeMLP)` adapter that early-returns on empty

Wrapped OlmoeMLP with a custom adapter that returns
`torch.zeros(*x.shape[:-1], hidden_size)` when `x.numel() == 0`,
bypassing inner Linears entirely on empty inputs.

```text
File "olmoe_adapters.py, in QuantizedOlmoeMLP.forward
    return super().forward(x)   # non-empty path
File "modeling_olmoe.py:229, in OlmoeMLP.forward
    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
File "...aimet_torch/v2/quantization/affine/quantizer.py:1124
RuntimeError: Failed to run QuantizeDequantize since quantization parameters
are not initialized. Please initialize the quantization parameters using
`compute_encodings()`.
```

Different error, deeper. The empty-input fix worked; non-empty path now
fails. Hypothesis: `@QuantizationMixin.implements(OlmoeMLP)` as a
parent-class wrapper somehow disrupts AIMET's per-quantizer
`patch_attr(quantizer, "forward", _no_op)` mechanism for inner
QuantizedLinears during `compute_encodings`. The patch sets quantizer
forwards to `_no_op` for stats collection — but our wrapper class
intercepts in a way that the patch doesn't propagate through.

### Iteration 3 — Class-level monkey-patch of `OlmoeMLP.forward` (not a Mixin) + disable per-expert output quantizers

Replaced the QuantizationMixin adapter with a transformers-class-level
monkey-patch (`OlmoeMLP.forward = _patched_forward`) so AIMET sees
stdlib OlmoeMLP and doesn't add MLP-level wrappers. **Plus** post-build
disable: walk `sim.model.named_modules()` and set
`output_quantizers = ModuleList([None])` on all 3072 inner expert
QuantizedLinears (`16 layers × 64 experts × 3 linears each`).

`compute_encodings` ran clean for **30 minutes** (1796 s). Then the
post-cal probe forward at step 6 hit the **same error**:

```text
File "modeling_olmoe.py:229, in OlmoeMLP.forward
    down_proj = self.down_proj(...)
File ".../quantizer.py:1124
RuntimeError: Failed to run QuantizeDequantize since quantization
parameters are not initialized.
```

This shouldn't happen — `_forward_no_dispatch` in true_quant.py:607-608
explicitly does `if self.output_quantizers[0]:` before applying. Setting
the entry to `None` should make this skip. But it didn't skip. Either:
(a) `compute_encodings` re-instantiated output_quantizers on the
disabled modules (bypassing the user's None setting), or (b) something
else holds an old reference. Did not pursue further — the diagnostic
loop here is open-ended and out of scope for this side-quest's budget.

### Verdict for SQ3-OLMoE

OLMoE's per-expert dispatch design is **architecturally hostile to
AIMET v2's PTQ pipeline** in ways that Granite's fused-experts and
Qwen3-MoE's hit-filtered iteration both avoid. Three layered
workarounds didn't yield a working pipeline. **A fourth+ workaround
might exist** — e.g., monkey-patching `OlmoeSparseMoeBlock.forward` to
emulate Qwen3-MoE's `expert_hitted` filter so empty-expert iterations
are simply skipped — but at this point the engineering cost equals or
exceeds writing a transformers-side PR to upstream that fix.

**Implication for the AIMET-MoE-adapter narrative.** The "~80 LOC per
arch" rule from SQ3-Granite was over-optimistic. It applies cleanly to
**fused-expert architectures**. **Per-expert-dispatch architectures
that don't filter empty experts** require either:
1. transformers-side fixes (upstream OLMoE to add `expert_hitted`),
2. AIMET-side fixes (encoding analyzer that handles empty stats),
3. or selective-quantization workarounds that we couldn't get working
   in the time budget.

For Specula's roadmap: **avoid OLMoE family until one of those fixes
lands**. Granite-MoE and (pending verification) Qwen3-MoE are the
locally-tractable MoE candidates.

### Cross-cutting AIMET-v2 limitation discovered

`compute_encodings`'s per-quantizer `patch_attr(quantizer, "forward",
_no_op)` mechanism doesn't survive in scenarios where:
- A quantizer is reached via a custom QuantizationMixin wrapper class
  whose forward calls `super().forward()` to delegate (Iteration 2),
- A quantizer's `output_quantizers[0]` is set to None post-sim-build,
  yet the post-cal forward still treats it as non-None (Iteration 3).

Both could be AIMET v2 bugs. Worth a github issue if Specula pursues
OLMoE further. For now, the workaround is "use Granite-MoE-shaped
architectures."

## Open follow-ups (not done in this session)
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
- **2026-04-28** — Follow-up B landed: SEQ_MSE with 4-prompt
  calibration *regressed* cos 0.712 → 0.640. Refutes the SQ2 prior
  that "SEQ_MSE on a small cal set closes catastrophic divergence."
  Hypothesis: per-layer MSE overfits a 4-prompt noise pattern. Need
  to retest with 32+ prompts to confirm before believing SEQ_MSE
  works at all on this hardware budget.
- **2026-04-28** — Follow-up B.5 landed: SEQ_MSE with 16-prompt
  calibration improved on B (cos 0.640 → 0.682) but still below
  basic-PTQ-A1's 0.712. Refines verdict: SEQ_MSE not pure overfitting,
  but diminishing returns suggest 64+ prompts (~hours) would only
  marginally help. **A1 is the locally-achievable champion** at this
  recipe; AdaScale is the next lever to try.
- **2026-04-28** — Follow-up C did NOT close: OLMoE-1B-7B's per-expert
  dispatch (vs Granite's fused experts and Qwen3-MoE's hit-filtered
  iteration) is architecturally hostile to AIMET v2 PTQ in ways that
  three layered workarounds couldn't fully overcome. Empty-tensor in
  encoding analyzer (fixed); `@QuantizationMixin.implements(OlmoeMLP)`
  disrupts inner-Linear quantizer-forward patching (worked around via
  monkey-patch); post-cal forward still hits "QuantizeDequantize not
  initialized" on disabled output_quantizers (root cause unclear, likely
  AIMET v2 bug). **Verdict: avoid OLMoE family until upstream fix.**
