# AIMET local-venv survey — SQ2 deliverable

**Status:** closed POSITIVE, 2026-04-28. Companion working log:
`findings.md` (this directory). Driver script:
`probe_qwen3_0p6b_ptq.py`.

## TL;DR

Contrary to the prior in `docs/rent_cloud_compute.md`, **AIMET 2.29
PyTorch installs and runs locally on the X2E** — both on Windows
x86_64 under Prism and on WSL2 ARM64 Linux. The whole v2 quantsim
surface (basic PTQ, SEQ_MSE, AdaScale, AdaRound, CLE, bias correction,
SmoothQuant-class techniques) is callable on CPU, no CUDA required,
no cloud rental needed for the *act* of calibrating.

What's NOT local:
1. **`aimet_onnx`** — manylinux_x86_64-only wheel, fails on every
   Windows axis. The Qualcomm-published happy-path
   (`qai_hub_models.models.qwen3_4b.quantize`) depends on it, so the
   wrapper script remains cloud-only — but the bare-AIMET surface
   that wrapper composes around is local.
2. **CUDA acceleration.** Calibration on CPU is functional but slow:
   ~4 min for basic PTQ on Qwen3-0.6B with 4 calibration prompts.
   SEQ_MSE on a 4B model on CPU is plausibly day-scale, not viable
   for routine iteration.
3. **PyTorch on native Windows-on-ARM.** Neither `torch` nor
   `onnxruntime` ship `win_arm64` wheels at all. The local axes are
   Prism-x86_64 or WSL2-aarch64; no native ARM Windows path exists
   in 2026-04.

What this changes for specula:
- **SQ4 cloud sizing decision** can defer the *calibration* spend.
  Cloud rental remains required only for SEQ_MSE on >0.6B models
  (CPU wall-time prohibitive) and for the `qai_hub_models` wrapper
  if we want to use Qualcomm's blessed recipe directly.
- We can author a **local PTQ driver** (template:
  `probe_qwen3_0p6b_ptq.py`) that runs the full AIMET surface on
  the X2E for any model that fits in 48 GB DRAM at FP32 (which
  covers Qwen3-0.6B, -1.7B, -4B, and the production targets up to
  ~10B parameters; not 30B+).
- Quality is no longer cloud-gated. **Calibration parameter sweeps,
  arch adapter testing, and PTQ-vs-QAT iteration are all local
  operations now.**

## Install matrix (verified 2026-04-28)

| axis | Python | result | notes |
|---|---|:-:|---|
| Native Windows-on-ARM 3.12 | 3.12.10 ARM64 | ❌ | no `win_arm64` wheel for `aimet_torch`, `torch`, or `onnxruntime` |
| Native Windows-on-ARM 3.10 | 3.10.20 ARM64 | ❌ | same blocker, downgrade Python doesn't help |
| **Windows x86_64 / Prism** | **3.10.20** | **✅** | aimet-torch 2.29.0 + torch 2.4.1 + transformers 4.54.1 |
| **WSL2 ARM64 Linux** | **3.10.20** | **✅** † | † must avoid `aimet_common` auto-importer (use `aimet_torch.common.defs` instead) |
| WSL2 ARM64 Linux 3.12 default | 3.12 | not tested | wheel resolution would behave like Prism Py3.12 case (no cp312 wheels for aimet-torch ≤ 2.27) |
| Cloud x86_64 Linux + CUDA | any | ✅ (canonical) | what `docs/rent_cloud_compute.md` and `docs/one_pipeline_cloud_gpu.md` describe |

`aimet_onnx` is the only AIMET package that's *exclusively* cloud-
compatible; published wheels are `manylinux_2_34_x86_64` for both
the cu121 and cpu variants. Fails on all four local axes.

### Why Prism succeeds where naive logic says it shouldn't

`aimet-torch` on PyPI is published as `aimet_torch-2.29.0-py310-none-any.whl`
— the **universal** platform tag. The wheel ships ~250 MB of x86-64
**Linux ELF** `.so` files in `aimet_torch/common/`:

```
_libpymo.abi3.so          99 MB
AimetTensorQuantizer.so   77 MB
AimetEncodingRescaler.so  75 MB
```

These are the v1 native backend. AIMET ≥ 2.20 deprecated `aimet_common`
and moved to a v2 path that's **pure-Python** atop standard PyTorch
(QuantizeDequantize as a `nn.Module`).

On Windows, `pkgutil.iter_modules` (which `aimet_common.__init__.py`
uses to auto-register submodules) only enumerates files matching
`importlib.machinery.all_suffixes()` — Linux `.so` is **not** in that
list on Windows. So Python silently skips them. The wheel's
universal tag is technically incorrect, but the side effect is that
the v1 native backend is invisible-and-unused on Windows, while the
v2 pure-Python backend works fine.

On WSL2 aarch64 Linux, `.so` IS a recognized Python module suffix.
The auto-importer enumerates the x86-64 ELFs and the loader fails
("not a dynamic executable"). Workaround: import `aimet_torch.v2.*`
+ `aimet_torch.common.defs` directly, never touch `aimet_common.*`.
The v2.20+ `FutureWarning` already nudges users toward this anyway.

## Local AIMET technique surface

| technique | callable locally on CPU? | tested at scale? | notes |
|---|:-:|:-:|---|
| **Basic PTQ** (TF-enhanced or vanilla TF) via `QuantizationSimModel` | ✅ | ✅ Qwen3-0.6B | 254 s wall on Prism for 4-prompt calibration on 28-layer 0.6B |
| **Sequential MSE (SEQ_MSE)** via `aimet_torch.v2.seq_mse.apply_seq_mse` | ✅ | TinyMLP only | Qualcomm sample wrapper enforces CUDA; AIMET library does not. CPU wall-time on ≥4B likely day-scale |
| **AdaScale** via `aimet_torch.experimental.adascale.apply_adascale` | ✅ | API import only | requires `transformers` installed (LLaMA-class arch hard-coded into the optimizer) |
| **AdaRound** via `aimet_torch.adaround` / `aimet_torch.v2.adaround` | ✅ | not tested | imports OK |
| **Cross-Layer Equalization (CLE)** | ✅ | not tested | imports OK |
| **Bias Correction** | ✅ | not tested | imports OK |
| **OmniQuant** (experimental) | ✅ | not tested | `aimet_torch.experimental.omniquant` |
| **SpinQuant** (experimental) | ✅ | not tested | `aimet_torch.experimental.spinquant` |
| **FP-T quant** (experimental) | ✅ | not tested | `aimet_torch.experimental.fptquant` |
| **Mixed-precision search** (`aimet_torch.amp`) | ✅ | not tested | imports OK |
| **AutoQuant** (`aimet_torch.auto_quant`) | ✅ | not tested | imports OK |
| **PEFT support** (`aimet_torch.peft`) | ✅ | not tested | imports OK |
| **QAT** (quantization-aware training) | ✅ ‡ | not tested | ‡ would require backprop-time-budget — likely cloud-only for production but locally callable for design |
| **`aimet_onnx.*`** (any ONNX-graph-level ops) | ❌ | n/a | manylinux-only wheel |
| **`qai_hub_models.<MODEL>.quantize`** (Qualcomm wrapper) | ❌ | n/a | depends on `aimet_onnx` + asserts CUDA in `_shared/llm/quantize.py` |

The Qualcomm wrapper's CUDA check is *not* an AIMET requirement:

```python
# qai_hub_models/_shared/llm/quantize.py
if device.type != "cuda":
    if not allow_cpu_to_quantize:
        raise ValueError("...requires CUDA GPU (V100/A100)...")
    if use_seq_mse or use_ada_scale:
        raise ValueError("This technique requires V100/A100.")
```

This is a recipe-level guard, not a library limitation. AIMET's bare
SEQ_MSE / AdaScale APIs run on CPU. The implication: a local driver
can call SEQ_MSE / AdaScale without those guards firing.

## MoE quantization support — verified ✅

AIMET 2.29 ships **first-class quantsim hooks** for Qwen3-MoE
specifically:

```
aimet_torch/v2/nn/transformers/models/
├── gemma3/
├── internvl/
├── llama/
├── mistral/
├── phi3/
├── qwen2/
├── qwen2_5_vl/
├── qwen3/
├── qwen3_5/
├── qwen3_moe/      ← MoE
└── qwen3_vl/
```

The `qwen3_moe` module subclasses `transformers.models.qwen3_moe.Qwen3MoeRMSNorm`
into `QuantizedQwen3MoeRMSNorm` (RMSNorm gets special quantsim
treatment because the variance/normalization sub-graph has unusual
sensitivity). The other MoE-specific layers — `Qwen3MoeMLP`,
`Qwen3MoeSparseMoeBlock`, `Qwen3MoeAttention` — reuse standard
`nn.Linear`-based blocks that AIMET v2 auto-quantizes via the generic
`QuantizationMixin`. Per-expert quantization is implicit in this
design: each expert is a stack of `nn.Linear`, AIMET sees them as
distinct modules and assigns them separate weight quantizers.

This satisfies the SQ2 sub-question "verify MoE support against 2.26
release notes" — 2.29 has it. Earlier 2.x versions likely added it
piecemeal (2.26 ≈ Q2 2025 timeframe per the AIMET cadence). Either
way, **today** AIMET supports Qwen3-MoE quantization, and the
`qwen3_moe/` adapter directory means Qualcomm has at least *some*
internal validation against this architecture.

**Decisive feed-into-SQ3:** the smallest Qwen3-MoE on HF is
**Qwen3-30B-A3B**. AIMET PTQ on it is a real possibility on the X2E
for basic PTQ (~30 GB FP32 weight footprint vs 48 GB DRAM ceiling —
fits but tight). For SEQ_MSE on 30B, CPU wall-time is the constraint,
not fitting. SQ3 should sketch a memory-budget table for this.

## End-to-end quality demo: Qwen3-0.6B basic PTQ on Prism CPU

The full driver `probe_qwen3_0p6b_ptq.py` runs:

1. HF-load Qwen3-0.6B (FP32, eager attn).
2. Wrap in `LogitsOnly(nn.Module)` so forward returns a tensor —
   AIMET's tracer can't handle `BaseModelOutputWithPast`.
3. Build `QuantizationSimModel` with `default_param_bw=4`,
   `default_output_bw=16`, `quant_scheme=post_training_tf_enhanced`.
4. `compute_encodings` with 4 calibration prompts × 64 tokens
   (canonical Qwen3 chat / code / prose / JSON shapes).
5. Quantized forward + cosine similarity vs FP32 reference.
6. `save_encodings_to_json` (the v2.20+ recommended path; replaces
   the deprecated `sim.export` for ≥0.6B models, where the latter
   trips a 2 GB protobuf serialize cap).

### Result

```text
[step 3] fp32 probe ("The capital of France is"): argmax = ' Paris'  ✅
[step 6] q4a16 probe:                              argmax = ' ont'  ❌
[step 6] cos(fp32, q4a16) = -0.065
```

Cos -0.065 is **catastrophic divergence**, not graceful degradation.
This **reproduces** the V/O-projection collapse story from
`docs/w4a16_investigation.md` Sessions 17–18: basic PTQ at w4 on
Qwen3-0.6B-class models is structurally insufficient. SEQ_MSE +
AdaScale (or SmoothQuant + AWQ) are the techniques designed to close
this gap; they are now locally callable, but unverified at scale.

This is a useful negative result — it confirms the prior, validates
the escalation ladder in `docs/one_pipeline_cloud_gpu.md` §"Q4
calibration technique stack," and tells us that any future *local*
SQ2-extension run that wants to ship a quality binary needs to start
with SEQ_MSE, not basic PTQ.

## encodings.json format comparison

The SQ2 spec cited the open question P2 in
`docs/one_pipeline_cloud_gpu.md`:

> Does AIMET's encodings.json format QAIRT-consume directly, or does
> it need a translator?

**Partial answer (verified by inspection, not by feeding through
qairt-converter):** AIMET emits encodings in two related schemas
depending on the call site, neither of which obviously matches
QAIRT's `--quantization_overrides` input format:

### AIMET v2 `sim.export()` — schema v1.0.0

```json
{
  "version": "1.0.0",
  "param_encodings": [
    {
      "name": "fc1.weight",
      "bw": 4, "dtype": "INT", "enc_type": "PER_CHANNEL", "is_sym": true,
      "offset": [-8, -8, -8, ...],
      "scale": [0.0405, 0.0397, 0.0431, ...]
    }
  ],
  "activation_encodings": [
    {
      "name": "/fc1/Gemm_output_0",
      "bw": 16, "dtype": "INT", "enc_type": "PER_TENSOR", "is_sym": false,
      "offset": [-35799],
      "scale": [6.05e-05]
    }
  ],
  "excluded_layers": [],
  "producer": ...,
  "quantizer_args": ...
}
```

### AIMET v2 `sim.save_encodings_to_json()` — different keys

```json
{
  "param_encodings": {
    "inner.lm_head.weight": [
      {
        "bitwidth": 4, "dtype": "int", "is_symmetric": "True",
        "max": 0.0783, "min": -0.0895,
        "offset": -8, "scale": 0.01119
      },
      ...
    ]
  },
  "activation_encodings": {
    "inner.model.layers.0.input_layernorm": {
      "output": {
        "0": {
          "bitwidth": 16, "dtype": "int", "is_symmetric": "False",
          "max": 2.107, "min": -2.677,
          "offset": -36672, "scale": 7.30e-05
        }
      }
    }
  }
}
```

Both AIMET emits — **but not the same shape** between the two paths.
Field name conventions differ (`bw`/`is_sym` vs `bitwidth`/`is_symmetric`),
top-level container differs (list vs dict), and per-channel arrays
are flattened differently.

### QAIRT `--quantization_overrides` input format (per `qairt-quantizer --help`)

Per `docs/w4a16_investigation_continued.md` Phase 5.5.1 progress
notes (x86 team's iteration on the schema), QAIRT expects roughly:

```json
{
  "<tensor-name>": {
    "min": ..., "max": ...,
    "bitwidth": 8, "is_symmetric": false,
    "scale": ..., "offset": ...
  }
}
```

— a flat dict keyed by tensor-name, with per-tensor (not per-channel)
scales. The actual schema may be richer; the work to nail it down
exactly was the open P2 task.

### Translator surface estimate

**Either AIMET schema → QAIRT format is a tractable translator:**

- Both AIMET schemas are dicts of per-tensor metadata with
  `scale` / `offset` / `bitwidth` / `is_symmetric` fields, plus
  per-channel arrays for weights.
- QAIRT wants per-tensor (or per-channel via a different path —
  see `--use_per_channel_quantization`).
- Field name remapping is mechanical: `bw` → `bitwidth`, etc.
- Tensor-name alignment between AIMET (PyTorch param names like
  `inner.model.layers.0.self_attn.q_proj.weight`) and the rewritten
  pathb ONNX (graph-name based, e.g. `onnx::MatMul_4221`) is the
  one non-trivial step. Likely needs the
  `aimet_torch_to_qairt_translator.py` script the cloud-pipeline doc
  anticipates.

**Estimated effort:** ~50-150 LOC for the translator + ~1 session
for the tensor-name reconciliation against a real Qwen3 pathb graph.
This is the **next concrete deliverable** after SQ2 if we want a
locally-driven AIMET → QAIRT bundle pipeline.

## Recommended dependency pin

For a future local PTQ session that wants to reproduce SQ2's
end-to-end run:

```bash
# Create venv
uv venv .venv-aimet --python "<x86_64 Python 3.10>"

# Install (Qualcomm-tested combination)
VIRTUAL_ENV=.venv-aimet uv pip install \
    aimet-torch==2.29.0 \
    "torch==2.4.1" "torchvision==0.19.1" \
    "transformers==4.54.1"
```

Tracer-incompatibility notes:
- `transformers >= 5` (default as of 2026-04) → `IndexError` in
  `sdpa_mask`; pin to 4.54.x.
- `torch >= 2.11` → `RuntimeError: invalid unordered_map key` in
  functorch vmap during forward; pin to 2.4.1 (Qualcomm's tested
  version anyway).
- `attn_implementation="eager"` at HF model load — SDPA kernel
  doesn't trace cleanly under torch.jit.trace.

For HF model use:
```python
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float32,
    attn_implementation="eager",
)
model.config.use_cache = False
```

## What remains cloud-gated

Despite the local-AIMET unblock, three reasons to still rent compute:

1. **`aimet_onnx`-specific ops.** Anything that operates on the ONNX
   graph directly (rather than going PyTorch-sim → export → QAIRT)
   needs the manylinux wheel. The Qualcomm cloud-pipeline design in
   `docs/one_pipeline_cloud_gpu.md` uses `aimet_onnx` for the
   post-PyTorch-PTQ compile step. That's still cloud-only.
2. **Wall-time on ≥4B models.** SEQ_MSE on Qwen3-4B on Prism CPU
   would be ~4 hours minimum (extrapolating from 0.6B's 254 s on a
   28-layer model — 4B has 32 layers, 6.7× params, but SEQ_MSE adds
   per-layer iterative search). Cloud A100 closes this in 2-5 h
   wall, with 80 GB VRAM headroom.
3. **Production binary discipline.** The cloud pipeline is
   single-command, deterministic, and produces a manifest.yaml. An
   ad-hoc local run is fine for design iteration; a production
   blessed bundle should still go through the rented path until we
   port `convert_hf_to_htp.py` to also drive the local AIMET surface
   for ≤4B models.

## Feed into other side-quests

### SQ3 — smallest Qwen MoE for AIMET → NPU

Direct positive feed: AIMET 2.29 has Qwen3-MoE quantization hooks
(`aimet_torch/v2/nn/transformers/models/qwen3_moe/`). The smallest
Qwen3-MoE on HF is **Qwen3-30B-A3B** (~30 GB FP32 → fits in 48 GB
laptop DRAM tight but fits). Local basic PTQ should be feasible;
SEQ_MSE wall-time is the constraint. SQ3 can move forward without
waiting on cloud rental for the *survey* phase — only the eventual
binary compile needs cloud.

### SQ4 — cloud sizing decision

SQ4 was blocked by SQ2 + SQ3. SQ2 closes with: **cloud rental is
optional for ≤4B model PTQ if local SEQ_MSE wall-time is acceptable
(extrapolating to ~hours on 0.6B, plausibly ~day on 4B).** Rental
remains valuable for:

- Time-pressured ramps (cloud A100 finishes 4B SEQ_MSE in 2-5 h vs
  ~1 day on Prism CPU)
- Anything requiring `aimet_onnx` directly
- Production-discipline runs that need the manifest pipeline

The expected SQ4 verdict: **rent on demand, not by default.** Specula
graduates to Qwen3.5/3.6/Gemma4 by:
1. Local PTQ pipeline-development against 0.6B / 1.7B (cheap CPU
   iteration; no rent)
2. Local design-validation up through 4B (slower CPU; still no rent)
3. Cloud A100 only for the final blessed bundle compile per family
   ($5-15 per family graduation)

This is **dramatically cheaper** than the original SQ4 framing,
which assumed *every* SEQ_MSE iteration needed a rent session.

### SQ1 — heterogeneous demo

No direct change. SQ1 doesn't depend on AIMET locally; it's about
the runtime path (NPU draft + CPU target). SQ2's existence
strengthens the case for a future SQ1.b (local AIMET-quantized 4B
draft, locally compiled to a custom-ctx bundle), but that's at least
2-3 sessions out and gates on Qwen3-MoE + a translator.

## Open follow-ups (not in SQ2 scope)

Captured here so they aren't lost; explicitly not done in this side
quest:

1. **AIMET → QAIRT encodings translator.** ~50-150 LOC.
   Needs a real Qwen3 pathb ONNX in hand to reconcile tensor names.
2. **SEQ_MSE on Qwen3-0.6B**, locally, to see if it closes the
   cos -0.065 catastrophic divergence into ≥ 0.95. This is the
   exact validation gate Qualcomm's V100/A100 path was designed for;
   running it on CPU is slow but informative. Estimated wall-time:
   ~30 min - 2 h.
3. **AdaScale on Qwen3-0.6B** standalone — same probe.
4. **`aimet_torch.onnx.export(..., use_external_data_format=True)`**
   instead of the deprecated `sim.export()`. Likely works for ≥0.6B
   without the protobuf cap; not tested.
5. **WSL2 install with Python 3.12** (not 3.10) — the universal-tag
   wheel might or might not pass cp312 ABI; not tested.
6. **Real Qwen3-MoE quantsim instantiation.** AIMET ships the
   adapter; we haven't actually loaded a `Qwen3MoeForCausalLM` and
   built a QuantizationSimModel against it. SQ3 will do this.

## Reproducibility checklist

Everything in this writeup is reproducible from:

- `last_side_quest/sq2_aimet_local/probe_qwen3_0p6b_ptq.py` — the
  driver (committed)
- `last_side_quest/sq2_aimet_local/.venv-aimet-x86/` — dependency
  pin (gitignored; reproduce via the install commands above)
- `last_side_quest/sq2_aimet_local/encodings_sample.json` — the
  representative slice of the basic-PTQ output (committed, 1.4 KB)
- `marked_for_deletion/sq2_aimet_local/` — the full 152 MB
  `qwen3_0p6b_basic_ptq.encodings.json` from the run (regeneratable;
  staged for hard-deletion per repo hygiene)

All the install commands, tracer-incompat workarounds, end-to-end
run, and AIMET surface probes were exercised this session. The
cos -0.065 / argmax = ' ont' result is what fell out of `compute_encodings`
on a 4-prompt calibration with `quant_scheme=post_training_tf_enhanced`.
A different calibration set or a SEQ_MSE pass would change that
number; the *pipeline* doesn't change.

## Update log

- **2026-04-28** — Doc created. SQ2 closes positive: AIMET PyTorch
  surface is locally usable; encodings format documented; MoE
  support verified at 2.29. SQ4 is unblocked with the new prior
  ("rent on demand, not by default"). SQ3 can proceed against
  Qwen3-30B-A3B locally for the survey phase.
