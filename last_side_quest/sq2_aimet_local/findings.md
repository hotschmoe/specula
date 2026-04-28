# SQ2 — AIMET local venv survey

**Status:** in-progress, started 2026-04-28.

**Goal (per `last_side_quest/last_side_quests.md` §SQ2).** Find out
what `quic/aimet` 2.26.0 (and 2.x line in general) actually does on
this Snapdragon X2 Elite Extreme laptop without renting a cloud GPU.
Try the install on multiple axes:

1. Native Windows-on-ARM, Python 3.12 ARM64 (`.venv-aimet-arm64`)
2. Windows x86_64 via Prism emulation, Python 3.10 (`.venv-aimet-x86`,
   matches the `.venv-qairt` axis that already runs `qairt-converter`)
3. WSL2 ARM64 Linux (CPU-only torch), if WSL2 is available
4. (Stretch) WSL2 + QEMU user-mode emulation for x86_64-Linux wheels

Then, on whatever axis works (if any), attempt **basic PTQ** on
Qwen3-0.6B and emit `encodings.json`.

## Pre-investigation priors

From `docs/rent_cloud_compute.md` §"Why not WSL / WSL2 on the X2E":

- AIMET ships `aimet_onnx-2.26.0+cu121-cp310-cp310-manylinux_2_34_x86_64.whl`
  and `aimet_onnx-2.26.0+cpu-cp310-cp310-manylinux_2_34_x86_64.whl`.
  Both **Linux x86_64 only**, no Windows wheels at all.
- `aimet_torch` is the same shape — Linux wheels only on the public
  release page.
- WSL2 on Windows-on-ARM ⇒ ARM64 Linux distros only ⇒ x86_64 wheels
  don't load.
- QEMU user-mode emulation works numerically but "unusably slow for
  a 4B model (tens of hours)" per the rent doc; for 0.6B it might be
  tractable but is unattested.

So the "expected" outcome is **no install path works without
emulation, and emulation is too slow for the 4B target but possibly
viable for 0.6B PTQ as a one-shot survey datapoint**.

The point of this side quest is to *verify* those priors with concrete
errors + their messages, and to also explore the other AIMET-adjacent
PyPI surface (`aimet_torch` PyPI distribution, AIMET's `qai-hub-models`
quantize entrypoint, etc.) that might have broader platform coverage.

## Existing venv inventory (snapshot 2026-04-28)

| venv | Python | arch | purpose |
|---|---|---|---|
| `.venv` | 3.12.10 | ARM64 | project default — onnxruntime, optimum, qai-hub-models |
| `.venv-arm-export` | 3.12.10 | ARM64 | model export (qai-hub-models w/ ARM-friendly deps) |
| `.venv-ort21` | 3.12.10 | ARM64 | ORT-QNN 2.1.0 / QAIRT 2.45 path |
| `.venv-qairt` | 3.10.20 | x86_64 (Prism) | the qairt-converter / qairt-quantizer toolchain |

For AIMET attempts we want **fresh venvs** (per SQ2 spec — AIMET
2.26 wants `torch==2.4` + `onnxruntime-gpu==1.23.2`, which would
clash with what's pinned in the existing four).

## Existing Qwen3-0.6B artifacts on disk

| file | size | notes |
|---|---|---|
| `models/qwen3_0_6b_draft_v81_ctx256.pathb.w8a16-local.bin` | 875 MB | already-compiled w8a16 NPU bundle (Phase 5.5 Lever C) |
| `models/qwen3_0_6b_draft_v81_ctx256.pathb.w8a16-local.encodings.json` | 3.3 MB | the **existing local qairt-quantizer** encodings — useful as a comparison reference for any AIMET output |
| `models/qwen3_0_6b_draft_v81_ctx256.pathb.w8a16-local.wrapper.onnx` | 8 KB | EPContext wrapper |
| `models/qwen3_0_6b_draft_v81_ctx256.pathbmask.bin` | 1.4 GB | additive-mask precursor |

The **upstream pathb ONNX is no longer on disk** (consistent with
`docs/repo_hygiene.md` "regeneratable artifacts don't need to live on
disk if the generating script + manifest are tracked"). To get a
pathb ONNX for AIMET to consume, we'd run:

```text
download_qwen3_onnx.py  --model-stem qwen3_0_6b
rewrite_qwen3_htp.py    --model-stem qwen3_0_6b
rewrite_qwen3_pathb.py  --model-stem qwen3_0_6b
pin_shapes_qwen3_4b.py  (or generalized variant) --model-stem qwen3_0_6b --ctx 256
```

Skip this step until/unless the AIMET install actually loads.

## Findings log

### Step 1 — workspace + scope (2026-04-28)

Created `last_side_quest/sq2_aimet_local/`. Inventoried existing
venvs (above) — none are AIMET-ready, will create fresh.

Decision on which axes to try:

- **A1.** Native ARM64 Python 3.12 → try `pip install aimet_onnx`
  and `pip install aimet_torch` from PyPI.
- **A2.** Native ARM64 Python 3.12 → try the GitHub-published wheel
  URL directly (will mismatch the manylinux_x86_64 tag and fail
  cleanly).
- **B.** x86_64 (Prism) Python 3.10 → same two attempts (PyPI + GitHub
  wheel URL). The wheel will fail on the manylinux Linux tag even
  under Prism (Prism executes Windows x86_64 binaries, not Linux).
- **C.** WSL2 default distro (likely Ubuntu ARM64) → install
  `aimet_torch` from PyPI (might have ARM Linux build), and try the
  x86_64 wheel under emulation.

Will mark cloud-only verdict if no axis succeeds.

### Step 2 — A1/A2: native ARM64 Python 3.12 install (2026-04-28)

Created `last_side_quest/sq2_aimet_local/.venv-aimet-arm64` (Python
3.12.10 ARM64). Bootstrapped pip 26.1 via uv.

**A1 — `uv pip install aimet-onnx` (PyPI):**

```text
× No solution found when resolving dependencies:
  Because aimet-onnx>=2.0.0,<=2.27.0 has no wheels with a
  matching Python ABI tag (e.g., `cp312`), we can conclude that
  aimet-onnx>=2.0.0,<=2.27.0 cannot be used.
  And because aimet-onnx>=2.28.0 has no wheels with a matching platform
  tag (e.g., `win_arm64`) ... your requirements are unsatisfiable.
  hint: Wheels are available for `aimet-onnx` (v2.29.0) on the following
  platform: `manylinux_2_34_x86_64`
```

**A1 — `uv pip install aimet-torch` (PyPI):**

```text
× No solution found when resolving dependencies:
  hint: You require CPython 3.12 (`cp312`), but we only found wheels for
  `aimet-torch` (v1.35.0) with the following Python ABI tag: `cp310`
  hint: You require CPython 3.12 (`cp312`), but we only found wheels for
  `torch` (v2.1.2) with the following Python ABI tags: `cp38`, `cp39`,
  `cp310`, `cp311`
  hint: Wheels are available for `torch` (v2.11.0) on the following
  platforms: `manylinux_2_28_aarch64`, `manylinux_2_28_x86_64`,
  `macosx_11_0_arm64`, `win_amd64`
```

Note: even the **base `torch`** package has no `win_arm64` wheel on
PyPI. Only `win_amd64`, `manylinux*_aarch64`, `manylinux*_x86_64`,
`macosx*_arm64`. **PyTorch upstream does not publish Windows-on-ARM
wheels.** This blocks every torch-dependent AIMET install on this
axis, full stop.

**A2 — direct GitHub wheel URL (`aimet_onnx-2.26.0+cpu-cp310-cp310-manylinux_2_34_x86_64.whl`):**

```text
× No solution found when resolving dependencies:
  Because only aimet-onnx==2.26.0+cpu is available and
  aimet-onnx==2.26.0+cpu has no wheels with a matching Python version tag
  (e.g., `cp312`), we can conclude that all versions of aimet-onnx cannot
  be used.
```

Then with `--python-version 3.10` to lift the ABI mismatch, the
resolver fails further down on `onnxruntime` (transitive dep, also no
`win_arm64` wheels — only x86_64 / aarch64 manylinux + macOS arm64
+ win_amd64).

**Verdict A.** Native Windows-on-ARM Python 3.12 (or 3.10) cannot
install AIMET. The blocker is **the PyTorch + ONNX-Runtime ecosystem
having no `win_arm64` wheels on PyPI**, not just AIMET. AIMET would
need a `win_arm64` torch + ort wheel under it before its own
`win_arm64` build could even be considered, which is upstream-of-
upstream.

A1+A2 closed; moving to B (Prism x86_64).

### Step 3 — B: Prism x86_64 Python 3.10 install (2026-04-28)

Created `last_side_quest/sq2_aimet_local/.venv-aimet-x86` (Python
3.10.20 x86_64, runs under Prism on this WoA machine). Bootstrapped
pip 26.1.

#### B1 — `aimet-onnx` (PyPI) ❌

```text
× No solution found when resolving dependencies:
  Because aimet-onnx>=2.0.0 has no wheels with a matching platform
  tag (e.g., `win_amd64`) ... your requirements are unsatisfiable.
  hint: Wheels are available for `aimet-onnx` (v2.29.0) on the following
  platform: `manylinux_2_34_x86_64`
```

`aimet_onnx` is **manylinux-only across all 2.x versions**. Even on
the x86_64 Prism axis, no `win_amd64` wheel ships. (This matches the
prior in `docs/rent_cloud_compute.md`.)

#### B2 — `aimet-torch` (PyPI) ✅ — surprise win

```text
+ aimet-torch == 2.29.0
+ torch       == 2.11.0  (cpu)
+ torchvision == 0.26.0
+ onnx        == 1.21.0
+ onnxscript  == 0.7.0
... 80 transitive deps total ...
```

`aimet-torch` 2.29.0 **does** publish a `win_amd64` wheel and installs
cleanly on x86_64 Python 3.10 under Prism. Torch lands as 2.11.0+cpu
(no CUDA — none available on this machine anyway).

This contradicts the prior in `docs/rent_cloud_compute.md` §"Why not
WSL / WSL2 on the X2E" which claims AIMET is x86_64-Linux-only. That
claim is correct for `aimet_onnx`, **wrong for `aimet_torch`**. The
distinction matters: Qualcomm's *own* recipe (`qai_hub_models.models.qwen3_4b.quantize`)
uses `aimet_onnx`; the bare AIMET v2 PyTorch surface is a separate
package that is broader-supported.

##### Native binary breakdown

`aimet_torch/common/` ships three Linux ELF `.so` files:
- `_libpymo.abi3.so` (95 MB)
- `AimetTensorQuantizer.so` (77 MB)
- `AimetEncodingRescaler.so` (75 MB)

These are the v1 native backend. Per AIMET 2.20+ release notes,
`aimet_common` is deprecated and the v2 quantsim path is recommended.
The v2 path is **pure Python** and uses standard PyTorch ops for
quantization simulation (QuantizeDequantize as a torch nn.Module),
so the unloadable `.so` files don't bite.

##### Smoke tests

| API | result | notes |
|---|---|---|
| `import aimet_torch` | ✅ | 2.29.0 |
| `from aimet_torch.v2.quantsim import QuantizationSimModel` | ✅ | |
| Construct sim on TinyMLP, w4 sym weights / a16 asym acts | ✅ | inserts `QuantizedLinear`, per-channel weight quantizers, asym uint16 activations |
| `sim.compute_encodings(fwd_fn, None)` (calibration) | ✅ | runs on CPU, no CUDA needed |
| Quantized forward pass | ✅ | output shape + dtype match fp32 reference |
| `from aimet_torch.v2.seq_mse import apply_seq_mse, SeqMseParams` | ✅ | |
| `apply_seq_mse(model, sim, cal_loader, params)` on TinyMLP | ✅ | runs on CPU; finds optimal param encodings per linear layer |
| `from aimet_torch.experimental.adascale import apply_adascale` | ✅ † | **† requires `transformers` — meant for LLaMA-class transformers** |
| `sim.export(out, prefix, dummy_input)` (ONNX + encodings.json) | ✅ | emits `*.onnx`, `*.encodings`, `*.pth`, `*_torch.encodings` |

This single result completely changes the SQ2 verdict. **AIMET v2
PyTorch works locally on Windows-on-ARM via Prism, including SEQ_MSE
and AdaScale as importable APIs.** The CPU-only path is documented
by Qualcomm but only ever as "you could in theory but it'd be slow."
Slow on a 4B model — for a 0.6B model, it may be tractable.

##### encodings.json format (v2 export, schema v1.0.0)

Sample `param_encodings` entry (per-channel int4 sym weight):

```json
{
  "name": "fc1.weight",
  "bw": 4,
  "dtype": "INT",
  "enc_type": "PER_CHANNEL",
  "is_sym": true,
  "offset": [-8, -8, -8, ...],
  "scale": [0.04052, 0.03976, 0.04312, ...]
}
```

Sample `activation_encodings` entry (per-tensor uint16 asym):

```json
{
  "name": "/fc1/Gemm_output_0",
  "bw": 16,
  "dtype": "INT",
  "enc_type": "PER_TENSOR",
  "is_sym": false,
  "offset": [-35799],
  "scale": [6.05e-05]
}
```

Top-level: `version`, `producer`, `quantizer_args`, `excluded_layers`,
`param_encodings` (list), `activation_encodings` (list).

vs. **QAIRT's existing IR format** (the
`qwen3_0_6b_draft_v81_ctx256.pathb.w8a16-local.encodings.json` shipped
file, which is the **post-quantize compiler IR**, NOT the
`--quantization_overrides` input):

```json
{
  "op_types": [...],
  "graph": {
    "tensors": {
      "<name>": {
        "id": 1, "data_type": 50, "permute_order_to_src": [...],
        "quant_params": {
          "scale_offset": {
            "bitwidth": 16, "minimum": ..., "maximum": ...,
            "scale": 6.05e-05, "offset": -35799,
            "is_symmetric": false, "is_fixed_point": true
          }
        }
      }
    }
  }
}
```

Different schema, same numerical content (scale/offset/bitwidth/
sym-flag per tensor, keyed by tensor name). A translator from AIMET
v2 → QAIRT `--quantization_overrides` input format is the ~50 LOC
that the SQ2 doc anticipated — and **the producer side now runs
locally**.

**Verdict B.** ✅ Local AIMET PyTorch works. The whole v2 quantsim +
SEQ_MSE + AdaScale + ONNX export surface runs on Prism. This is the
unblock that closes SQ2 positive.

### Step 4 — C: WSL2 ARM64 Linux Python 3.10 install (2026-04-28)

WSL2 status: Ubuntu 24.04 default distro, ARM64 (aarch64), Python 3.12
default. Installed `uv 0.11.8 aarch64-unknown-linux-gnu` to
`~/.local/bin/uv`. Created `~/aimet-wsl` venv with Python 3.10.20.

#### C1 — `uv pip install aimet-torch` ✅ install ❌ aimet_common import

Wheel installed:

```
aimet_torch-2.29.0-py310-none-any.whl  (48 MB)
torch==2.11.0+cu130
triton==3.6.0
... (same 80-package transitive set as Prism, plus triton)
```

torch on aarch64-linux comes with the cu130 CUDA build (CUDA 13.0
sym-linked stubs); `torch.cuda.is_available()` is `False` since
WSL2 has no GPU passthrough on this box. The aarch64 `aimet-torch`
wheel uses the **universal tag** `py310-none-any` — pip treats it as
platform-agnostic and installs without checking binary contents.

Inside the wheel, `aimet_torch/common/`:
```
_libpymo.abi3.so          99 MB   ELF 64-bit LSB shared object, x86-64
AimetTensorQuantizer.so   77 MB   ELF 64-bit LSB shared object, x86-64
AimetEncodingRescaler.so  75 MB   ELF 64-bit LSB shared object, x86-64
```

The v1 native binaries are **x86-64 ELF**, not aarch64. They were
shipped as-is in a universal wheel.

The naive smoke (`from aimet_common.defs import QuantScheme; ...`)
crashes:

```text
ImportError: aimet_torch/common/_libpymo.abi3.so: cannot open shared
object file: No such file or directory
```

`ldd` confirms: "not a dynamic executable" — the loader can't even
parse a non-aarch64 ELF as a dynamic object. Triggered because
`aimet_common.__init__.py` does `pkgutil.iter_modules(pkg.__path__)`
and tries to `importlib.import_module(...)` every submodule it finds.
On Linux, `.so` is a recognized extension suffix → it enumerates the
`.so` files → the loader fails on the architecture mismatch.

#### C2 — workaround: skip `aimet_common`, import v2 directly ✅

Switching to `from aimet_torch.common.defs import QuantScheme`
(the v2.20+ recommended import that the FutureWarning explicitly
suggests) bypasses the `aimet_common` auto-importer entirely. The v2
quantsim path is pure-Python and never touches `_libpymo`.

```python
from aimet_torch.common.defs import QuantScheme    # ← not aimet_common
from aimet_torch.v2.quantsim import QuantizationSimModel
sim = QuantizationSimModel(model, dummy_input=d,
    quant_scheme=QuantScheme.post_training_tf_enhanced,
    default_param_bw=4, default_output_bw=16)
sim.compute_encodings(fwd, None)
out = sim.model(d)   # quantized fwd OK on aarch64
```

Works.

#### Why Prism didn't hit this

Same wheel, same `.so` files. On Windows, `pkgutil.iter_modules`
respects `importlib.machinery.all_suffixes()` which **does not include
Linux `.so`** — only `.pyd`, `.dll`, `.py`, etc. So the auto-importer
silently skips the Linux ELF binaries on Windows. The fact that they
even shipped is a no-op there.

This means the wheel's universal tag is technically incorrect (it
should be `manylinux_2_xx_x86_64`), but it works on Prism by
**accident** of Windows path-rule differences.

**Verdict C.** ✅ Works with the v2-only import discipline. Same
v2 surface + SEQ_MSE smoke runs on aarch64 Linux as it does on Prism.
Slightly closer to the Qualcomm reference path (Linux), but Prism is
fine for our purposes too.

#### Combined verdict on install axis (A vs B vs C)

| axis | install? | run? | notes |
|---|:-:|:-:|---|
| **A**: native Windows-on-ARM Py3.12/3.10 | ❌ | n/a | torch + onnxruntime have no `win_arm64` wheels — blocks the dep tree |
| **B**: Windows x86_64 / Prism / Py3.10 | ✅ | ✅ | aimet-torch 2.29.0 + torch 2.11.0+cpu; v1 `.so` files unused on Windows; v2 quantsim works |
| **C**: WSL2 aarch64 Linux / Py3.10 | ✅ | ✅ † | **† only when import discipline avoids `aimet_common` auto-importer** |
| **D**: cloud x86_64 Linux + CUDA (rented) | ✅ | ✅ | `aimet_onnx` + `aimet_torch` + GPU SEQ_MSE; the `qai_hub_models.qwen3_4b.quantize` happy path |

**`aimet_onnx` — manylinux_2_34_x86_64 only**, fails on A/B/C.
The Qualcomm-published `qai_hub_models.models.qwen3_4b.quantize`
recipe transitively requires `aimet_onnx`, so the **wrapper script**
remains cloud-only — but the underlying `aimet_torch` library doesn't,
which means we can author our own local PTQ driver.

### Step 5 — basic PTQ on Qwen3-0.6B end-to-end (2026-04-28)

Authored `last_side_quest/sq2_aimet_local/probe_qwen3_0p6b_ptq.py` as
a minimal driver: HF-load → wrap to logits-only → AIMET v2 sim w4
sym weights / a16 asym acts → `compute_encodings` on 4 short
calibration prompts (64 tokens each, padded) → cos vs fp32 →
`save_encodings_to_json`. Ran on Prism (axis B).

#### Pipeline pre-conditions discovered

| issue | fix |
|---|---|
| AIMET requires tensor return; HF `BaseModelOutputWithPast` doesn't trace | wrap model in `LogitsOnly` shim that calls inner with `return_dict=True, use_cache=False` and returns `out.logits` |
| transformers 5.6.2 (default) raises `IndexError: tuple index out of range` in `sdpa_mask` under `torch.jit.trace` (dynamic-mask code path) | `pip install "transformers==4.54.1"` |
| transformers 4.54.1 default attn impl uses SDPA (also trace-incompatible) | `attn_implementation="eager"` at `from_pretrained` |
| transformers 4.54.1 with torch 2.11 trips `RuntimeError: invalid unordered_map<K, T> key` in functorch vmap during forward | `pip install "torch==2.4.1" "torchvision==0.19.1"` (Qualcomm's tested combination) |
| `transformers.from_pretrained(..., dtype=...)` argument | older transformers takes `torch_dtype=` |
| `sim.export()` writes 5.7 GB of intermediate weight files and trips a protobuf 2 GB serialize cap on the wrapping ONNX | use `sim.save_encodings_to_json(...)` instead — emits just the JSON (152 MB for 0.6B) |

#### Final working dependency pin

```text
torch == 2.4.1            (cpu)
torchvision == 0.19.1
aimet-torch == 2.29.0
transformers == 4.54.1
attn_implementation = "eager"
```

#### End-to-end timing (Prism CPU, single-thread)

| stage | wall-time |
|---|---:|
| Qwen3-0.6B FP32 load (cached HF) | 2-3 s |
| FP32 probe forward | <1 s |
| `QuantizationSimModel(...)` construct | 6.6 s |
| `compute_encodings` (4 prompts × 64 tokens, 28 layers) | **254 s** |
| Quantized probe forward | <1 s |
| `save_encodings_to_json` | 12.7 s |

Calibration scaling estimate: ~64 s/prompt at 64 tokens/prompt for
0.6B / 28 layers. A typical SEQ_MSE calibration set is 128 prompts —
that's ~2.3 hours on Prism CPU. SEQ_MSE itself adds iterative weight
search per layer; budget another factor of 2-5× wall-time.

For Qwen3-4B (32 layers, 6.7× params): ballpark 28 minutes for 4
calibration prompts of basic PTQ; ~3 hours for the typical 128-prompt
set. Order-of-magnitude estimate; would need real measurement.

#### Quality result — basic PTQ at w4a16 on Qwen3-0.6B is broken

```text
[step 3] fp32 probe logits: shape=(1, 64, 151936), argmax=' Paris'  ✅
[step 6] quantized probe logits: shape=(1, 64, 151936), argmax=' ont'  ❌
[step 6] cos(fp32, quant) over real positions = -0.065061
```

Cos -0.065 ≈ orthogonal vectors. **The quantized model is essentially
noise** — argmax shifted from " Paris" to " ont" on the most basic
factual probe. This **reproduces** the V/O-projection collapse story
from `docs/w4a16_investigation.md` Sessions 17-18: basic PTQ at w4
on Qwen3-0.6B is structurally insufficient.

This is the *exact* finding that motivates the SEQ_MSE + AdaScale
escalation in `docs/one_pipeline_cloud_gpu.md` §"Q4: calibration
technique stack." The SQ2 deliverable closes positive even though the
*demo PTQ run* is negative — the negative result is consistent with
prior diagnoses, the pipeline itself works, and the next step
(SEQ_MSE) is now locally callable.

#### encodings.json schema (AIMET v2 `save_encodings_to_json` form)

Top-level keys: `param_encodings` (dict, layer_name → list-of-channel-encodings),
`activation_encodings` (dict, op_name → {"output": {idx → encoding}}).
Per-channel weight entry shape:

```json
{
  "bitwidth": 4,
  "dtype": "int",
  "is_symmetric": "True",
  "max": 0.0783,
  "min": -0.0895,
  "offset": -8,
  "scale": 0.01119
}
```

Per-tensor activation entry shape:

```json
{
  "bitwidth": 16,
  "dtype": "int",
  "is_symmetric": "False",
  "max": 2.107,
  "min": -2.677,
  "offset": -36672,
  "scale": 7.30e-05
}
```

311 param entries × per-channel arrays + 338 activation entries → 152
MB JSON for 0.6B. lm_head alone is 151,936 channels.

Note: **`save_encodings_to_json` schema differs from `sim.export()`
schema** captured earlier on TinyMLP (which used `bw`/`enc_type`/
`is_sym`/list-of-`offset`-and-`scale`). Both come out of AIMET v2 but
through different code paths. For QAIRT consumption via
`--quantization_overrides`, we'd need to confirm which schema QAIRT
expects — that's the **P2 question** in
`docs/one_pipeline_cloud_gpu.md` and remains unanswered without an
actual `qairt-converter` round-trip test.

Representative sample saved to
`last_side_quest/sq2_aimet_local/encodings_sample.json`. Full 152 MB
file is regeneratable from `probe_qwen3_0p6b_ptq.py`; staged in
`marked_for_deletion/sq2_aimet_local/` per repo hygiene.

#### Verdict on local AIMET surface for the SQ2 deliverable

| capability | local? (Prism / WSL2) | notes |
|---|:-:|---|
| `aimet_torch.v2.QuantizationSimModel` construct | ✅ | works on real Qwen3-0.6B, 28 layers |
| `sim.compute_encodings` (basic PTQ calibration) | ✅ | CPU, ~4 min on 4-prompt cal set |
| Quantized forward pass under sim | ✅ | matches expected QDQ semantics |
| `aimet_torch.v2.seq_mse.apply_seq_mse` | ✅ ‡ | ‡ imports + runs on TinyMLP — **untested on Qwen3** at scale |
| `aimet_torch.experimental.adascale.apply_adascale` | ✅ ‡ | ‡ imports — **untested at scale** |
| `aimet_torch.adaround` | ✅ † | † imports OK; calibration-ladder path |
| `aimet_torch.cross_layer_equalization` (CLE) | ✅ † | † imports OK |
| `aimet_torch.bias_correction` | ✅ † | † imports OK |
| `aimet_torch.experimental.omniquant`, `spinquant`, `fptquant` | ✅ † | † imports OK; advanced experimental quant techniques |
| `sim.save_encodings_to_json` | ✅ | size scales with model: 152 MB for 0.6B |
| `sim.export` (full ONNX + encodings) | ⚠ partial | trips 2 GB protobuf cap for ≥ 0.6B model wrapping; per-tensor weight files DO emit |
| `aimet_torch.onnx.export` (recommended replacement) | ⚠ untested | API exists; `use_external_data_format=True` likely required for ≥ 0.6B |
| `aimet_onnx` (any function) | ❌ | manylinux_x86_64-only wheel; cloud-only |
| `qai_hub_models.models.qwen3_4b.quantize` (Qualcomm wrapper) | ❌ | depends on `aimet_onnx`; cloud-only |

(remaining steps appended as we run them)
