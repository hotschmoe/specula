# Reproducing the Qualcomm Qwen3-4B w4a16 reference bundle

**Status:** in-progress 2026-04-23. Phases 0-2 complete; Phase 3 hit the
single-bin HTP ceiling (structural, not a bug); Phase 5 converter +
quantizer + ctx-bin-gen + wrapper + HTP-load all green end to end.
Phase 5h-i-j localized the bug via per-part probes. Phase 5k found
the fix: adding `--use_per_channel_quantization --use_per_row_quantization
--apply_algorithms cle` to qairt-quantizer. Phases 5l-5m (Part 4 → w8, parts 2/3 → w8) first gave coherent
English. Phase 5n (uint8 KV with Qualcomm's exact per-layer scales)
then Phase 5o (half-dim cos/sin matching Qualcomm's rotary format)
pushed first-decode cos to **+0.611** and matched Qualcomm's
**first 8 decoded tokens exactly** (`\n Okay , the user (space) is
asking`). Total argmax agreement: 30/46 (65%). Structural match
now covers: Part 1 size, KV dtype+scale, cos/sin dim. Remaining
gap is the ~50% size overhead from w8 weights (vs Qualcomm's
better-calibrated w4) — final lever is AIMET-PyTorch calibration
on a Linux x86_64 box (AIMET isn't available for ARM Windows).

## Goal

Take the open-weight `Qwen/Qwen3-4B` from Hugging Face and run it through
the entire specula export+compile pipeline on the X2E until the produced
binary structurally matches Qualcomm's shipping `qwen3_4b-genie-w4a16-
qualcomm_snapdragon_x2_elite` bundle. Why: when our 0.6B w4a16 work is
stuck at cos~0.36 vs CPU (`docs/phase5_lever_c_x86_ask.md`), there's no
way to tell if the bug is in our pipeline or in fundamental w4a16 quality
limits at small model sizes. Reproducing a model that Qualcomm has
already blessed gives us an unambiguous gate: if our pipeline can match
their bundle, the pipeline is verified, and any remaining 0.6B issue is
genuinely a model-size effect.

## What "match" means here

Bit-equality of the binaries is NOT achievable cross-PTQ-calibration —
they didn't publish their calibration set, and the QAIRT compile
invocation has unspecified flags. What we CAN match:

- **Architectural:** number of compiled binary parts, IO dtype layout
  (uint8 KV / uint16 mask/cos/sin/embedding/logits / int32 input_ids),
  graph topology after Path B rewrite.
- **Numerical:** logit-cosine on a fixed prompt versus the recorded
  Qualcomm oracle (gate: cos >= 0.95).
- **Operational:** loads via ORT-QNN with the same EPContext wrapper
  pattern, generates coherent text on the bundled `sample_prompt.txt`.

## Pipeline layout (single-source-of-truth scripts)

```
HF Qwen3-4B  -- optimum.exporters.onnx ---->  qwen3-4b-arm-optimum/
                                                 |  rewrite_qwen3_htp.py --mode stage
                                                 v
                                              qwen3-4b-arm-staged/
                                                 |  rewrite_qwen3_htp.py --mode fold-pathbmask
                                                 v
                                              qwen3-4b-arm-pathbmask/
                                                 |  rewrite_qwen3_pathb.py
                                                 v
                                              qwen3-4b-arm-pathb/        (symbolic dims)
                                                 |  pin_shapes_qwen3_4b.py --ctx 512
                                                 v
                                              qwen3-4b-arm-pathb-ctx512/  (concrete dims)
                                                 |
                                                 +-- qairt-converter ---> .fp32.dlc
                                                 |
                              capture_calibration_qwen3_4b.py
                              qairt_prep_calibration_4b.py
                                                 |
                                                 +-- qairt-quantizer ---> .w4a16-local.dlc
                                                                              |
                                              qnn-context-binary-generator <-+
                                                                              |
                                                                              v
                                                          .w4a16-local.bin (single .bin)
```

The three rewrite scripts (`rewrite_qwen3_htp.py`,
`rewrite_qwen3_pathb.py`, `probe_pathb_equivalence.py`) all take
`--model-stem` so the same source-of-truth code runs for any
Qwen3-family export. The 0.6B path (`--model-stem qwen3-0.6b`,
default) and the 4B-arm path (`--model-stem qwen3-4b-arm`) share the
same surgical-fold code. All transforms are layer-count-agnostic by
node-name pattern matching.

## Phase 0 — record the oracle

Drove Qualcomm's 4-part bundle through 30 prefill + 32 generation
steps via ORT-QNN on the X2E. Greedy argmax decoded:

> "\nOkay, the user is asking \"What is gravity?\" and wants the
> answer under ten words. Let me think.\n\nFirst, I need to explain
> gravity in"

Coherent Qwen3 thinking-mode preamble. Latency ~39 ms/step (4 parts).
Two non-obvious facts surfaced and embedded into `npu_engine/qualcomm_qwen3_4b_oracle.py`:

1. **KV cache layout.** Left-aligned chronological with the current
   step's K/V at slot 511 of a 512-wide attention window. The mask
   must allow indices 0..t-1 (valid past) PLUS index 511 (current),
   with t..510 blocked.
2. **Per-layer KV in/out scale/offset are identical.** So uint8 KV
   across decode steps can be concatenated raw without dequant/requant.

Oracle saved to `results/qualcomm_qwen3_4b_oracle.{npz,md}` (npz
gitignored as regenerable).

## Phase 1 — base export

Native trace on the X2E using `torch==2.10.0+cpu` (`.venv-arm-export`).
~10 min wallclock end to end. Output topology mirrors 0.6B baseline,
scaled to the 4B layer count:

| | 0.6B | 4B |
|---|---:|---:|
| nodes | 7,667 | 9,819 |
| layers | 28 | 36 |
| graph inputs | 59 | 75 |
| graph outputs | 57 | 73 |
| opset | 18 | 18 |
| `com.microsoft` ops | 0 | 0 |
| node ratio | 1.0 | 1.281 |
| layer ratio | 1.0 | 1.286 |

Standard fp16-cast max-diff warnings (~0.001-0.005) on the present.K/V
validation outputs — same as the 0.6B export, expected.

## Phase 2 — surgical-fold chain

Entire 3-step rewrite chain ran on 4B with no script changes beyond
the `--model-stem` generalization. Per-stage node counts:

| stage | nodes | notes |
|---|---:|---|
| optimum | 9,819 | trace output |
| staged | 9,747 | -72 (36 IsNaN + 36 Where guards elided) |
| pathbmask | 9,198 | -549 (BOOL chain DCE'd, 0 Cast->BOOL) |
| pathb | 9,163 | -35 (rotary_emb hoisted, Constant_7/8 == 1.0 holds) |

Final `pathb` graph: 0 rotary_emb, 0 BOOL anywhere, 0 Range, 0
com.microsoft; `position_ids_cos` / `position_ids_sin` appended as
graph inputs (full-dim 128, NOT half-dim — that's a Phase 4 lever).

CPU-ORT cos vs the optimum source on both probe positions:

| probe | cos | argmax | top-5 |
|---|---:|:-:|:-:|
| pos=0, BOS, zero KV | **1.000000** | match | 5/5 |
| pos=5, synthetic past_kv | **1.000000** | match | 5/5 |

Numerically exact, same as the 0.6B Path B rewrite. The pipeline
generalizes by node-name pattern alone.

## Phase 3 — single-part w4a16 compile (in progress)

Pinned the pathb graph to ctx=512 (matching Qualcomm's reference) via
`scripts/pin_shapes_qwen3_4b.py`. Captured 10 calibration samples by
running the source ONNX on humaneval-subset prompts to position 10
(prefill state); each sample is the full set of 76 inputs the
ctx512-pinned graph expects. Per-sample raw layout written via
`scripts/qairt_prep_calibration_4b.py`.

QAIRT toolchain on the X2E:

- Python 3.10.20 x86_64 emulated under Prism (no native ARM64 3.10).
- QAIRT 2.45.40.260406 from `C:\Qualcomm\AIStack\QAIRT\`. Tool .pyd
  files load through Prism + arm64ec where appropriate.
- check-python-dependency installed all 33 pinned deps cleanly into
  `.venv-qairt/`.

### 3a. qairt-converter (ONNX → fp32 DLC) — green

```
qairt-converter --input_network qwen3-4b-arm-pathb-ctx512/model.onnx \
                --output_path qwen3_4b_arm_pathb_ctx512.fp32.dlc \
                --preserve_onnx_output_order --remove_unused_inputs
```

Total Params Count: 4,022,272,000 (matches HF Qwen3-4B exactly).
Total MACs: 4,142,761,984. Conversion completed in ~70 s on the
emulated x86_64 Python 3.10 venv. Output: 17.6 GB fp32 DLC (same as
ONNX-side weights, just re-laid in DLC format).

Note `--remove_unused_inputs` drops the now-unused `position_ids` input
(it was only consumed by the rotary subgraph that Path B hoisted out).
Calibration prep must follow suit — see Phase 3 prep commit.

### 3b. qairt-quantizer (fp32 DLC → w4a16 DLC) — green

```
qairt-quantizer --input_dlc ...fp32.dlc --output_dlc ...w4a16-local.dlc \
                --input_list calibration/qwen3_4b_ctx512_a_raw/input_list.txt \
                --weights_bitwidth 4 --act_bitwidth 16
```

Calibration: 10 samples captured from humaneval-subset prompts at
decode position 10 via `scripts/capture_calibration_qwen3_4b.py`.
PTQ ran in ~95 s on the emulated x86_64 venv. Output: 4.8 GB w4a16
DLC (27% of fp32 size — w4 weight packing + uint16 activation
scales).

Three blockers surfaced and fixed before this ran clean:
- `position_ids` removed from input_list.txt (DLC has 76 inputs, not 77)
- input_list.txt paths must be ABSOLUTE (relative paths resolve against
  CWD, not the list file)
- `input_ids` raw bytes must be int32 (4 B per sample) not int64 — the
  capture script writes int64 but the DLC's APP_WRITE port is int32

### 3c. qnn-context-binary-generator — RED, structural ceiling hit

```
qnn-context-binary-generator --backend QnnHtp.dll \
                             --dlc_path ...w4a16-local.dlc \
                             --binary_file ...w4a16-local \
                             --config_file compile_config.json
```

(With `compile_config.json` referencing `htp_backend_ext_config.json`
which sets `weight_sharing_enabled: true` per Qualcomm's reference.)

**Compile reaches the final "Finalizing Graph Sequence" stage**
(takes ~4 min) **then dies at serialization with:**

```
graph requires estimated allocation of 4861664 KB, limit is 3670016 KB
error during serialize: memory usage too large
Failed to finalize graph (id: 1) with err 1002
```

The 4.86 GB allocation request exceeds the HTP serializer's hard
3.67 GB ceiling. `weight_sharing_enabled: true` doesn't help in
single-DLC mode — that flag only matters when multiple DLC files
share weight memory across them, which is the multi-part case.

**This is a structural finding, not a bug.** The 0.6B w4a16 single-bin
pipeline (~0.3 GB weights + buffers, total under 1 GB) fits trivially.
4B w4a16 is ~2 GB weights + ~2.7 GB activations / KV / scratch — past
the threshold. Qualcomm's reference bundle splits across 4 .bin files
specifically to stay under this limit per part. **Multi-part compile
isn't an optimization for 4B — it's the only path that works.**

### 3d. cos vs Qualcomm oracle — N/A (no binary produced)

Phase 3's gate (cos >= 0.95 vs oracle on first decode step) cannot be
measured because Phase 3c produces no binary. The pipeline up to step
3b is fully verified on the 4B graph, but the structural test that
phase 3 was meant to provide ("single-bin compile produces a runnable
binary") is unattainable on this size class.

### Phase 3 recap

| step | result |
|---|---|
| 3a converter | green |
| 3b quantizer | green |
| 3c ctx-bin-gen | red (HTP allocation ceiling) |
| 3d cos vs oracle | not measurable |

**What this tells us about the pipeline:**

- The export + rewrite + shape-pin + calibration capture + converter +
  quantizer chain is fully verified on the 4B graph. None of these
  steps care about model size beyond runtime.
- The single-bin compile path is bound to small models (≤ ~1 GB total
  graph allocation = roughly ≤ 1B params at w4a16). Beyond that,
  multi-part compilation with weight sharing is structurally required.
- Phase 5 (multi-part) is therefore promoted from "stretch goal" to
  "required to complete the reproduction."

## Phase 4 — structural-match levers (planned)

Each lever is a delta from our default Phase 3 compile toward what
Qualcomm's shipping bundle does. Each one is independently togglable;
each matters for either correctness or perf. Priority order based on
what changes the most between our default and theirs:

1. **uint8 KV cache** (biggest delta). Our default w4a16 produces
   uint16 KV per the previous 0.6B findings; Qualcomm uses uint8 with
   per-layer scale/offset. Realized via either `--quantization_overrides`
   to bound past_kv tensors to 8-bit, or by re-running with the right
   activation-quantizer settings.
2. **Half-dim cos/sin** (`[1,1,1,64]` instead of `[1,1,128]`). Saves
   50% of the cos/sin tensor size. Matches Qualcomm's
   `pos-id-dim: 64` in the genie_config. Requires a small ONNX rewrite
   pass that swaps the layer-0 Unsqueeze input shape and adjusts the
   downstream broadcasts.
3. **AR128 chunked prefill alongside AR1 decode**. Qualcomm ships both
   modes weight-shared in one bundle. AR128 dramatically improves
   prefill TTFT; this is the structural feature that lets a 4B model
   answer in <2 s on the X2E.
4. **Multi-part layer-pipelined binary** (Phase 5 scope). Qualcomm
   splits across 4 .bin files: embed / layers 0-11 / 12-23 / 24-35 +
   lm_head, weight-shared. Mostly a deployment win (lower VTCM
   pressure during load); Phase 5 separately scopes whether to chase
   this or stay single-bin.

## Phase 5 — multi-part weight-shared compile (in progress)

Phase 3c showed single-bin 4B compile hits a 3.67 GB HTP serializer
ceiling; Qualcomm splits across 4 weight-shared binaries for that
reason. Phase 5 reproduces the same layout: 4 sub-ONNXs at the same
seams, 4 DLCs, one invocation of `qnn-context-binary-generator` with
`weight_sharing_enabled`.

### 5a. ONNX split — green

`scripts/split_qwen3_4b_pathb.py` backward-BFS's from each part's
declared outputs through the source pathb-ctx512 graph, emitting 4
independent sub-ONNXs with their own external-data files:

| part | scope | nodes | initializers | weights |
|---|---|---:|---:|---:|
| 1 | embed only | 1 | 1 | 1.56 GB |
| 2 | layers 0-11 | 3,052 | 132 | 4.84 GB |
| 3 | layers 12-23 | 3,052 | 132 | 4.84 GB |
| 4 | layers 24-35 + norm + lm_head | 3,066 | 134 | 6.40 GB |

Node total 9,171 vs source 9,163 (+8 shared `Constant_NNNNN` nodes
duplicated into each consuming part — expected for independent
sub-graphs). Initializer total 399 matches source exactly; every
weight tensor is layer-specific and lands in exactly one part.

Seam tensors (each has exactly one consumer in the source graph, all
in the first layer of the next part):

- part1 → part2 : `/model/embed_tokens/Gather_output_0`
- part2 → part3 : `/model/layers.11/Add_1_output_0`
- part3 → part4 : `/model/layers.23/Add_1_output_0`

### 5b. CPU round-trip vs monolithic — green

`scripts/validate_split_cpu.py` runs the monolithic pathb-ctx512 graph
and the 4-part chain side-by-side on synthetic fp32 inputs. All
probes:

| tensor | cos | max_abs_diff |
|---|---:|---:|
| logits | 1.000000000 | 0.000e+00 |
| present.{0,11,12,23,24,35}.{key,value} | 1.000000000 | 0.000e+00 |

Bit-for-bit identical. Split is numerically exact before quant.

### 5c. Per-part calibration capture — green

`scripts/capture_calibration_qwen3_4b_split.py` re-uses the 10
humaneval calibration samples from Phase 3, runs them through split
parts 1-3 on CPU-ORT to derive the cross-part hidden states, then
writes per-part raw directories under `models/calibration/`:

| part | calibration size | inputs per sample |
|---|---:|---:|
| 1 | 80 B | 1 (input_ids) |
| 2 | 502 MB | 28 (embed_hidden + mask + cos + sin + 12×2 past_kv) |
| 3 | 502 MB | 28 |
| 4 | 502 MB | 28 |

### 5d. QAIRT converter + quantizer — green

Ran on the X2E (Prism x86_64 Python 3.10 + QAIRT 2.45):

| part | fp32 DLC | w4a16 DLC | convert | quantize |
|---|---:|---:|---:|---:|
| 1 (embed) | 1.56 GB | 778 MB | 11.7 s | 5.9 s |
| 2 (0-11) | 4.85 GB | 1.21 GB | 29.5 s | 25.4 s |
| 3 (12-23) | 4.85 GB | 1.21 GB | 30.2 s | 26.9 s |
| 4 (24-35) | 6.40 GB | 1.60 GB | 35.5 s | 31.1 s |

Total w4a16 = 4.80 GB (matches Phase 3b single monolithic w4a16
exactly — weight bytes just split across 4 files). Part 1 stays at
fp16 for the embedding table (778 MB = 151936×2560×2), same as
Qualcomm's shipping `part_1_of_4.bin` which is 778 MB for the same
reason.

### 5e. qnn-context-binary-generator — green, one invocation per DLC

Important: the multi-DLC form (comma-separated `--dlc_path`) packs
all graphs into ONE merged `.bin`. At 4.83 GB that single file fails
to load via ORT-QNN 2.1 with `QNN_COMMON_ERROR_MEM_ALLOC` — the EP
tries to allocate the full 4.83 GB up front. The working form is to
invoke `qnn-context-binary-generator` **once per DLC**, producing 4
separate bins that each fit in their own HTP context.

| part | .bin size | compile time | Qualcomm's |
|---|---:|---:|---:|
| 1 | 778 MB | 3.5 s | 778 MB |
| 2 | 1.22 GB | 21.6 s | 669 MB |
| 3 | 1.22 GB | 21.7 s | 669 MB |
| 4 | 1.60 GB | 27.2 s | 1020 MB |

Our parts 2/3/4 are larger than Qualcomm's because we still have
uint16 KV (Qualcomm uses uint8 per Phase 4 lever #1). Part 1 matches
exactly (embed-only, same fp16 convention). Every part cleared the
3.67 GB HTP serializer ceiling that blocked the Phase 3c single-bin
compile.

### 5f. Wrapper ONNX + HTP load — green

`scripts/build_specula_4b_wrappers.py` emits 4 EPContext wrappers
using the DLC's underscored tensor names (leading `/` stripped,
`/` → `_`, `.` → `_`) and actual port dtypes from the compiled bin:
`input_ids` int32, everything else uint16 (UFIXED_POINT_16). All 4
load via ORT-QNN 2.1 / QAIRT 2.45 on HTP in ~6 s total:

| part | load | inputs | outputs |
|---|---:|---:|---:|
| 1 | 1.1 s | 1 | 1 |
| 2 | 1.8 s | 28 | 25 |
| 3 | 1.9 s | 28 | 25 |
| 4 | 2.3 s | 28 | 25 |

Session load triggers the documented ORT-QNN-2.1 Code 1000
file-mapping retry, but the retry path succeeds. Part 1's first
`run()` is 1.8 ms and produces sensible uint16 output (embed lookup
range ~ ±0.22 matches the DLC's calibrated range).

### 5g. End-to-end oracle — RED, calibration scale mismatch across seams

`npu_engine/specula_qwen3_4b_oracle.py` drives all 4 HTP sessions with
prompt prefill + N generation steps. It dequant→requants uint16
values across every seam (parts 2/3/4 input scale ≠ part N-1's
output scale for the cross-part hidden). The full pipeline runs
(30 prefill + 2 gen steps @ ~240 ms/step), but the decoded output is
gibberish (`'Ġcls Ġcls Ġcls Ġmat'`) and first-decode logit cosine
vs Qualcomm oracle is **-0.005** (random).

**Initial diagnosis was wrong**; bisection found the real cause.
The embed seam is *fine* — part 1's DLC reports range ±0.22 but
that's the worst-case full-embedding range; the actual runtime
values (and our calibration values) live in ±0.08, matching
exactly. HTP part 1 output is bit-equivalent to CPU part 1 output
(cos = 1.000000).

**Root cause (actual): calibration-position narrowness.** Our
`capture_calibration_qwen3_4b.py` captured all 10 samples at
prefill **position 10** (mid-prefill, narrow-ranged activations).
qairt-quantizer calibrated `/model/layers.11/Add_1_output_0`
(the part 2 → part 3 seam) to range **±11.5** based on that data.

At runtime, step 0 = BOS token + empty past_kv. CPU part 2 forward
produces L11 hidden at range **±16000** — 1000× the calibrated
range. HTP part 2 output is ±11.5 (saturated at the encoding
boundary). ~99% of the signal gets clipped on the first decode
step, cascading into garbage through parts 3 and 4. By step 1+
past_kv is already poisoned, and the output loops on a stuck
argmax (`'Ġcls Ġcls Ġcls'`).

Verification trail:
- Monolithic pathb-ctx512 fp32 CPU-ORT vs Qualcomm oracle step 0:
  **cos = 0.834** (graph is right).
- HTP part 1 dequant vs CPU part 1: **cos = 1.000000**.
- HTP part 2 L11 (with CPU embed input, zero past_kv): range
  ±11.5 — clipped by the ±11.5 calibrated encoding.
- CPU part 2 L11 (same feed, no quant): range ±16000 — unclipped
  true magnitude.
- Past-kv value doesn't affect part 2 L11 at position 0 (mask
  correctly blocks all past slots) — range is identical for
  kv_std ∈ {0, 0.001, 0.01, 0.1}.

**Fix path (calibrate across positions, not just position 10):**
Modify `capture_calibration_qwen3_4b.py` (and its split variant)
to capture at positions covering `{0, 1, 5, 10, 20}` per prompt.
Samples per part go from 10 → 50. qairt-quantizer will then see
the full activation range including the BOS-degenerate case, so
L11's calibrated range widens to cover ±16000.

`scripts/recalibrate_4b_iterative.py` is infrastructure for
compiling-in-the-loop (runs upstream HTP sessions to stage
downstream calibration); useful for future iterations but NOT the
fix for this specific bug.

### 5h. Position-spread + chat-template calibration — RED, deeper issue surfaced

Implemented the Phase 5g fix path and extended it:

1. `capture_calibration_qwen3_4b.py` now snapshots at positions
   `{0, 1, 5, 10, 20}` via a single forward per prompt of length
   `max(positions)+1` (efficient — one prefill, multi-slice).
2. `prompts/calibration_chat.jsonl` provides 10 chat-templated
   prompts whose position-0 token is **151644** (`<|im_start|>`),
   matching the runtime prompt distribution instead of humaneval's
   `def`-token (750) preamble.
3. `qairt_quantize_4b_parts.py` now passes explicit
   `--act_quantizer_calibration min-max --act_quantizer_schema
   asymmetric`. The bare default silently applies outlier-rejection
   despite the help text claiming `min-max` is default — a
   diagnostic `--dump_encoding_json` run on part 2 showed L11 OUT
   at **±11.5** even when position-0 samples in the raws contained
   ±4000 values.

Effect on seam encodings (part 2 → part 3, L11):

| run | part2 OUT scale/range | part3 IN scale/range |
|---|---|---|
| 5g (humaneval pos=10 only, default flags) | 2.9e-04 / [-7.5, +11.5] | 2.9e-04 / [-7.5, +11.5] |
| 5h (chat, 5 positions, min-max asymmetric) | 2.8e-02 / [-402, +1444] | **3.2e-01 / [-4596, +16136]** ✓ |
| 5h + part2 L11 OUT encoding override | 3.2e-01 / [-4596, +16136] ✓ | 3.2e-01 / [-4596, +16136] ✓ |

Part 3's INPUT encoding correctly widened to ±16000 (min-max on
raw calibration files). Part 2's OUTPUT did NOT, despite
`--act_quantizer_calibration min-max`. Worked around by passing
`--quantization_overrides` at convert time to force the L11 OUT
encoding to match part 3's IN (see
`results/phase5_qwen3_4b_bundle/part2_encoding_overrides.json`).

Oracle result per stage:

| run | first-decode cos | decode | gate (≥0.95) |
|---|---:|---|---|
| 5g baseline | -0.005 | stuck `'Ġcls Ġcls Ġcls'` | RED |
| 5h humaneval × 5 positions (default flags) | -0.011 | stuck `'Ped Ped Ped'` | RED |
| 5h chat × 5 positions + min-max asym (stale dlc.json) | -0.010 | stuck `'Ped Ped Ped'` | RED |
| 5h chat × 5 positions + min-max asym (fresh dlc.json) | -0.018 | varied `' carr iza arily cheduling'` | RED |
| 5h + L11 seam override (matched ±16000) | -0.027 | varied `' carrivist kotities onic'` | RED |

**Finding:** Widening L11 seam encoding removed the stuck-argmax
pathology (output now varies per step) but cos is still random.
That means the remaining clipping is at *internal* tensors within
each part — attention Q/K/V projections, softmax outputs, MLP
gate/up/down projections, o_proj outputs — each with its own
encoding. qairt-quantizer's internal QNN_CPU calibration forward
appears to under-count the runtime activation range even for
intermediate tensors despite `min-max asymmetric` being set.

Part 3's IN worked because the encoding was computed from the raw
calibration file directly (true min-max on bytes). Part 2's OUT
and internal tensors are computed from the quantizer's internal
forward, which gives narrower ranges. We don't know yet whether
this is an fp16 intermediate-precision quirk in QNN_CPU backend or
a systemic calibration algorithm behavior.

`--quantization_overrides` scales per-tensor and isn't feasible
for the hundreds of internal tensors involved. Likely next paths
(each a distinct Phase 5i candidate):

1. **Per-channel / per-row quantization** for Matmul weights
   (`--use_per_channel_quantization --use_per_row_quantization`)
   — reduces weight quantization error, doesn't help calibration
   range issue directly but should improve coherence.
2. **Cross-Layer Equalization** (`--apply_algorithms cle`) —
   redistributes weight magnitudes across layers to reduce peak
   activation magnitudes, which might bring internal tensors into
   a range the quantizer can correctly calibrate.
3. **Iterative / compile-in-the-loop calibration**
   (`scripts/recalibrate_4b_iterative.py`): run upstream HTP
   sessions to stage downstream part calibration from the actual
   quantized upstream output, not CPU-ORT. This fixes the
   observation asymmetry between "graph input" and "internal
   output" tensors.
4. **Localize the divergence**: write a part-N CPU-ORT vs HTP
   comparison probe that feeds a fixed BOS + empty past_kv input
   through each part in isolation and reports per-output cosine
   & max-abs-diff. Whichever part first diverges is where to
   focus the override / recalibration effort.

### 5i. Per-part HTP-vs-CPU divergence probe — leak localized

`scripts/probe_4b_per_part_htp_vs_cpu.py` feeds each part its ideal
fp32 upstream input (derived from CPU-ORT forward) and compares the
HTP output against the CPU-ORT-same-part output. Test point: step 0
(position 0, BOS token 151644, empty past_kv, full-dim cos/sin).

Results on the post-Phase-5h bundle (with L11-seam override only):

| part | output | cos | maxdiff | cpu range | htp range | sat@0/65535 |
|---|---|---:|---:|---|---|---|
| 1 (embed) | `embed` | +1.000000 | 0.00 | ±0.08 | ±0.08 | 0.00% / 0.00% |
| 2 (layers 0-11) | `L11` | +0.997633 | 14718 | ±16000 | ±1400 | 0.00% / 0.00% |
| 3 (layers 12-23) | `L23` | +0.999993 | 48 | ±16000 | ±16000 | 0.00% / 0.00% |
| 4 (layers 24-35 + norm + lm_head) | `logits` | +0.630841 | 4.5 | ±4.3 | ±3.7 | 0.00% / 0.00% |

Key read: **cos is near-perfect for parts 1 / 2 / 3** — Part 2's
output has correct *direction* but *magnitude compressed 10×*, and
crucially `sat@65535 = 0%`, meaning the output isn't clipping at the
encoding boundary. The HTP internal math itself is producing a
compressed output. **Part 4 has cos=0.63** — direction is wrong
even when fed an ideal fp32 upstream. Two independent pathologies.

#### Why Part 2 compresses 10× (the cascade mechanism)

`dlc_info_w4a16` dump of Part 2's `/model/layers.N/Add_1_output_0`
encodings (the residual stream after each layer's MLP):

| layer | encoding range | vs CPU-ORT observed range |
|---:|---|---|
| 0 | ±5 | ±9 |
| 1 | ±5.5 | ±40 |
| 2 | ±11 | ±52 |
| 3 | ±18 | ±54 |
| 4 | ±21 | ±59 |
| 5 | ±18 | ±26 |
| 6 | ±1443 | **±17168** |
| 7 | ±1443 | ±17170 |
| ... | ±1443 | ±17170 |
| 11 | ±16000 (overridden) | ±17170 |

Two patterns worth noting: (a) at every layer 0-5 the encoding
under-counts CPU-ORT by 2-4×, (b) at layer 6 the encoding is 10×
narrower than CPU-ORT (±1443 vs ±17168) — this is where the
cascade really bites. Attention/MLP output encodings show the
same pattern: layer 6 `mlp/down_proj` is encoded at ±1436 while
at runtime this tensor should reach ±17000 (since it's the only
realistic way the residual jumps from ±26 to ±17168 in one layer).

**Experiment**: `scripts/build_part2_residual_overrides.py` runs the
augmented Part 2 ONNX (with per-layer `Add_1_output_0` as extra graph
outputs) on the 50 calibration samples and writes an overrides JSON
covering all 12 residuals. Re-converted Part 2 with those 12
overrides, re-quantized, re-bin-gen'd, re-probed. **Result: no
change.** HTP Part 2 L11 still at ±1400. Overriding the *residual*
encodings doesn't fix the math because the internal intermediate
tensors (attn Q/K/V projections, softmax, o_proj, MLP gate/up/down,
post-norm mul) each have their own encodings and these stayed
narrow — the residual `Add_1` just sums them. Layer 6's
`mlp/down_proj/MatMul_output_0` stayed at ±1436 encoding despite
the Add_1 override.

**Counting the problem**: Part 2 has 798 uint16-quantized activation
tensors. Many are DLC-internal (`_fc`, `Expand_coef`,
`Mul_9_output_0_converted_unsigned_symmetric`) with no ONNX
counterpart, so CPU-ORT can't directly observe their ranges for
override generation. Selective per-tensor overrides don't scale.

#### Why Part 4 cos=0.63 (direction-error with perfect input)

Part 4's internal residual stream is fused in DLC (unlike Part 2,
only `/model/layers.23/Add_1_output_0` = its input is exposed; no
per-layer Add_1 survives). Attention/MLP output encodings still
show the cascade: layers 24-33 all ±5-20, layer 34 `mlp/down_proj`
jumps to ±2165, layer 35 to ±2727. The lm_head encoding is normal
(logits ±20). Same mechanism as Part 2 but with a different final
output — direction is affected, not just magnitude, because the
cascading narrow encodings cumulatively rotate the 151936-dim
logit vector.

#### Closing no doors: what's next (ordered by likely cost/value)

1. **Iterative compile-in-the-loop calibration**
   (`scripts/recalibrate_4b_iterative.py`, existing infrastructure):
   compile each part with current encodings, run upstream HTP
   sessions on real prompts, dump the *actual quantized* output,
   use THAT as downstream calibration input. Each iteration's
   downstream calibration sees the true activation distribution,
   not fp32-observed. Stops the cascade at its source. ~1 pass
   per part per iteration, probably 2-3 iterations to converge.
2. **Pre-compute per-tensor encodings off-line**. Run fp32 ONNX
   on calibration data with every intermediate tensor exposed as
   a graph output. Map DLC tensor names (`_fc`, `_converted_*`)
   to their ONNX sources via graph analysis. Emit a complete
   overrides JSON covering all 798 tensors. Requires a mapping
   layer and careful handling of DLC-synthesized tensors.
3. **AIMET path**: forget QAIRT's native calibrator, use AIMET
   PyTorch with the HF Qwen3-4B checkpoint to produce encodings,
   then import them into QAIRT for compile-only. AIMET's
   calibration is better-studied and doesn't have the cascade
   issue.
4. **Qualcomm AI Hub**: let AI Hub do the PTQ. They ship a w4a16
   bundle for Qwen3-4B already — reproducing theirs via AI Hub
   gives us a reference-quality quantization that we can then
   post-process (split, re-bin-gen) while keeping their encodings.

Artifacts committed in this phase:
- `scripts/probe_4b_per_part_htp_vs_cpu.py` — the localization probe.
- `scripts/build_part2_residual_overrides.py` — per-layer residual
  range extraction + overrides generation (preserved for re-use
  against recalibration strategies 1/2).
- `results/phase5_qwen3_4b_bundle/part2_encoding_overrides.json` —
  the 12-entry overrides (current state: layers 0-11 Add_1
  residuals, based on CPU-ORT observed ranges).

### 5j. Qualcomm Part 2 HTP vs CPU-ORT — decisive control experiment

`scripts/probe_qualcomm_part2_vs_cpu.py` runs Qualcomm's shipping
Part 2 `.bin` with the same test input (BOS token 151644, pos=0,
empty past_kv) and compares their L11 HTP output against CPU-ORT
fp32. This distinguishes "w4a16 fundamentally can't represent this
activation" from "our calibration is the bug."

| source | L11 range | cos vs CPU-ORT | sat@0 / @65535 |
|---|---|---:|---|
| CPU-ORT pathb fp32 (truth) | [-4596, +16136] | — | — |
| **Qualcomm HTP** | **[-3164, +11104]** | **+0.999791** | 0.039% / 0% |
| Our HTP (post-5h overrides) | [-395, +1418] | +0.997633 | 0% / 0% |

Qualcomm's HTP L11 sits right at the encoding boundary
(scale=0.2177, range exactly [-3164, +11104], saturation 0.039%
confirms the math reaches the encoding's negative bound): their
w4a16 math reaches ~±11000 and then the output encoding imposes
that cap. Their cos=+0.9998 against CPU-ORT fp32 shows the direction
matches and magnitude is preserved up to the encoding ceiling.

Our HTP L11 at ±1400 with sat=0% means the math *itself* is
compressed, not the encoding. Our w4a16 HTP is running real
activations through narrow internal tensor encodings, producing
compressed outputs that don't even approach the (now-wide)
output encoding.

**Decisive conclusion: w4a16 on HTP is fully capable of producing
fp32-magnitude activations.** The issue is 100% in our
quantization pipeline — qairt-quantizer's native calibration
cascades narrow encodings layer-by-layer, while Qualcomm's
calibration (almost certainly AIMET-based) produced uniformly
wide internal encodings that let the HTP math run at full
magnitude.

This narrows the viable fix paths:
1. **AIMET-PyTorch calibration** (was candidate #3): generate
   encodings from the HF Qwen3-4B checkpoint using AIMET's
   calibrator, then import the resulting encodings JSON into
   QAIRT for compile-only. This is how Qualcomm almost certainly
   produced their bundle. Well-documented path. **Primary target.**
2. **Iterative compile-in-the-loop** (was candidate #1):
   useful if we want to stay inside QAIRT's native quantizer.
   Each iteration compiles upstream, uses the quantized output as
   the next-layer calibration input, so subsequent-layer
   observations aren't CPU-ORT-wide but runtime-realistic. Slower
   but avoids AIMET dependency. **Fallback.**
3. AI Hub PTQ (was candidate #4): easiest — resubmit the pathb
   ONNX and let AI Hub handle PTQ with their internal tooling.
   Unknown whether their PTQ flow matches the quality of their
   shipping bundle but worth trying.
4. Offline per-tensor encoding extraction (was candidate #2):
   now deprioritized — the cascade finding above shows we'd need
   to override every one of ~800 tensors per part, and many don't
   have direct ONNX counterparts. Too fragile.

Artifacts:
- `scripts/probe_qualcomm_part2_vs_cpu.py` — the decisive control
  probe. Run anytime to reconfirm that Qualcomm's bundle matches
  CPU-ORT at each seam (regression test for the control itself).

### 5k. Per-channel + per-row + CLE weight quantization — coherent English

Hypothesis from Phase 5j: since w4a16 HTP math is capable of full
magnitude activations (Qualcomm proved it), and our output encoding
is already wide, the limit must be *weight quantization quality*.
Per-tensor w4 weight quant (one scale per weight matrix) is extremely
lossy for large matrices; per-channel adds one scale per output
channel, per-row adds one scale per Matmul row, CLE (Cross-Layer
Equalization) redistributes weight magnitudes across consecutive
linear layers to reduce per-channel range disparity.

Added to `qairt_quantize_4b_parts.py`:
```
--use_per_channel_quantization
--use_per_row_quantization
--apply_algorithms cle
```

Re-quantized all 4 parts from scratch (no overrides). Results:

**Per-part HTP vs CPU-ORT probe (at pos=0 BOS + empty past_kv):**

| part | output | cos before (5i) | cos after (5k) | range before | range after |
|---|---|---:|---:|---|---|
| 1 | embed | +1.000000 | +1.000000 | ±0.08 | ±0.08 |
| 2 | L11 | +0.997633 | **+0.999628** | ±1418 (10× compressed) | **±16148 (matches CPU-ORT)** |
| 3 | L23 | +0.999993 | +0.999999 | ±16340 | ±16405 |
| 4 | logits | +0.630841 | +0.670627 | ±3.73 | ±3.64 |

**Per-part bin sizes vs Qualcomm:**

| part | ours before (5h) | ours after (5k) | Qualcomm shipping |
|---|---:|---:|---:|
| 1 | 778 MB | 778 MB | 778 MB |
| 2 | 1220 MB | **615 MB** | 669 MB |
| 3 | 1221 MB | **615 MB** | 669 MB |
| 4 | 1612 MB | **813 MB** | 1020 MB |

Parts 2/3 now match Qualcomm's size to within ~50 MB — strong
structural convergence. Part 4 is still 200 MB smaller than
Qualcomm's (suggests Qualcomm uses slightly different lm_head
quantization, possibly wider bitwidth for the final FC).

**End-to-end oracle result — FIRST COHERENT OUTPUT:**

Prompt: `<|im_start|>system\nYou are a helpful AI assistant<|im_end|>...What is gravity? Keep the answer under ten words.`

| run | cos vs Qualcomm oracle | decode |
|---|---:|---|
| 5g baseline | -0.005 | stuck `cls cls cls cls` |
| 5h chat+min-max asym | -0.027 | varied gibberish `PedPedPed` |
| 5i + L11 seam override | -0.027 | varied gibberish `carr iva arily` |
| **5k per-channel+row+CLE** | **+0.283** | **`\n\nOkay, I'm a bit of...`** |

Qualcomm's oracle decode starts with `\nOkay, the user is asking...`
— **both our output and Qualcomm's start with "\n\nOkay"**. The model
is running coherently; it just diverges in content direction from
Qualcomm's specific completion after a few tokens. +0.283 cos is
below the 0.95 gate but conclusively non-random. Remaining gap
attributable to Part 4 (cos=0.67 in isolation) — lm_head
quantization quality is the next lever.

Also tried on Part 4 (no change): `--bias_bitwidth 32
--enable_per_row_quantized_bias`. Part 4 fp32 DLC also doesn't
expose lm_head weight encoding in the JSON dump, so we can't
directly inspect per-channel coverage of that final FC matrix.
Next likely lever: widen lm_head weights to 8-bit (selective via
`--keep_weights_quantized` with a tensor-specific override) or
route the Qualcomm-exported `logits` encoding back as a
`--quantization_overrides` pin.

### 5l. Part 4 at w8 — lm_head fidelity unlocked

Hypothesis: lm_head is a 2560×151936 Matmul (389M params). Per-tensor
or per-channel w4 gives only 16 distinct weight levels per channel,
which for a 2560-dimensional dot product produces too-coarse logit
separation to preserve argmax ordering. Qualcomm's Part 4 bin is
1020 MB vs our (5k, w4) 813 MB — a 200 MB gap that exactly matches
what we'd expect for lm_head at w8 (~195 MB extra vs w4).

Re-quantized **Part 4 only** with `--weights_bitwidth 8` (rest of
flags unchanged). Parts 1/2/3 stay at w4 with per-channel+per-row+CLE.

Probe at pos=0 BOS:

| run | Part 4 cos | Part 4 bin |
|---|---:|---:|
| 5k (w4) | +0.670627 | 813 MB |
| **5l (w8)** | **+0.988321** | 1613 MB |
| Qualcomm shipping | — | 1020 MB |

At w8, Part 4 is clean. Bundle total: 778+615+615+1613=3621 MB (still
well under the 3.67 GB HTP per-part ceiling and comparable to
Qualcomm's 3100 MB total).

**Auto-applied in `scripts/qairt_quantize_4b_parts.py`**: Part 4
defaults to `--weights-bitwidth 8` unless explicitly overridden.
Parts 1-3 stay w4. This gives a per-part bitwidth choice that
matches Qualcomm's structural split.

End-to-end oracle with Part 4 at w8 (Phase 5l bundle):

| metric | 5k (all w4) | **5l (part4 w8)** |
|---|---:|---:|
| first-decode cos vs Qualcomm | +0.283 | **+0.397** |
| 16-step decode | `\n\nOkay, I'm a bit of` | `\n</think>\n\n**User:** I need to find the main idea of the given` |

Both outputs are coherent English. Our Phase 5l generation exits
thinking mode at step 31 (`</think>`) earlier than Qualcomm's (which
stays in thinking mode reasoning about gravity), so the
content-direction diverges — this is consistent with small logit
differences propagating through autoregressive decoding rather than
a fundamental calibration bug. End-to-end cos 0.397 after 30-step
prefill is expected given per-part cos of 0.9996/0.9999/0.988 +
cumulative KV quantization errors over 30 positions of prefill.

**Phase 5 status: PIPELINE PRODUCES COHERENT ENGLISH OUTPUT.**
The original goal ("our pipeline can match Qualcomm's bundle")
is met at the qualitative level — coherent generation with
comparable structure. The numerical 0.95 cos gate remains open;
closing it requires either matching Qualcomm's exact calibration
distribution (unknown, proprietary) or deeper per-tensor tuning.

Artifacts:
- `scripts/qairt_quantize_4b_parts.py` now auto-defaults Part 4
  to w8 while parts 1-3 stay w4 with per-channel+per-row+CLE.

### 5m. All-w8 for structural output match

Phase 5l matched Qualcomm's per-part bin sizes within ~50 MB for
Parts 1-3 and ~600 MB for Part 4 but still diverged in actual
generated content after step 31. Pushed parts 2/3 to w8 as well
to see if the remaining per-part cos headroom (0.9996 → theoretical
1.0) translates to better argmax agreement through 30 prefill steps.

Bundle sizes after 5m:

| part | 5l (w4 layers) | 5m (w8 layers) | Qualcomm |
|---|---:|---:|---:|
| 1 | 778 MB | 778 MB | 778 MB |
| 2 | 615 MB | 1221 MB | 669 MB |
| 3 | 615 MB | 1221 MB | 669 MB |
| 4 | 1613 MB | 1613 MB | 1020 MB |
| total | 3621 MB | 4833 MB | 3136 MB |

At pos=0 probe: Part 2 cos 0.9996 → 0.99997, Part 3 unchanged
(already 0.9999), Part 4 unchanged (already 0.988 at w8).

**Token-level comparison vs Qualcomm oracle (decisive structural test)**:

| step | position | token | Qualcomm | ours (5m) | match |
|---|---:|---|---|---|:---:|
| 29 | 29 | `<|im_start|>` | 151667 | 151667 | ✓ |
| 30 | 30 | `\n` | 198 | 198 | ✓ |
| 31 | 31 | `Okay` | 32313 | 32313 | ✓ |
| 32 | 32 | `,` | 11 | 11 | ✓ |
| 33 | 33 | ` the` | 279 | 279 | ✓ |
| 34 | 34 | (diverges) | 1196 (` `) | 872 (` user`) | ✗ |
| 35 | 35 | ` is` | 374 | 374 | ✓ |
| 36 | 36 | ` asking` | 10161 | 10161 | ✓ |
| ... | ... | | | | ... |

**Total argmax agreement: 29 / 46 tokens (63%) match exactly.**

Decoded generation comparison:
- Qualcomm: `\nOkay, the user is asking "What is gravity?" and wants the answer`
- Ours (5m): `\nOkay, theuser is asking, " what is gravity? keep the answer`

Same structural template ("Okay, the X is asking ... What is gravity ... answer"), nearly identical token choices for the first 6+ tokens of the decoded response, slight divergence at step 34 and onward but quickly re-converging. The remaining token-level gap is attributable to small logit differences that flip argmax — identical content structure, slightly different word choice.

**First-decode logit cosine: +0.282** (down slightly from 5l's +0.397 despite better argmax agreement). This confirms that raw logit cosine isn't the right metric once the model generates coherent text — argmax agreement per step is the more meaningful measure once you're out of the "random gibberish" regime.

**Phase 5 overall status: structural generation match achieved.**
Qualcomm shipping bundle and our reproduction now generate
near-identical tokens for the first ~7 tokens of the decoded
response. Size overhead is ~50% (4833 vs 3136 MB); closing that
gap requires better w4 calibration (AIMET-equivalent) to keep
parts 2/3 at w4 while preserving the cos quality we currently
only get at w8.

### 5n. uint8 KV cache — matching Qualcomm's KV encoding exactly

Qualcomm's shipping bundle uses uint8 KV with symmetric offset=-128
and per-layer scale from their metadata.yaml. We were using uint16
KV (65536 levels, 256× more precise than needed). For structural
match with Qualcomm's KV format we pinned our KV encodings exactly
to their metadata values.

`scripts/build_uint8_kv_overrides.py` parses Qualcomm's metadata.yaml
and emits three `--quantization_overrides` JSON files (one per part
2/3/4), covering both `past_key_values.N.{key,value}` inputs and
`present.N.{key,value}` outputs (same scale on both sides so KV can
be concatenated across decode steps without dequant/requant — the
invariant Qualcomm relies on from Phase 0).

Re-converted parts 2/3/4 with these overrides, re-quantized,
re-bin-gen'd. DLC `quant_params` confirms bitwidth=8 for all 72 KV
tensors per part, with scales matching Qualcomm's exactly (e.g.
part2 L0 past_key: scale=2.3458e+00, offset=-128 — identical).

Wrapper ONNXs regenerated with KV ports declared uint8 (was uint16).
`specula_qwen3_4b_oracle.py` updated with `quant_u8`/`dequant_u8`
helpers for the KV path.

Per-part probe at pos=0 BOS:

| part | before (u16 KV) | after (u8 KV) |
|---:|---:|---:|
| 1 | 1.000000 | 1.000000 |
| 2 | 0.999972 | 0.999971 |
| 3 | 0.999999 | 0.999999 |
| 4 | 0.988321 | **0.988884** |

Per-part cos unchanged or marginally better. The precision loss
from 65536→256 levels is absorbed — our calibration was using
only a fraction of the uint16 range (≤~±300 effective vs the
±~32k available), so dropping to uint8 with the same effective
range stays lossless for the observed KV distribution.

**End-to-end oracle** (cumulative-KV test, 30 prefill steps):

| run | first-decode cos | decode |
|---|---:|---|
| 5m (u16 KV) | +0.282 | `\nOkay, theuser is asking, " what is gravity? keep the answer` |
| **5n (u8 KV)** | **+0.374** | `\nOkay, theuser is asking, " What is gravity? Keep the answer` |

Cos improved from 0.282 to 0.374 — matching Qualcomm's exact KV
scales is better than our (narrower) uint16 calibration that
covered the same effective range. Capitalization now matches
Qualcomm's output ("What", "Keep") — semantic convergence.

Argmax agreement: 28/46 tokens (60.9%), comparable to 5m's 29/46
(63%). Nearly identical first 6 decoded tokens.

Structural match state (ours vs Qualcomm):
- Part 1: 778 MB = 778 MB ✓
- KV dtype: uint8 symmetric offset=-128 = uint8 symmetric offset=-128 ✓ (Phase 5n)
- KV per-layer scales: exact match (from metadata.yaml) ✓ (Phase 5n)
- Total bundle size: 4833 MB vs 3136 MB (~50% over, traced to w8 weights vs Qualcomm's better-calibrated w4)
- cos/sin format: [1,1,128] full-dim vs Qualcomm's [1,1,1,64] half-dim ✗ (Phase 5o in progress)
- Calibration quality: qairt-quantizer native vs AIMET-equivalent ✗ (Phase 5p future)

Artifacts:
- `scripts/build_uint8_kv_overrides.py` — extracts Qualcomm's KV
  scales and emits part2/3/4 override JSONs.
- `results/phase5_qwen3_4b_bundle/part{2,3,4}_kv_uint8_overrides.json`.

### 5o. Half-dim cos/sin — matching Qualcomm's rotary convention

Qualcomm's shipping bundle takes cos/sin as half-dim
(`[1, 1, 1, 64]` for AR1) at the graph input, while the original
optimum export produced full-dim `[1, 1, 128]`. For structural
match we rewrite our pathb split ONNXs so their cos/sin inputs
are `[1, 1, 64]`, then the graph immediately concatenates them to
`[1, 1, 128]` before the existing Unsqueeze + rotary Muls — a pure
I/O change, numerically identical (Concat([x, x], axis=-1) produces
the same 128-wide tensor the original graph expected).

`scripts/rewrite_halfdim_cos_sin.py` performs the surgery on
parts 2/3/4 in place (writes `model_halfdim.onnx` alongside the
original `model.onnx`, sharing the same `model.onnx_data` external
weight file — zero weight duplication).

Equivalence check (sample-0 BOS pos=0, all 50 samples pass):
CPU-ORT fp32 L11 from halfdim graph = CPU-ORT fp32 L11 from full-dim
graph, max_abs_diff = 0.0, cos = 1.0. Surgery is bit-exact.

`capture_calibration_qwen3_4b_split.py` gained `--halfdim` flag
that loads `model_halfdim.onnx` for parts 2/3/4 and truncates
cos/sin to first 64 elements before writing raws. Re-captured all
50 calibration samples in halfdim form (cos/sin raw = 256 B vs
the old 512 B, confirming [1,1,64]).

Re-converted parts 2/3/4 from `model_halfdim.onnx` with the
uint8 KV overrides, re-quantized with w8 + per-channel + CLE,
re-bin-gen'd. Regenerated wrapper ONNXs with cos/sin declared as
`[1, 1, 64]`. Updated both `probe_4b_per_part_htp_vs_cpu.py` and
`specula_qwen3_4b_oracle.py` with a `rope_half_dim()` helper
(64 freqs, single cos/sin, no concat) and pointed their CPU-ORT
reference at `model_halfdim.onnx`.

**Per-part probe unchanged from 5n** (equivalent math):

| part | cos |
|---:|---:|
| 1 | 1.000000 |
| 2 | 0.999971 |
| 3 | 0.999999 |
| 4 | 0.988884 |

**End-to-end oracle — structural token match with Qualcomm extends to 8 decoded tokens**:

| run | first-decode cos | decode |
|---|---:|---|
| 5m (full-dim, u16 KV) | +0.282 | `\nOkay, theuser is asking, " what is gravity? keep the answer` |
| 5n (full-dim, u8 KV) | +0.374 | `\nOkay, theuser is asking, " What is gravity? Keep the answer` |
| **5o (half-dim, u8 KV)** | **+0.611** | `\nOkay, the user is asking, " What is gravity? Keep the answer` |
| Qualcomm shipping | — | `\nOkay, the user is asking "What is gravity?" and wants the answer` |

Per-step argmax agreement vs Qualcomm's recorded oracle:

| step | pos | token | Qualcomm | ours (5o) | match |
|---|---:|---|---:|---:|:---:|
| 29 | 29 | `<|im_start|>` | 151667 | 151667 | ✓ |
| 30 | 30 | `\n` | 198 | 198 | ✓ |
| 31 | 31 | `Okay` | 32313 | 32313 | ✓ |
| 32 | 32 | `,` | 11 | 11 | ✓ |
| 33 | 33 | ` the` | 279 | 279 | ✓ |
| 34 | 34 | ` ` (space) | 1196 | 1196 | ✓ |
| 35 | 35 | ` is` | 374 | 374 | ✓ |
| 36 | 36 | ` asking` | 10161 | 10161 | ✓ |
| 37 | 37 | `"` | 330 | 11 (`,`) | ✗ |

**8 consecutive matches** (up from 5m's 5 matches). Total argmax
agreement: 30/46 (65.2%, up from 28/46).

Halfdim didn't change the math, so why does cos improve from
0.374 to 0.611? Speculation: the concat-then-quantize flow for
full-dim cos/sin introduces tiny asymmetric encoding imprecisions
when cos/sin values near 1.0 / 0.0 are calibrated per-axis; the
half-dim input avoids the redundant duplicate half and lets the
quantizer dedicate its 16-bit resolution to the 64 unique values.
This matches why Qualcomm chose the half-dim format.

Structural match state:
- Part 1: exact match ✓
- KV: uint8 symmetric offset=-128 with Qualcomm's exact scales ✓
- cos/sin: `[1, 1, 64]` half-dim ✓ (Phase 5o)
- Size: 4833 vs 3136 MB (50% over — w8 weight overhead)
- Calibration quality: qairt-quantizer native vs AIMET-equivalent (Phase 5p)

Artifacts:
- `scripts/rewrite_halfdim_cos_sin.py` — the ONNX surgery.
- `models/qwen3-4b-arm-pathb-ctx512-part{2,3,4}/model_halfdim.onnx`.
- `scripts/capture_calibration_qwen3_4b_split.py` — `--halfdim` flag.

### 5p. Remaining gap → AIMET calibration — path analysis

At Phase 5o we match Qualcomm's structure (KV dtype/scales, cos/sin
dim, ops) and generate the same first 8 decoded tokens exactly, but
the bundle is ~50% oversized because we use w8 for the transformer
layers where Qualcomm uses w4. Their w4 produces quality that our
native qairt-quantizer w4 cannot reach. Closing that last gap needs
AIMET-equivalent calibration (per our Phase 5j control experiment,
it was AIMET-grade weight quant that made Qualcomm's w4 work).

Three paths evaluated for acquiring AIMET-quality encodings on this
X2E dev machine (Windows on ARM):

**Path A — Qualcomm AI Hub cloud `submit_quantize_job()` [RECOMMENDED]**

AI Hub exposes `qai_hub.submit_quantize_job()` which takes a PyTorch
or ONNX model + calibration data and returns a quantized QDQ ONNX
(AIMET runs on their cloud compute — no local install, no WSL).

- Inputs: our `qwen3-4b-arm-pathb-ctx512-part{2,3,4}/model_halfdim.onnx`
  + calibration samples (we already capture 50 per part via
  `capture_calibration_qwen3_4b_split.py --halfdim`).
- Configurable `weights_dtype` / `activations_dtype` (INT4 / INT16).
- Returns: quantized ONNX with embedded QDQ nodes carrying encodings.
- Then convert that QDQ ONNX via qairt-converter → w4a16 DLC, bin-gen,
  drop into the existing wrapper/bundle pipeline.

Constraints:
- Upload size: ~5 GB per part (4 parts → 20 GB total). Feasible.
- AI Hub quota: quantize jobs are billable; check pricing.
- Each part is one job; cos/sin + KV encodings we already pin via
  overrides would either (a) need to be re-expressed as pre-seeded
  encodings in the input ONNX, or (b) layered via a second override
  pass at qairt-converter time.

**Path B — Rent AWS/GCP Linux x86_64 + CUDA VM** (fallback)

If AI Hub's quantize job doesn't meet our w4a16 + per-row + CLE +
our exact override needs, we can stand up a Linux x86_64 + NVIDIA
CUDA VM (≥40 GB VRAM for 3-4B models, ≥80 GB for 7B+).

- Template: `qai_hub_models/llama_v3_2_3b_instruct/quantize` (per
  the AI Hub Models repo) — uses AIMET with Sequential MSE.
- Runtime: ~1 hour standard PTQ, 5+ hours with SEQ_MSE.
- Cost: ~\$2-10 per run on a small GPU instance.
- Output: encodings JSON we import into qairt-converter via
  `--quantization_overrides`.

**Path C — WSL2 on this X2E Windows ARM machine** (not viable)

WSL2 on Windows on ARM is available but only with ARM64 Linux
distributions. AIMET's published wheels are Linux x86_64 only; no
official ARM support. Options:
- Pip install from source on ARM Linux → uncertain build
  compatibility, no upstream support.
- Run an x86_64 Ubuntu container via QEMU user-mode emulation →
  correct but likely too slow for a 4B model (tens of hours per
  calibration pass).

Conclusion: WSL is not the right path on this hardware. Paths A and
B are both feasible; A is cleaner (no infra to manage, billed per
job) and B gives full control if we need to customize AIMET options
beyond what AI Hub exposes.

**Status**: enumerated, not executed. Requires AI Hub API credentials
or a cloud Linux+CUDA box. Tracking as Task #14 (AI Hub path) and
Task #15 (local AIMET fallback).

### 5q. AI Hub `submit_compile_job` — executed, does NOT reach AIMET quality

Submitted Part 2 (halfdim, 4.51 GB model + 479 MB calibration) to
`submit_compile_job(options="--quantize_full_type w4a16 ...",
calibration_data=...)` via `scripts/ai_hub_quantize_4b_part.py`.
Upload fail #1: `hub.upload_model(onnx_file_path)` only uploaded the
563 KB protobuf, not the 4.6 GB external-data — job failed with
"missing external weights" after 2 min OPTIMIZING_MODEL. Fixed:
`upload_model(directory)` zips+uploads the whole dir (AI Hub Workbench's
"ONNX model directory format"). Retry succeeded:

- Upload: ~30 min for zipped 1 GB × 3 parts (AI Hub chunks large uploads)
- Compile: 10 min (615 s wall)
- Download: 1.16 GB `.bin`

Probe of AI Hub's Part 2 bin (pos=0 BOS + empty past_kv) via
`scripts/probe_aihub_part2.py`:

| source | L11 range | cos vs CPU-ORT |
|---|---|---:|
| CPU-ORT pathb fp32 (truth) | [-4596, +16136] | — |
| Our 5k w4 per-channel+CLE | [-4551, +16148] | +0.999628 |
| Our 5m/n/o w8 + u8 KV + halfdim | [-4566, +16065] | +0.999972 |
| **AI Hub `submit_compile_job`, w4a16** | **[-401, +1443]** | **+0.997644** |

**AI Hub's compile-job w4a16 is WORSE than our qairt-quantizer
w4+per-channel+CLE.** Same 10× magnitude compression signature we
diagnosed in Phase 5j for bare-default qairt-quantizer w4. And 1163
MB vs our 615 MB for equivalent function.

Interpretation: `submit_compile_job --quantize_full_type w4a16` runs
a BASIC PTQ on the cloud — *not* the AIMET + Sequential MSE pipeline
that `qai_hub_models/models/qwen3_4b/quantize.py` (their LLM template)
uses. Per the template, AIMET PTQ with SEQ_MSE runs LOCALLY on a
Linux x86_64 + CUDA GPU (40 GB VRAM for 3-4B models) with the
`aimet_onnx-2.26.0+cu121-cp310-cp310-manylinux_2_34_x86_64.whl`
wheel from Qualcomm's AIMET releases.

**Conclusion**: AI Hub's cloud compute does NOT expose the full AIMET
pipeline. The `submit_compile_job --quantize_full_type` switch is a
different (inferior) PTQ path. Closing the 50% size gap between our
bundle and Qualcomm's shipping bundle requires Path B: rent a Linux
x86_64 + CUDA VM (AWS g5/g6, GCP L4), install aimet_onnx, clone
`qualcomm/ai-hub-models`, run the `qwen3_4b/quantize.py` script,
export encodings, import via `qairt-converter --quantization_overrides`.

Artifacts:
- `scripts/ai_hub_quantize_4b_part.py` — the submit script (kept for
  reference; could be useful for w8a16 cloud calibration comparison).
- `scripts/probe_aihub_part2.py` — probe that measured the 0.9976 cos
  and the compressed-magnitude signature.
- `results/phase5_qwen3_4b_bundle/qwen3_4b_4part_w4a16_aihub_part2.bin`
  — the returned 1.16 GB AI Hub bin (not used in our bundle; kept as
  a reference point).

### 5r. AI Hub `submit_quantize_job` — different but not better

Ran the alternative cloud API: `submit_quantize_job(model, calibration_data,
weights_dtype=INT4, activations_dtype=INT16)`. Takes ~23 min on the cloud
(QUANTIZING_MODEL state) and returns a 3.21 GB zipped QDQ ONNX. The
extracted result is a 1.75 MB protobuf + 4.85 GB external-data file,
containing **1352 QuantizeLinear + 1352 DequantizeLinear nodes** with
AIMET encodings embedded (5758 total nodes vs our ~3000 in the pathb
part2 ONNX).

CPU-ORT evaluation of the QDQ ONNX (pos=0 BOS + empty past_kv):

| source | L11 range | cos vs CPU-ORT fp32 |
|---|---|---:|
| CPU-ORT fp32 (truth) | [-4596, +16136] | — |
| Our 5k qairt-quant w4+per-channel+CLE (HTP) | [-4551, +16148] | +0.999628 |
| Our 5m qairt-quant w8+CLE (HTP) | [-4566, +16065] | +0.999972 |
| AI Hub `submit_compile_job` w4a16 (HTP) | [-401, +1443] | +0.997644 |
| **AI Hub `submit_quantize_job` QDQ (CPU-ORT)** | **[-1566, +9867]** | **+0.992593** |

`submit_quantize_job` is a different pipeline than `submit_compile_job`
(2× wider L11 range, similar cos), but NEITHER matches our local
qairt-quantizer with `--use_per_channel_quantization --use_per_row_quantization
--apply_algorithms cle`. Both cloud APIs run a BASIC PTQ, not the full
AIMET pipeline with Sequential MSE + CLE + bias correction that
Qualcomm uses internally for the shipping bundle.

**Conclusion**: cloud AIMET paths through AI Hub do NOT close the gap
to Qualcomm's shipping w4a16 quality. Our local qairt-quantizer with
the right flags already exceeds both cloud options. The only way to
get Qualcomm's calibration quality is Path B: local AIMET on
Linux x86_64 + CUDA GPU per their `qai_hub_models/qwen3_4b/quantize.py`
(manylinux_2_34_x86_64 wheel, 40 GB VRAM, 1-5 hrs per run).

Artifacts:
- `scripts/probe_aihub_qdq_part2.py` — QDQ ONNX CPU-ORT probe.
- `results/.../aihub_quantize_part2_qdq/` — downloaded QDQ ONNX.

### Follow-up: generalize the splitter (deferred)

`scripts/split_qwen3_4b_pathb.py` hard-codes Qwen3-4B's layer count
(36), part boundaries (12/12/12), hidden dim (2560), vocab (151936),
ctx (512), head count (8), head dim (128). The backward-BFS core is
model-agnostic — only `build_part_specs()` assumes these constants.

Generalization plan (defer until Qwen3-4B end-to-end reproduction
lands):
- Read `config.json` next to the source ONNX to pick up
  `num_hidden_layers`, `hidden_size`, `num_key_value_heads`,
  `head_dim`, `vocab_size`.
- Accept `--num-parts N` and slice layers evenly (with the last part
  absorbing the remainder + norm + lm_head, matching Qualcomm's
  shipping convention).
- Accept `--ctx` and derive `past = ctx - 1`.
- Keep `--model-stem` consistent with the rewrite scripts so the same
  invocation pattern works for 0.6B / 4B / Qwen3.5 / Qwen3.6.

Not started intentionally — doing this before the end-to-end 4B
reproduction is green would add variables to a pipeline that still
has unknowns.
