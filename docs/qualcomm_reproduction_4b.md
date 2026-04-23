# Reproducing the Qualcomm Qwen3-4B w4a16 reference bundle

**Status:** in-progress 2026-04-23. Phases 0-2 complete; Phase 3 hit the
single-bin HTP ceiling (structural, not a bug); Phase 5 converter +
quantizer + ctx-bin-gen + wrapper + HTP-load all green end to end.
Oracle runs but cos vs Qualcomm is ~0 (gibberish output) because the
per-part calibration is mis-matched across seams — see Phase 5g
below. Fix path: iterative calibration using upstream's quantized
output as downstream's calibration input.

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
Two non-obvious facts surfaced and embedded into `scripts/qualcomm_qwen3_4b_oracle.py`:

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

`scripts/specula_qwen3_4b_oracle.py` drives all 4 HTP sessions with
prompt prefill + N generation steps. It dequant→requants uint16
values across every seam (parts 2/3/4 input scale ≠ part N-1's
output scale for the cross-part hidden). The full pipeline runs
(30 prefill + 2 gen steps @ ~240 ms/step), but the decoded output is
gibberish (`'Ġcls Ġcls Ġcls Ġmat'`) and first-decode logit cosine
vs Qualcomm oracle is **-0.005** (random).

**Root cause (identified):** calibration-scale mismatch at the
seams. Part 1's output `/model/embed_tokens/Gather_output_0` was
calibrated by qairt-quantizer running part 1's own graph — with w4
weights, so the embed output range is ±0.22. Part 2's input
`/model/embed_tokens/Gather_output_0` was calibrated with the clean
fp32 CPU-ORT Gather output (which we pre-computed and wrote as raw
calibration), range ±0.08. At runtime, part 1 produces values in
±0.22; when we requant those fp32 values into part 2's ±0.08
encoding, ~60% of the dynamic range clips, destroying signal. Same
story at the L11 and L23 seams (each downstream calibrated against
the clean CPU-ORT output of the upstream, not the w4 output).

**Fix path (iterative calibration):**
1. Compile part 1 → w4a16.bin (already done).
2. Run part 1's HTP session over the 10 calibration samples to
   generate *quantized* embed_hidden outputs; write those as part 2's
   calibration input.
3. Re-quantize part 2 with the new calibration; compile to part 2's
   bin.
4. Run parts 1+2 to generate part 3's calibration; quantize +
   compile part 3. Repeat for part 4.

This requires a small new script
(`scripts/capture_calibration_qwen3_4b_split_iterative.py` or an
extension to the existing split capture) that invokes ORT-QNN
sessions on upstream `.bin` files to stage each downstream part's
calibration. One pass per downstream part — ~2.4 s each for 10
calibration samples at ~240 ms/step.

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
