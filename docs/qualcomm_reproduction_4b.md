# Reproducing the Qualcomm Qwen3-4B w4a16 reference bundle

**Status:** in-progress 2026-04-23. Phases 0-2 complete; Phase 3 (single-part
w4a16 compile) running.

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

## Phase 5 — multi-part weight-shared compile (planned)

**TBD — only attempt once Phases 3+4 land cleanly.**
