# Probe: Qualcomm Qwen3-4B Genie w4a16 via ORT-QNN

Phase 5.5 Lever C side-quest, 2026-04-22. Question: does our
ORT-QNN 1.24.4 runtime accept a Qualcomm-reference fully-quantized-IO
w4a16 context binary at all? If yes, we have a clear IO convention
to target for any w4a16 path forward (AIMET, local QAIRT, surgery,
runtime patch) — and the perf-ceiling question for w4a16 on X2E
becomes concrete.

**TL;DR — yes, easily, and a first perf datapoint looks strong.**

## Setup

- Bundle: `models/qualcomm-qwen3-4b-ref/qwen3_4b-genie-w4a16-qualcomm_snapdragon_x2_elite/`
  (already on disk from the earlier zoo check).
- Four `.bin` parts forming the full Qwen3-4B model (36 layers split
  9/12/12/12ish across partitions). Each `.bin` is a multi-graph
  weight-shared QNN context binary carrying **10 graphs per file**
  (5 context lengths × 2 AR sizes — AR=1 decode + AR=128 prefill at
  ctx ∈ {512, 1024, 2048, 3072, 4096}).
- Probe script: `scripts/probe_qualcomm_qwen3_4b.py` — builds an
  EPContext wrapper ONNX with dtypes from `metadata.yaml`, loads via
  legacy QNN EP (same pattern as our fp16 pathbmask runtime).

## Binary-format match

`xxd` of Qualcomm's `qwen3_4b_part_1_of_4.bin` vs our own
`qwen3_0_6b_draft_v81_ctx256.pathb.w4a16-a.bin`:

```
Qualcomm qwen3_4b_part_1_of_4.bin : 00000000: 0000 0002 0000 0003 0000 0000 0000 0001
Ours   pathb.w4a16-a.bin         : 00000000: 0000 0002 0000 0003 0000 0000 0000 0001
```

Identical QNN context binary magic. Same QAIRT 2.42 (confirmed by
`tool-versions.yaml` on the Qualcomm side + our `--qairt_version 2.42`
compile flag). ORT-QNN 1.24.4 should be structurally able to load
either — format mismatch is not our problem.

## Graph-name lookup

The multi-graph binary needs the EPContext `name` attribute to match
one of its internal graph names. `strings` on part 1 revealed the
pattern `<phase>_<ar>_<ctx>_<N>_of_4`:

```
token_ar1_cl512_1_of_4       ← single-token decode at ctx=512
token_ar1_cl1024_1_of_4      ← ctx=1024
...
prompt_ar128_cl512_1_of_4    ← 128-token prefill at ctx=512
```

Ten such graphs in each `.bin`, all weight-shared. Our use case is
single-token decode → `token_ar1_cl512_*_of_4`.

## IO-name convention

Metadata.yaml uses forward slashes (`/model/model/embed_tokens/Gather_output_0`)
but the compiled binary stores node-graph outputs with two forms:
- External IO names (inputs/outputs seen by the caller):
  `_model_model_embed_tokens_Gather_output_0` — underscored.
- Internal op names kept as slashes for debug.

Rule for wrapper construction: take the metadata name, replace `/`
and `.` with `_`. Verified on both `embed_tokens/Gather_output_0`
(part 1 output) and `layers.11/Add_1_output_0` (part 2 output).

## Probe results

### Part 1 — embedding lookup

Wrapper declares 1 input (int32 `input_ids [1,1]`) and 1 output
(uint16 `_model_model_embed_tokens_Gather_output_0 [1,1,2560]`).

AC rerun with warmup + 20 measured iters (2026-04-22):

```
  loaded in 1.0 s
  providers: ['QNNExecutionProvider', 'CPUExecutionProvider']
  run latency (20 iters, warmup discarded):
    min 0.03 ms  median 0.04 ms  max 0.12 ms  mean 0.05 ms
  output shape=(1, 1, 2560) dtype=uint16 min=17571 max=44165
```

Single-call battery measurement (before warmup add): **3.27 ms** —
dominated by graph-init / HMX context setup, not the embedding op
itself. The 0.04 ms AC median reflects the actual per-call cost
once the graph is hot. Dequantized with metadata's
`scale=7.197e-6, offset=-30800`: output range = `[-0.095, +0.096]`
— plausible embedding magnitude for a transformer post-embed-scale.
**Load + forward pass green.**

### Part 2 — layers 0..11 (full-quant IO surface)

Wrapper declares 37 inputs (uint16 hidden + uint16 mask +
uint16 cos/sin + 24 × uint8 past_kv) and 25 outputs (uint16 hidden
+ 24 × uint8 present_kv slices). All shapes + dtypes as declared
matched by ORT-QNN's `session.get_inputs()/get_outputs()`.

AC rerun with warmup + 20 measured iters (2026-04-22):

```
  loaded in ~1 s
  providers: ['QNNExecutionProvider', 'CPUExecutionProvider']
  run latency (20 iters, warmup discarded):
    min 7.12 ms  median 7.20 ms  max 7.42 ms  mean 7.22 ms
  _model_model_layers_11_Add_1_output_0  shape=(1,1,2560) uint16 min=12108 max=23169
  past_key_0_out                          shape=(8,1,128,1) uint8 min=113 max=142
  past_key_1_out                          shape=(8,1,128,1) uint8 min=104 max=153
  ... (all 24 present_kv tensors non-trivial distributions)
```

**Load + forward pass green.** uint16 hidden output, uint8
per-layer present_kv outputs, all consumed and produced cleanly.

Battery-power single-call measurement was 13.37 ms; AC median is
**7.22 ms** — a 1.85× speedup, consistent with but slightly larger
than the +27% battery→AC gap we saw on fp16 pathbmask. The variance
is low (7.12–7.42 ms over 20 iters) so this is a tight estimate.

## Key findings

1. **ORT-QNN 1.24.4 accepts fully-quantized IO on EPContext wrappers
   with `TensorProto.UINT8` / `TensorProto.UINT16`.** No flag
   gymnastics required — just declare the correct dtype in the
   wrapper ONNX and feed raw quantized bytes. This was the gating
   question for the w4a16 investigation.

2. **Dtype convention for w4a16 on X2E (Qualcomm's reference,
   matches what we should target):**
   - `input_ids`: int32
   - Embedding / attention mask / cos / sin / logits: **uint16** with
     per-tensor asymmetric quant (`real = (q + offset) * scale`).
   - past_key_N, past_value_N: **uint8 per-layer**, offset=-128
     (symmetric int8 shifted), scale varies per layer.

3. **Perf datapoint (AC, clean):** 12 layers of Qwen3-4B at w4a16
   runs in **7.22 ms median** (min 7.12, max 7.42 over 20 iters
   with warmup discarded). That's **0.60 ms/layer** on the
   Qualcomm reference w4a16 IO convention.

   **Extrapolation to Qwen3-0.6B** (28 layers, single-partition graph):
   28 × 0.60 = **~16.8 ms/step** projected.

   Full-pipeline Qwen3-4B decode (all 4 partitions chained):
   ~(1 × 0.04 embed) + (3 × 7.22 attention blocks) ≈ **~22 ms/step**
   → ~45 t/s standalone on the 4B model (bounded extrapolation —
   we only measured 1 of 3 attention partitions; the head-unembed
   partition likely adds ~2-3 ms).

   **Speculative-decode projection** (v Lever B's 18.12 t/s AC fp16
   baseline, same ctx=256, same target verify ~157 ms/call):
   Expected committed per round ≈ `1 + k × accept_rate`; round
   wall ≈ `max(k × step, verify_ms)`. For step=17 ms (w4a16 draft)
   and verify=157 ms:

   | k | committed (acc=0.80) | round wall | t/s |
   |---|---:|---:|---:|
   | 2 | 2.6 | max(34, 157) = 157 | 16.6 |
   | 4 | 4.2 | max(68, 157) = 157 | 26.8 |
   | 6 | 5.8 | max(102, 157) = 157 | 36.9 |
   | 8 | 7.4 | max(136, 157) = 157 | 47.1 |
   | 10 | 9.0 | max(170, 170) = 170 | 52.9 ← NPU-bound crossover |
   | 12 | 10.6 | 204 | 52.0 |

   Accept rate drops with k (Phase 5 close: 81% at k=2 → 55% at k=8),
   so the 7.4 committed at k=8 is optimistic. Realistic projection:
   **~30-45 t/s in the k=4-8 range** vs 18.12 t/s today. *w4a16's
   real leverage isn't per-call speed at k=2 (still verify-bound) —
   it's unlocking larger-k speculative where today's fp16 would be
   NPU-bound.*

4. **Weight-shared multi-graph binaries are a thing.** Qualcomm
   ships 10 graphs per `.bin` with shared weights and `use-mmap:
   true`. This is distinct from our single-graph compiles. Not
   needed for Phase 5.5, but a future lever for doing AR=1 decode
   + AR=8 verify out of one binary (matching our k∈{2..8} sweep
   without recompiling per k).

5. **Multi-part partition chaining is Genie's job, not the HTP's.**
   The four `.bin` files are separate sessions; Genie wires
   partition_N's output to partition_(N+1)'s input in host RAM
   between calls. For a single-partition graph like our Qwen3-0.6B,
   this complication doesn't apply.

## What this means for the w4a16 investigation order

Previous order was: side-quest → AIMET (3a) → local QAIRT (3b) →
surgery → runtime patch. The probe **confirms** the order but
sharpens the criteria:

- **Local QAIRT (old option 3b) becomes the strongest path.** We
  run `qairt-quantizer` ourselves *without* `--preserve_io_datatype`
  and get a Qualcomm-reference-style full-quant-IO binary. We've
  now proven such a binary loads in ORT-QNN and runs with
  real-looking outputs. Scale/offset for every tensor lives next to
  the binary (Qualcomm-style metadata.yaml or in a sidecar
  encodings JSON from the tool). Full pipeline ownership, no AI
  Hub orchestration bugs can touch us, no ONNX-graph dummy
  insertions.

- **AIMET (old option 3a) drops to a tie with local QAIRT.** It
  gives us ONNX-level control of quantization, but then we still
  need to get the QDQ-annotated ONNX through *some* compiler to QNN
  context binary — either AI Hub (back to the buggy path) or local
  QAIRT (same conclusion). AIMET might become useful if we later
  want to tune calibration at a finer grain than QAIRT supports,
  but it's not a shorter-path-to-working-binary than local QAIRT.

- **Surgery (old option 1) and runtime patch (old option 4) are
  now deprecated unless local QAIRT hits a wall.** Both are
  workarounds around AI Hub's bug; local QAIRT avoids the bug
  entirely. Keep them as contingency.

**New primary path: install QAIRT SDK 2.42 on the dedicated x86
compile box + build qairt-quantizer + qnn-context-binary-generator
into a local recipe.** The rewrite + prep + calibration work from
sessions 12-14 transfers as-is (we hand qairt-converter the same
pathb ONNX + the same calibration `.npz`).

## Open questions the probe didn't answer

- **Does ORT-QNN support quantization metadata in the wrapper** so
  it can dequantize outputs for the caller (return float32 from
  uint16 output tensors)? The probe returned raw uint16 — we
  handled dequant in-script. Not a blocker; just determines whether
  runtime scale/offset math lives in Python or is done by ORT-QNN
  transparently.
- **What's the true end-to-end Qwen3-4B decode t/s via chained
  4-partition calls?** We only measured one partition. Full
  extrapolation of 4 × ~13 ms = ~53 ms → ~19 t/s on a 4B model,
  which tracks but isn't measured. Interesting but not required
  for the investigation to move forward.
- **Is there a flag to disable AI Hub's auto-injected
  `--preserve_io_datatype`?** Would shortcut back to AI Hub compile
  with full-quant IO. Worth a quick ping on their forum before
  committing to a local QAIRT install.

## Artifacts

- Probe script: `scripts/probe_qualcomm_qwen3_4b.py`
- Wrapper ONNXs (gitignored, live next to the bins):
  `models/qualcomm-qwen3-4b-ref/.../part1.wrapper.onnx`,
  `.../part2.wrapper.onnx`

## Status

- Tasks 11–16 closed (side-quest complete).
- AC rerun landed: **7.22 ms median for 12 layers of Qwen3-4B
  w4a16**, variance tight (7.12–7.42 over 20 iters). Projects to
  ~17 ms/step for our 28-layer Qwen3-0.6B → realistic **30-45 t/s
  k=4-8 speculative** vs today's 18.12 t/s k=2 fp16.
- Next: local QAIRT install on the x86 compile box + single-graph
  w4a16 compile of our pathb ONNX (bypasses AI Hub's preserve-list
  bug). If that stalls, revisit AIMET; surgery + runtime patch
  remain as contingency.
