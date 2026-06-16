# Running the Qwen3-14B w8a16 bundle on the Hexagon NPU

Live investigation doc for the **runtime/deploy** workstream: load and
decode the 10-part Qwen3-14B w8a16 bundle on the X2E Hexagon via ORT-QNN.
Kickoff brief: `docs/next_session_npu_engine_14b.md`. The *build* is done
(`docs/threadripper_build_server.md`); this doc tracks getting it to *run*.

Append findings here as they land.

---

## 1. Ground-truth IO contract (confirmed 2026-06-16)

The shipped bundle (`models/qwen3_14b-w8a16-specula-x2e/`) had **no
`bin_info/`** — the build server's `bundle_14b.py` didn't emit it. We
regenerated it on-device by introspecting each `.bin` with QAIRT's
context-binary utility (this is the authoritative source — it reads the
compiled HTP context, not the split scripts which had been iterated):

```
UTIL=C:\Qualcomm\AIStack\QAIRT\2.45.40.260406\bin\aarch64-windows-msvc\qnn-context-binary-utility.exe
for i in 1..10:
  $UTIL --context_binary qwen3_14b_w8a16_part_${i}_of_10.bin \
        --json_file bin_info/part_${i}_of_10.json
```

The 10 JSONs are mirrored (tracked) at `npu_engine/bin_info_14b/` since
`models/` is gitignored. **Topology (definitive):**

| part | graphName | inputs | outputs |
|------|-----------|--------|---------|
| 1  | `part1` | `input_ids[1,1]` i64 | seam `_model_embed_tokens_Gather_output_0[1,1,5120]` f32 |
| 2  | `part2` | seam(embed) + `position_ids_cos[1,1,128]` + `position_ids_sin[1,1,128]` + `attention_bias[1,1,1,512]` + `past_key_values_{0..4}_{key,value}[1,8,511,128]` | seam `_model_layers_4_Add_1_output_0[1,1,5120]` + `present_{0..4}_{key,value}[1,8,512,128]` |
| 3  | `part3` | seam(`layers_4`) + cos/sin/bias + past_kv `{5..9}` | seam `_model_layers_9_Add_1_output_0` + present `{5..9}` |
| 4  | `part4` | seam(`layers_9`)  + … + past_kv `{10..14}` | seam `_model_layers_14_Add_1_output_0` + present `{10..14}` |
| 5  | `part5` | seam(`layers_14`) + … + past_kv `{15..19}` | seam `_model_layers_19_Add_1_output_0` + present `{15..19}` |
| 6  | `part6` | seam(`layers_19`) + … + past_kv `{20..24}` | seam `_model_layers_24_Add_1_output_0` + present `{20..24}` |
| 7  | `part7` | seam(`layers_24`) + … + past_kv `{25..29}` | seam `_model_layers_29_Add_1_output_0` + present `{25..29}` |
| 8  | `part8` | seam(`layers_29`) + … + past_kv `{30..34}` | seam `_model_layers_34_Add_1_output_0` + present `{30..34}` |
| 9  | `part9` | seam(`layers_34`) + … + past_kv `{35..39}` | seam `_model_layers_39_Add_1_output_0` + present `{35..39}` |
| 10 | `part9`(!) | seam(`layers_39`)`[1,1,5120]` | `logits[1,1,151936]` f32 |

So: **embed + 8×(5-layer) + lm_head-alone = 10 parts**. This refutes the
earlier `split_tail2.py` reading (embed + 6×5 + 2×5-with-head) — those
scripts were superseded; the binaries are the truth.

**Quirk:** part10's internal `graphName` is also `"part9"` (build-time
collision). Irrelevant for one-session-per-part loading; for a combined
wrapper, give EPContext nodes file-index-based names, not graphName.

### Differences from the 4B pathb bundle (`bench_pathb_ortqnn.py`)

- **Live `attention_bias[1,1,1,512]` graph input** on every decoder part —
  NOT the 4B's folded `ScatterND` causal mask threaded part-to-part. The
  runtime computes it per step and feeds the *same* tensor to all 8
  decoder parts. Simpler wiring (no cross-part mask seam).
- **`rope_theta = 1_000_000`** (from `config.json` — *not* 10000). RoPE is
  full-dim: `cos/sin` shape `[1,1,128]`.
- KV is the same ring buffer as 4B: `past[1,8,511,128]`, present
  `[1,8,512,128]`, roll `past[:] = present[:,:,1:,:]`.
- KV tensor names use underscores: `past_key_values_{L}_{key,value}`,
  `present_{L}_{key,value}` (the 4B `_kv_layer` digit-parse still works).

### Per-step input construction (ctx=512)

```python
# token (part1 only)
input_ids[0,0] = next_token_id                       # [1,1] i64
# RoPE (shared by all decoder parts) — theta=1e6, full-dim
cos[0,0,:] = rope_cos[pos]                            # [1,1,128] f32
sin[0,0,:] = rope_sin[pos]                            # [1,1,128] f32
# live additive mask (shared by all decoder parts)
attention_bias = np.full((1,1,1,512), -65504.0, np.float32)
attention_bias[..., 512-1-pos:] = 0.0                # valid window
```

`-65504.0` is the most-negative fp16-representable value (the HTP runs
fp16 internally). Valid (unmasked) slots are the trailing `pos+1` columns;
slot 511 is always the current token.

---

## 2. Runtime config (proven)

- **`.venv` ORT 1.24.4** is the only venv with `QNNExecutionProvider`
  (`.venv-ort21` / `.venv-qairt` have CPU/Azure only).
- ORT 1.24.4 bundles QAIRT 2.42, which fails a 2.45-built bundle with QNN
  error 5000 → override `backend_path` to the **system QAIRT 2.45
  `QnnHtp.dll`** (`C:\Qualcomm\AIStack\QAIRT\2.45.40.260406\lib\aarch64-windows-msvc\QnnHtp.dll`).
  This is exactly the `bench_pathb_ortqnn.py` `SYS_QNN` trick. (The brief's
  "use ORT-QNN 2.1.0" refers to a package version we don't have installed;
  the `.venv` + SYS_QNN combo is the working path.)
- CPU fp reference uses **`.venv-arm-export`** (torch 2.10 + transformers
  **4.57.6**, matching the bundle's build-time transformers).

---

## 3. The >7-session ceiling — plan

ORT-QNN historically tops out ~7 HTP sessions
([[reference_ortqnn_session_limit]]); we have 10 parts. The 4B combined
wrapper was *rejected* because AR1+AR128 graphs in one `.bin` had duplicate
input names at incompatible shapes (`docs/npu_engine_prefill_sidequest.md`).

**That failure does not apply here:** all 10 parts are single AR1 graphs in
separate `.bin`s with *distinct* input names (distinct seams, distinct
`past_key_values_{L}`). The only shared inputs — `position_ids_cos/sin`,
`attention_bias` — are *legitimately* one tensor feeding multiple EPContext
nodes (no rename needed, so no QNN name-lookup break). So a single ONNX
wrapper holding N EPContext nodes (one per `.bin`, seam outputs wired as
internal edges to the next node's seam input) should collapse 10 sessions
to far fewer.

Also new vs the 4B finding: this bundle's `genie_config.json` sets
`use-mmap: true` and `htp_backend_ext_config.json` sets
`weight_sharing_enabled: true` — the 7-session ceiling was measured on the
4B w4a16 bundle *without* mmap, so the real ceiling for this bundle is an
empirical unknown. **Approach: measure it.** Load parts incrementally as
separate sessions and find where it actually breaks; if <10, fall back to
the combined wrapper (M parts/session) to get under the ceiling.

---

## 4. RUNTIME BLOCKER (2026-06-16): decoder parts are fp16-oversized

**The shipped w8a16 bundle cannot run on the X2E HTP as built.** The 8
decoder parts (parts 2-9) fail to load with **QNN 1002**
(`Failed to create context from binary`) — and they fail **alone**, not
just when co-resident, so this is *not* the >7-session ceiling.

What loads vs. what doesn't (each tested in isolation, ORT 1.24.4 + SYS_QNN):

| part | role | size | constSize | loads? |
|------|------|------|-----------|--------|
| 1  | embed   | 1.56 GB | 1.555 GB (fp16 embedding) | ✅ |
| 2-9| decoder | 3.30 GB | **3.303 GB** | ❌ QNN 1002 |
| 10 | lm_head | 1.56 GB | 1.555 GB (fp16 head)      | ✅ |

### Root cause — the "w8a16" build stored fp16 weights, not int8

Three independent confirmations:

1. **`constSize` math.** One Qwen3-14B layer = 330.3M params; 5 layers =
   1.65B. int8 (1 B/param) → **1.65 GB**; fp16 (2 B/param) → **3.30 GB**.
   The decoder `constSize` is **3,303,236,608 B = 3.30 GB → fp16 weights.**
2. **All graph IO is `FLOAT_32`.** A *true* w8a16 HTP bundle (Qualcomm's 7B
   ref) ships **uint16/uint8** quantized KV/activation IO. Ours is fp32
   everywhere — the build used `qairt-converter --preserve_io_datatype`.
3. **No calibration.** `build_qairt_14b.sh` runs
   `qairt-quantizer … --weights_bitwidth 8 --act_bitwidth 16
   --use_per_channel_quantization` with **no `--input_list`** (basic-PTQ).

**The chain:** no activation calibration → activations get no int16
encodings → `qnn-context-binary-generator` compiles a **float (fp16)**
HTP graph → the int8 weights are materialized back to **fp16** for the
fp16 compute path → each decoder context is **2× oversized (3.30 GB)** →
exceeds the X2E runtime **per-context ceiling (between 1.56 and 3.30 GB,
~2 GB**; Qualcomm's loadable parts top out at 1.09 GB) → **QNN 1002**.

So for *this* 5-layer split, calibration is not merely an accuracy nicety
(as `next_session_npu_engine_14b.md` assumed) — it is **required for the
int8 weight storage that keeps a part under the runtime context ceiling.**

### Comparison anchor — the working 7B w8a16 bundle

`models/qualcomm-qwen2_5-7b-ref/` loads fine: decoder parts **711 MB**
(7 layers, w8a16, hidden 3584), uint16 KV IO, identical
`htp_backend_ext_config.json`. The only material difference is real int8
weights + quantized IO. Confirms the fix direction.

## 5. Build fix spec (→ task #7)

Re-quantize on the Threadripper from the saved `06_split` fp32 ONNX:

- **Add calibration:** `qairt-quantizer --input_list <cal.txt>` with
  representative per-part activations (see `end-to-end/lib/cal.py` for the
  AR1 cos/sin/attention_bias/KV construction; feed a handful of real
  prompt steps). This is what lets the HTP build a true int8-weight graph.
- **Quantize IO:** drop `--preserve_io_datatype` on the converter (or set
  IO to uint16/uint8) so KV/activation tensors are quantized like
  Qualcomm's 7B — smaller + native HTP consumption, and it forces the
  quantized compute path.
- **Target:** ~1.65 GB int8 decoder parts. 1.65 GB is just above the
  proven-loadable 1.56 GB, so it should load; if 1.65 GB still trips the
  ceiling, fall back to **4 layers/part** (~1.32 GB) → 12 parts, which the
  combined wrapper (§6) collapses under the session ceiling.

## 6. Engine + session-ceiling fix — VALIDATED on loadable parts

`npu_engine/engine_14b.py` is written general over part count + topology
and **validated as far as the (broken) bundle allows**:

- **part1 embed lookup runs:** `input_ids→[1,1,5120]`, all 5120 nonzero,
  mean≈0 std 0.024 — a sane embedding. Load→bind→run path correct.
- **Combined EPContext wrapper WORKS (the >7-session fix):** part1+part10
  loaded as **two EPContext nodes in ONE ORT-QNN session**
  (`QNNExecutionProvider`), correct gin/gout, ran to finite logits. The 4B
  combined-wrapper failure (`docs/npu_engine_prefill_sidequest.md`, dup
  AR1/AR128 input names) **does not recur** — these AR1 parts have distinct
  names and the shared cos/sin/bias are one tensor feeding several nodes.
  So once correctly-sized parts exist, `--groups` collapses 10 parts into
  ≤7 sessions (or fewer) and the full chain runs.

The engine cannot be exercised end-to-end (seam + KV + decode) until the
decoder parts load — that's gated on task #7.

**Known cosmetic issue:** ORT-QNN segfaults at interpreter teardown
(QNN context destruction) *after* all work + prints complete. Benign;
ignore the exit-time `Segmentation fault`.

## 8. FIX PROVEN (2026-06-16, same session) — calibration → int8 → loads

Drove the rebuild on the Threadripper (`root@192.168.10.5`, code +
intermediates on `/mnt/vm_8tb/specula-build`, `specula-qairt:2.45` Docker
image with QAIRT 2.45 **and** onnxruntime 1.23.2). Helper:
`end-to-end/build_server/boxssh.py` (paramiko; SFTP write fails on the unRAID
FUSE mount → use the `putx` base64-over-exec op; set `MSYS_NO_PATHCONV=1`).

Pipeline confirmed against `06_split` (9 ONNX parts: embed + 8×5-layer
decoder; **no lm_head ONNX** — part10 bin came from `05_pathb_ctx512`):

1. **Calibration capture** (`capture_calib_14b.py`, in-container ORT): runs
   the fp32 chain AR1 over 6 pre-tokenized prompts, threads the KV ring, dumps
   each decoder part's real input feeds as `.raw` + `input_list.txt`.
2. **Re-quant** (`requant_14b.sh`): convert → `qairt-quantizer … --input_list
   <cal>` → context-bin. **part2 dropped 3.30 GB → 1.66 GB** (`constSize`
   1.656 GB = int8 ✅). **It LOADS on the X2E Hexagon in 2.6 s** via ORT-QNN
   (QNNExecutionProvider). The fp16→unloadable / int8→loadable causality is
   nailed shut.

### IO is uint16, not fp32 — and that's the right answer

With calibration the quantizer makes **all IO `UFIXED_POINT_16`** (per-tensor
`scaleOffset`, QNN convention `real = scale·(q + offset)`). Attempts to force
fp32 IO failed: bare `--preserve_io_datatype` is ignored once calibration is
on; `--config` float32 on every IO makes `qnn-context-binary-generator` fail
op-validation (1002) — a *quantized* graph can't take float IO (the original
fp32-IO bins only built because they were *fully float*). uint16 IO is the
Qualcomm-native path (their 7B ref ships it) and it loads. So we embrace it.

### Refined plan (lower scope than a full 10-part rebuild)

- **Re-quantize only parts 2–9** (the broken decoder parts) → uint16 IO + int8
  weights (~1.66 GB each, loadable). **Keep part1 + part10** fp16 bins (they
  already load at 1.56 GB).
- **Engine bridges fp32↔uint16 at the two outer seams only:** quantize part1's
  fp32 embed seam → uint16 for part2; dequant part9's uint16 seam → fp32 for
  part10. Internal decoder seams + KV stay uint16 (bridge with the per-tensor
  `scaleOffset` if consecutive-part encodings differ; threaded calibration
  should make them match). cos/sin/attention_bias fed as uint16 per part
  (quantized with that part's encoding; 16-bit is ample for rope + the
  0/-65504 mask). Model on `npu_engine/qualcomm_qwen3_4b_oracle.py` (already
  runs Qualcomm uint16-IO bundles).

## 7. Progress log

- **2026-06-16** — bin_info extracted + topology confirmed; CPU fp ref
  built (coherent: `'<think>\nOkay, the user is asking…'`, argmax 151667).
  Engine written + validated (part1 embed, combined 2-node wrapper).
  **Found the runtime blocker: decoder parts are fp16-oversized (no-calib
  build) and exceed the ~2 GB X2E HTP per-context ceiling.** Fix = re-quant
  with calibration + quantized IO (task #7). Next: drive the rebuild.
