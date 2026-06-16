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

## 4. Progress log

- **2026-06-16** — bin_info extracted + topology confirmed (above). CPU fp
  reference generation kicked off. Engine implementation next.
