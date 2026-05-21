# qai_hub_recipe.md — Qualcomm's Qwen3-4B quantization recipe vs ours

Track 4 of `docs/e2e_optimizations.md`. Qualcomm's shipped
`qwen3_4b-genie-w4a16` bundle was produced by `qai-hub-models`; that
package's Qwen3 recipe is the source of truth. This doc extracts it,
diffs it against our `end-to-end/lib/aimet.py` recipe + the pathb
rewrites, and lists what to change.

**Source:** `qai_hub_models` **0.54.0** IS installed locally at
`/workspace/venvs/aimet-2.26-cu121-py310/lib/python3.10/site-packages/qai_hub_models/`.
All findings below are from reading that source (no GitHub needed).

## TL;DR — the root cause is confirmed

Our `QuantizationSimModel(...)` is built with **`config_file=None`**.
AIMET 2.26 then falls back to its built-in
`default_config_per_channel.json`, whose
`supergroup_pass_list` is **`["MatmulAdd"]`** — it has **no
`RMSNormalization` pass**.

Qualcomm builds QuantSim with an explicit
`config_file=default_config_llama.json` whose
`supergroup_pass_list` is **`["LayerNormalization", "RMSNormalization"]`**.

`aimet_onnx/graph_passes/passes/rmsnorm.py` (`@register_pass
"RMSNormalization"`) does exactly the thing Track 3 wants — it matches
the decomposed `Pow/ReduceMean/Add/Sqrt/Div(/Mul)` cluster and calls
`disable_output_quantizers(...)` + `disable_const_quantizers(...)` on
**every intermediate op**. Result: the `x²` tensor (range ~5e6), the
ReduceMean output, the Sqrt/Add/Div outputs etc. are left **unquantized
→ HTP float fallback** — no QNN_Convert, no int16 annihilation. Only
the RMSNorm input and the final affine `Mul` output keep quantizers.

Because we pass no config file, our recipe quantizes every one of those
intermediates → the int16-over-5e6-range collapse → probe cos 0.44-0.56.
**This is a one-line fix:** pass `config_file=` pointing at a config
with `RMSNormalization` (and `LayerNormalization`) in
`supergroup_pass_list`. It is the principled form of Tracks 1/2/3 and
needs no code in our pathb rewrites at all.

---

## (a) Qualcomm's recipe (Qwen3-4B, w4a16, genie)

### Pipeline (qai_hub_models)

`models/qwen3_4b/quantize.py` → `_shared/llm/quantize.py::llm_quantize`
→ `quantize()` → `Qwen3_4B_AIMETOnnx`. Stages:

1. **FP model adaptation** (`_shared/qwen3/model.py::monkey_patch`,
   `model_adaptations.py`):
   - `SHAQwen3Attention` — Split-Head Attention: each attention head is
     a separate `Conv2d` (q/k/v/o projections become 1x1 Conv2d, then
     per-head). Q/K/V/O are **Conv, not MatMul**.
   - `q_norm`/`k_norm` RMSNorm split per-head (`q_norm_sha`,
     `k_norm_sha`) — operate on `head_dim`.
   - RoPE hoisted out: `Qwen3RotaryEmbedding.forward` is bypassed;
     cos/sin are graph inputs (`position_ids_cos/sin`). Same as our
     pathb rewrite.
   - `apply_rotary_pos_emb` replaced with `_apply_rope_single`.
   - MLP: only `down_proj` becomes `ConvInplaceLinear` (`up_proj`,
     `gate_proj` left as-is — temporarily, per an AISW bug comment).
   - `lm_head` becomes `ConvInplaceLinear`.
   - Attention scaling `k / sqrt(head_dim)` is folded **into the K
     operand before matmul** (avoids fp16 overflow).
   - `attention_mask` is multiplied by `attention_mask_multiplier`
     (=1.0 for Qwen3-4B) and the FP model clips the mask to
     **`[-100, 0]`** (`attention_mask_min_clip_and_multiplier` returns
     `(-100.0, 1.0)` for the AIMETOnnx class; the position processor
     used by ORT-GenAI clips `[-50, 0]`). Genie itself uses `-1000`.

2. **ONNX export** (`_shared/llm/model.py::_export_to_onnx`):
   - **opset 17**, `dynamo=False` (static shapes; the default path).
   - opset 18 + `dynamo=True` only on the experimental
     `use_dynamic_shapes` path.
   - Per-seq-len ONNX files (`model_seqlen{S}_cl{C}.onnx`), one per
     export seq length, plus `ctx//2`.
   - `optimize_onnx_model` (`onnx_optimize.py`) runs onnxscript rewrite
     rules (const folding, no-op elimination, broadcast-to-matmul,
     redundant ScatterND removal, etc.). For >10k-node graphs the
     `_basic_rules` set is skipped. **No explicit RMSNorm fusion at the
     ONNX level** — RMSNorm stays decomposed in the ONNX graph; the
     RMSNorm handling happens inside QuantSim via the supergroup pass
     (see below).

3. **QuantSim build** (`_shared/llm/model.py::_build_quantsim`):
   ```python
   default_config = get_aimet_config_path("default_config_llama")
   quantsim.op_types_to_tie_qtzrs = ["Concat"]
   quantsim._tie_qtzrs = True
   quantsim.op_outputs_to_ignore.append("Slice")
   quantsim.op_outputs_to_ignore.append("Constant")
   qs.encoding_version = "1.0.0"
   QuantizationSimModel(
       model=onnx_model,
       param_type="int4",
       activation_type="int16",
       quant_scheme=QuantScheme.min_max,
       config_file=default_config,        # <-- default_config_llama.json
       providers=providers,
   )
   ```

4. **`default_config_llama.json`** — the QuantSim op config. Key fields
   vs the AIMET default:
   - `defaults.per_channel_quantization: "True"`,
     `params.is_symmetric: "True"`, activations asymmetric
     (`unsigned_symmetric: "False"`, `strict_symmetric: "False"`).
   - `params.bias.is_quantized: "False"` — biases not quantized.
   - `op_type` exclusions — `is_output_quantized: "False"` for:
     `Cast, Gather, GatherND, Reshape, Transpose, Slice, Split,
     Squeeze, Tile, Expand, Pad, Mean, ReduceMax, ReduceMin,
     ScatterElements, TopK, NonZero, MaxPool, Dropout, Upsample, ...`
   - `Gather` also `per_channel_quantization: "False"` (embedding).
   - `Softmax`/`Sigmoid` `encoding_constraints` pin range to `[0,1]`.
   - **`supergroup_pass_list: ["LayerNormalization", "RMSNormalization"]`**
     — THE critical line. Triggers the AIMET graph passes that disable
     output quantizers on all LayerNorm/RMSNorm internal ops.

5. **Precision config** (`_configure_quant_sim`, w4a16 path →
   `_apply_int8_kv_cache_tying_and_lm_head` in `_shared/llm/_utils.py`):
   - **Concat quantizers tied** (`_tie_quantizers_for_op_types(["Concat"])`).
   - **KV cache → int8 symmetric**: every `past_key*/past_value*` graph
     input AND output tensor set to 8-bit, `use_symmetric_encodings=True`.
   - **KV in/out quantizers tied** (`_tie_quantizers_for_kv_cache`) —
     past_key_in shares the quantizer of past_key_out so the cache
     round-trips without re-quantizing.
   - **lm_head weights → int8, per-channel** (`_set_lm_head_to_8b`:
     `blockSize=0`, `blockAxis=-1`, per-channel). Note Qwen3 ties
     embeddings, so lm_head weight == embedding table.
   - **`_set_matmul_second_input_to_8b` (use_16x8_matmuls=True)**: every
     `MatMul`'s second (activation) input is forced to int8 symmetric,
     and 8-bit is **propagated upstream** through
     `Concat/Transpose/Reshape/Slice/Div`. So the attention BMMs run
     **16x8** (16-bit one operand, 8-bit the other), not 16x16. Comment
     notes Qwen3 swaps MatMul/Div order so the `Div` is upstream and
     gets caught by the 8-bit propagation.

6. **Optimizers** (`utils/quantization_aimet_onnx.py::quantize`):
   - **SEQ_MSE**: `apply_seq_mse(quant_sim, data)` — default
     `DEFAULT_SEQ_MSE_NUM_SAMPLES = 20`, `num_candidates` left at the
     `apply_seq_mse` default (not overridden).
   - **AdaScale** (the shipped `qwen34_w4a16_adascale` checkpoint name
     confirms it was used): `DEFAULT_ADA_SCALE_NUM_SAMPLES = 128`,
     `DEFAULT_ADA_SCALE_NUM_ITERATIONS = 512`. For Qwen3 it sets
     `DecoderBlockQwen3.NUM_RMSNORM_PER_BLK = num_attn_heads +
     num_kv_heads + 1` (=32+8+1=41 for 4B — per-head q/k norms + the
     input layernorm).
   - **compute_encodings** last, over the calibration set.
   - Calibration data: `get_calibration_data()` — wikitext-style real
     text, `DEFAULT_CALIBRATION_SEQ_LEN`, `--num-samples` default 20.
   - `quant_scheme = QuantScheme.min_max` throughout.

7. **Encodings adaptation for split** (`Qwen3Base_AIMETOnnx::_adapt_aimet_encodings`):
   - Copies `model.model.embed_tokens.weight`'s encoding onto the
     activation `/model/model/embed_tokens/Gather_output_0` (the embed
     Gather output must share the table's encoding — same problem our
     `_pin_embedding_w16` solves, but Qualcomm solves it by *copying the
     encoding* post-hoc rather than pinning bitwidth pre-SEQ_MSE).
   - Copies every `*weight*` activation encoding into `param_encodings`.
   - `propagate_memory_encodings(encodings, model)` — propagates
     encodings across memory/no-op boundaries.

8. **Split + compile**: `NUM_SPLITS=4`, `NUM_LAYERS_PER_SPLIT=12`
   (36 layers / 4). Genie bundle for ctx ∈ {512,1024,2048,3072,4096},
   ar ∈ {1,128}.

### Concrete config values (from the reference bundle on disk)

`reference/qwen3_4b_qualcomm/.../metadata.json` + sibling JSONs:

| Tensor / setting | Value |
|---|---|
| Activations (boundary I/O) | **uint16 asymmetric per-tensor** |
| KV cache (`past_key/value_*`) | **uint8**, zero_point 128 (symmetric-ish) |
| `input_ids` | int32 |
| embed Gather output | uint16, scale 7.2e-6, zp 30800 |
| `attention_mask` input | uint16, scale 1.5e-3, zp 65535 |
| `position_ids_cos/sin` | uint16, scale 3.05e-5, zp 32768 |
| residual stream (`layers.N/Add_1_output_0`) | uint16 |
| `logits` | uint16 |
| precision | w4a16 |
| ctx lengths | 512/1024/2048/3072/4096 |
| ar | 1 (decode) + 128 (prefill) |
| parts | 4 |
| rope_theta | 1000000 |
| kv-dim 128, pos-id-dim 64 | `genie_config.json` |
| htp | soc_model 88, dsp_arch v81, weight_sharing_enabled |
| mask clip | genie uses -1000; FP/AIMET use -100/-50 |

Nothing exotic in the precision — confirms Session 29's insight:
Qualcomm wins on **op-config (not quantizing norm internals + 16x8
matmuls)**, not on activation precision.

---

## (b) Delta table — Qualcomm vs ours (`lib/aimet.py` + pathb rewrites)

| # | Field | Qualcomm (qai-hub-models) | Ours (`lib/aimet.py`) | Impact |
|---|---|---|---|---|
| 1 | **QuantSim `config_file`** | `default_config_llama.json` with `supergroup_pass_list:["LayerNormalization","RMSNormalization"]` | **`None`** → AIMET default `default_config_per_channel.json`, `supergroup_pass_list:["MatmulAdd"]` only | **CRITICAL.** RMSNorm internals (`Pow`/`x²`@~5e6, ReduceMean, Add, Sqrt, Div) get int16 activation quantizers → annihilation → cos 0.44-0.56. This is THE gap. |
| 2 | **RMSNorm intermediate quantizers** | Disabled (HTP float fallback) by the `RMSNormalization` pass | All quantized at int16 | Same root cause as #1; fixed by #1. |
| 3 | **Op-type exclusions** | `is_output_quantized:False` for Cast/Gather/Reshape/Transpose/Slice/Split/Expand/Pad/Mean/ReduceMax/ReduceMin/ScatterElements/Tile/Squeeze/TopK… | AIMET default only (fewer exclusions; default config differs) | Medium. Mask `Where`/`ConstantOfShape`/`Slice` constants (~4e37) get quantized in ours. Fixed by #1 (adopt the config). |
| 4 | **16x8 MatMul** | `_set_matmul_second_input_to_8b` forces every MatMul's 2nd input to int8 sym + propagates 8-bit upstream through Concat/Transpose/Reshape/Slice/Div | Not done — attention BMMs are 16x16 | Medium. Affects attention numeric range; also what HTP expects. |
| 5 | **KV cache precision** | int8 **symmetric**, in/out quantizers **tied** | int16 (QuantSim build type); no KV tying | Medium. Our KV is over-precise (int16 vs int8) and untied → cache re-quantizes each step. metadata.json shows uint8 KV. |
| 6 | **Concat quantizers tied** | `op_types_to_tie_qtzrs=["Concat"]`, `_tie_qtzrs=True` | Not set | Low/Medium. Concat-of-KV needs consistent scale. |
| 7 | **`op_outputs_to_ignore`** | `["Slice","Constant"]` appended | Not set | Low. Slice/Constant outputs skipped. |
| 8 | **lm_head / embed weights** | lm_head → int8 per-channel; embed Gather-output encoding *copied from* the table encoding post-hoc (`_adapt_aimet_encodings`) | `_pin_embedding_w16` pins embed table to int16 per-tensor pre-SEQ_MSE; no lm_head special-casing | Low/Medium. Qualcomm keeps embed/lm_head at int8 (Qwen3 ties them); we force int16. Different but both "work"; ours is heavier. |
| 9 | **V/O proj weight bitwidth** | No V/O-specific pin. V/O are Conv2d (SHA); 16x8-matmul rule handles attention BMMs | `_bump_vo_to_w8` bumps v_proj/o_proj MatMul weights to w8 | Ours is a workaround for a collapse Qualcomm avoids structurally (SHA Conv + 16x8). Likely unnecessary once #1+#4 land. |
| 10 | **SHA attention** | Split-Head: q/k/v/o = per-head `Conv2d`; per-head q_norm/k_norm | MHA MatMul projections (optimum-cli export) | Structural. Qualcomm's graph is Conv-based; ours is MatMul-based. Affects which AIMET passes fire (`MatMul` per-channel vs `Conv`). Not required for cos but explains naming/topology diffs. |
| 11 | **MatMul/Div swap** | K divided by `sqrt(head_dim)` before BMM (fp16-overflow fix); Div upstream so 8-bit propagation catches it | Standard post-matmul scale (optimum export) | Low for int activations; relevant if matching exactly. |
| 12 | **AdaScale `NUM_RMSNORM_PER_BLK`** | Set to `num_attn_heads+num_kv_heads+1` (=41 for 4B) | `ada_scale_num_rmsnorm_per_blk` not set; our AdaScale uses default block detection | Low/Medium. Wrong RMSNorm count per block → AdaScale mis-segments decoder blocks. |
| 13 | **AdaScale samples/iters** | 128 samples / 512 iterations (defaults) | `ada_scale_iters` configurable; currently `--no-use-ada-scale` for 4B | Tuning. Qualcomm's shipped checkpoint name is `qwen34_w4a16_adascale` → AdaScale was used. |
| 14 | **ONNX opset** | 17, `dynamo=False` (static) | optimum-cli export is opset 18 (our patches add ReduceMean v18 handler) | Low. opset 18 works with our patches; 17 is Qualcomm's. Not a quality issue. |
| 15 | **SEQ_MSE** | `apply_seq_mse`, 20 samples, default candidates | `apply_seq_mse`, configurable candidates | Aligned. |
| 16 | **quant_scheme** | `QuantScheme.min_max` | `min_max` (forced when SEQ_MSE on) | Aligned. |
| 17 | **encoding_version** | `"1.0.0"` | 1.0.0 (AIMET 2.26 default) | Aligned. |
| 18 | **mask clip** | FP model clips mask to `[-100,0]`; genie runtime `-1000` | pathb folds mask; probe feeds `-65504` | Medium. Our extreme `-65504` mask values blow up the activation range an int16 quantizer sees on mask-adjacent tensors. Qualcomm clips to -100. |
| 19 | **Calibration data** | wikitext real text, `get_calibration_data()` | `cal_iter` (our `lib/cal.py`) | Check our cal data is real text, not random. |

---

## (c) Prioritized recommendations for `lib/aimet.py` / pathb rewrites

### P0 — adopt Qualcomm's AIMET config file (the fix)

In `run_aimet`, stage 4, pass an explicit `config_file` to
`QuantizationSimModel(...)`. Two options:

- **Best:** vendor `default_config_llama.json` into the repo (e.g.
  `end-to-end/lib/aimet_config_llama.json`, copied verbatim from
  `qai_hub_models/utils/aimet/default_config_llama.json`) and pass its
  path. This brings `supergroup_pass_list:["LayerNormalization",
  "RMSNormalization"]` plus all the op-type exclusions and
  `Softmax`/`Sigmoid` constraints in one shot.
- Then add, before constructing the QSM (mirroring `_build_quantsim`):
  ```python
  from aimet_onnx import quantsim
  quantsim.op_types_to_tie_qtzrs = ["Concat"]
  quantsim._tie_qtzrs = True
  quantsim.op_outputs_to_ignore.append("Slice")
  quantsim.op_outputs_to_ignore.append("Constant")
  ```

This alone should move probe cos from ~0.5 to ~0.99 — it makes the
RMSNorm-internal `x²`/ReduceMean/Sqrt/Div tensors fall back to HTP
float (no quantizer, no QNN_Convert). It is the principled,
already-shipped form of Tracks 1/2/3 and requires **no pathb-rewrite
changes**. Verify on 0.6B-w8a16 first.

### P1 — match the w4a16 precision config

Port `_apply_int8_kv_cache_tying_and_lm_head` behavior into `lib/aimet.py`
for the w4a16 path (functions are in
`qai_hub_models/models/_shared/llm/_utils.py` — can be copied or
imported since `qai_hub_models` is installed):
- KV cache tensors → int8 symmetric, in/out quantizers tied.
- `_set_matmul_second_input_to_8b` → attention BMMs run 16x8.
- lm_head weights → int8 per-channel.

This makes the attention path match Qualcomm and removes the need for
our `_bump_vo_to_w8` V/O workaround (delta #9) — the V/O collapse is a
symptom of 16x16 attention BMMs that the 16x8 rule fixes structurally.
Re-evaluate whether `_bump_vo_to_w8` is still needed after this lands.

### P2 — clip the attention mask

In the pathb mask handling / probe, clip the folded additive mask to
`[-100, 0]` instead of `-65504`/`-1000`. An int16 quantizer over a
[-65504, 0] range has ~2-unit granularity that destroys small logit
contributions on mask-adjacent tensors. Qualcomm's FP model clips to
`[-100, 0]` (`attention_mask_min_clip_and_multiplier` → `-100.0`).
Apply in `rewrite_qwen3_pathb*` (fold-pathbmask) and in the
`run_aimet` probe's `attention_bias` construction.

### P3 — AdaScale RMSNorm-per-block count

If we re-enable AdaScale for Qwen3-4B, set
`DecoderBlockQwen3.NUM_RMSNORM_PER_BLK = num_attention_heads +
num_key_value_heads + 1` before calling `apply_adascale` (=41 for 4B,
matching `Qwen3_4B_AIMETOnnx.ada_scale_num_rmsnorm_per_blk`). Our
`_patch_apply_adascale_for_pathb_kv` already monkey-patches AdaScale;
add this constant override there. Without it AdaScale mis-segments
decoder blocks on the SHA/per-head-norm graph.

### P4 — verify calibration data is real text

Confirm `lib/cal.py::cal_iter` feeds real tokenized text (wikitext-like),
not random tokens. Qualcomm calibrates on real text via
`get_calibration_data()`. Outlier statistics of real text vs random
materially change `min_max` encodings.

### Notes / non-actions

- **opset**: keep opset 18 + our existing ReduceMean-v18 /
  AdaScale-KV-naming patches; opset is not a quality lever. Qualcomm
  uses 17 only because they export from torch directly.
- **SHA Conv vs MatMul**: Qualcomm's per-head-Conv attention is a
  deeper structural rewrite. Not required to close the cos gap (P0
  does that). Revisit only if exact topological parity with the
  reference bundle is needed for `compare_to_qualcomm.py`.
- After P0, re-baseline; P1-P4 are refinements. Expect P0 alone to
  clear the 0.99 gate per the Session 28 "disable all activation
  quantizers → 0.99" experiment — P0 disables exactly the harmful
  subset (norm internals) while keeping the rest quantized for HTP.
