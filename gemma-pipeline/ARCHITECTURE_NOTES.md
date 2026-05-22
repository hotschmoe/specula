# Gemma 4 → Hexagon NPU — architecture notes

The load-bearing design doc for this pipeline. The Qwen3 pipeline
(`end-to-end/`) does **not** transfer to Gemma 4 by config-swap: the
two decoders differ structurally in ways that touch every stage of the
conversion. This doc enumerates each difference, what it breaks, and
what the Gemma pipeline has to do instead.

Reference config: `google/gemma-4-E2B-it` (`config.json` →
`text_config`), fetched 2026-05-22. The E2B text decoder is the
smallest Gemma 4 and the target for first conversion.

## Gemma 4 E2B text decoder — the numbers

| field | value | note |
|---|---|---|
| top-level arch | `Gemma4ForConditionalGeneration` | multimodal wrapper |
| text `model_type` | `gemma4_text` | the decoder we want |
| hidden_size | 1536 | |
| num_hidden_layers | 35 | |
| num_attention_heads | 8 | |
| num_key_value_heads | 1 | 8:1 GQA — tiny KV |
| head_dim | 256 | sliding-attn layers |
| global_head_dim | 512 | **full-attn layers differ** |
| intermediate_size | 6144 | `use_double_wide_mlp=true` |
| hidden_activation | `gelu_pytorch_tanh` | GeGLU, not SwiGLU |
| vocab_size | 262144 | ~1.7× Qwen3's 152k |
| max_position_embeddings | 131072 | 128k native context |
| sliding_window | 512 | |
| layer_types | 4 sliding : 1 full, repeating | 28 sliding / 7 full (approx) |
| num_kv_shared_layers | 20 | last 20 layers reuse earlier KV |
| hidden_size_per_layer_input | 256 | **Per-Layer Embeddings (PLE)** |
| final_logit_softcapping | 30.0 | tanh clamp on logits |
| rms_norm_eps | 1e-6 | |
| rope (full) | θ=1e6, `proportional`, partial_rotary_factor=0.25 | only 25% of dims rotated |
| rope (sliding) | θ=1e4, `default` | full rotary |
| enable_moe_block | false | E2B is dense (26B variant is MoE) |

## Difference matrix vs Qwen3 (the `end-to-end/` pipeline target)

| aspect | Qwen3-4B | Gemma 4 E2B | pipeline impact |
|---|---|---|---|
| HF arch class | `Qwen3ForCausalLM` | `Gemma4ForConditionalGeneration` | **Stage 1** — optimum export must isolate the *text* decoder; the wrapper also carries vision+audio towers. Export `text_config` only or strip towers post-export. |
| attention | uniform full attention, all layers | interleaved **4×sliding(512) : 1×global** | **Wins the long-context war.** 28/35 layers cap KV at 512 tokens → the `long_context_scaling.md` §8.8 TCM-tiling wall (a 32 MiB per-layer KV slice that won't fit VTCM) simply does not arise on sliding layers. SWA — the session-31/32 "real fix" — is *native*. |
| RoPE | single θ | **dual**: global θ=1e6 `proportional` partial=0.25; sliding θ=1e4 `default` | **Stage 4 (rotary hoist)** must build TWO cos/sin caches and route each layer to the right one. Global layers rotate only 64 of 256 dims (partial_rotary_factor 0.25) — the hoist must split rotary/pass-through dims. Qwen3's single-table hoist is unusable. |
| head_dim | uniform 128 | 256 sliding / **512 global** | per-layer-type KV shape; the partition seam map and KV-cache size estimate must be computed per layer, not once. |
| GQA | 4-ish KV heads | **1 KV head** (8:1) | KV cache is tiny — strongly positive for the part-count / VTCM budget. |
| KV sharing | none | **`num_kv_shared_layers=20`** | the last 20 layers have **no KV-cache graph I/O of their own** — they read an earlier layer's K/V. Only 15 layers own KV. The pathb rewrite + `split.py` seam map + the ORT-QNN KV manager must all model this; assuming 35 independent KV pairs is wrong. |
| Per-Layer Embeddings | none | **PLE**: a 2nd `262144×256` embed table, residual fed into every layer | new graph input / extra Gather; the pathb rewrite must preserve the per-layer residual injection. No Qwen3 analogue — this is the single biggest new graph-surgery item. |
| norm | RMSNorm | Gemma RMSNorm `(1+w)·x̂` + QK-norm + pre/post-FFN norms | more ReduceMean ops → the AdaScale ReduceMean-v18 crash (`end-to-end/README.md` "Known issue") is *more* likely. Plan on `--no-use-ada-scale` or the v18 converter patch from day one. |
| activation | SwiGLU (silu) | **GeGLU** (`gelu_pytorch_tanh`) | quantization-sensitive; AIMET handles gelu but the tanh approximation should be checked in the cos probe. |
| logit softcap | none | **`final_logit_softcapping=30.0`** | extra `30·tanh(logits/30)` at the lm_head output. Keep it in the graph or fold consistently — dropping it shifts the argmax distribution. |
| embedding scale | none | embeddings ×`sqrt(hidden_size)` | constant-fold into the embedding weights or keep as a Mul. |
| vocab / lm_head | ~152k | **262144** | bigger embed + lm_head tensors → bigger Part-1/Part-N; revisit the size-ceiling part split. |
| native context | 40960 | **131072** | the 32k target is comfortably *in-window* — no NTK theta rescale needed (unlike Qwen3-4B beyond 40960). |

## What this means for the long-context goal (32k)

The Qwen3 project hit a wall (`current_status.md` sessions 31–32): dense
global attention can't tile a 32k KV slice into VTCM, so dense 32k/64k
is "structurally impossible" without sliding-window attention, and
Qwen3-4B was never SWA-trained.

**Gemma 4 removes that wall by construction:**

1. **SWA is native and trained.** 28 of 35 layers are sliding-window
   (512). Their KV slice is fixed at 512 tokens regardless of context
   length — it tiles into VTCM trivially. No quality hack.
2. **Only ~7 layers are global.** Those carry an O(ctx) KV cache. At
   32k with 1 KV head and head_dim 512: `1 · 32768 · 512 · 2(K+V) ·
   2(uint8 path: 1)` — even fp16 that is ~32 MiB/layer × 7, and uint8
   KV (the `end-to-end` `--uint8-kv` lever) quarters it. Far inside
   the part-count budget that 19-part Qwen3-4B blew through.
3. **KV sharing** halves it again: 20 of 35 layers own no KV at all.
4. **1 KV head** vs Qwen3's multiple — another constant-factor cut.

So 32k (and likely the full 128k native window) is a realistic target
on Gemma 4 E2B in a way it never was on dense Qwen3-4B. The global
layers still want uint8 KV; the sliding layers are free.

## New risks Gemma introduces (not present in Qwen3)

1. **PLE graph surgery** — highest-uncertainty item. The second
   embedding table and per-layer residual injection have no Qwen3
   precedent; the pathb rewrite is new code, not a port.
2. **Multimodal wrapper export** — `optimum-cli` may try to trace the
   vision/audio towers. Need a clean text-decoder-only export
   (`text_config` task, or post-export tower strip).
3. **Dual + partial RoPE** — the rotary hoist is the most-rewritten
   stage; partial_rotary_factor=0.25 means rotary/pass-through dim
   splitting that Qwen3 never needed.
4. **No native AIMET `gemma` AdaScale adapter** — `model_type` falls
   back to `llama`; block detection must be verified with the
   find_blocks debug script before burning A100 hours.
5. **`E2B` is an elastic/MatFormer-style checkpoint.** The "effective
   2B" is a sub-network of a ~5.1B-param table. Confirm `optimum-cli`
   exports the full instantiated decoder (35 layers, 1536 hidden), not
   an elastic-slice artifact.

## Stage-by-stage delta (vs `end-to-end/quantize_to_npu.py`)

| stage | Qwen3 pipeline | Gemma pipeline change |
|---|---|---|
| 1 optimum export | `--task text-generation-with-past` | same task, but isolate the text decoder from the multimodal wrapper |
| 2–3 htp rewrite / fold-pathbmask | `rewrite_qwen3_htp.py` | new `rewrite_gemma4_htp.py` — Gemma norm chain, GeGLU, dual attention masks (sliding + causal) |
| 4 rotary hoist | `rewrite_qwen3_pathb.py` (single θ) | new `rewrite_gemma4_pathb.py` — dual cos/sin caches, partial-rotary dim split, PLE preservation |
| 5 pin shapes | `pin_shapes_qwen3_4b.py` | new `pin_shapes_gemma4.py` — per-layer-type head_dim, KV-sharing-aware input set |
| 6 AIMET | `model_type="qwen3"` | `model_type="llama"` fallback; expect ReduceMean-v18; verify block detection |
| 7 qairt-converter | unchanged | unchanged (graph-agnostic) |
| 8 qnn-context-binary-gen | HTP v75 config | unchanged; reuse `configs/` |
| 9 bundle | unchanged | unchanged |

Stages 7–9 are architecture-agnostic and reused as-is. Stages 2–6 are
the Gemma-specific work — see `scripts/README.md`.

## Open questions (resolved by doing)

- Does `optimum-cli` cleanly export the Gemma 4 text decoder, or does
  the multimodal wrapper need pre-surgery?
- Is the AIMET `llama` AdaScale adapter close enough for Gemma block
  detection, or do we need a `--no-use-ada-scale` SEQ_MSE-only run?
- Does Gemma 4 in `qai-hub-models` exist yet (model released
  2026-04-02)? If so the AI Hub cloud path (`submit_ai_hub.py`) may
  skip the local AIMET/QAIRT work entirely.
- Does partial RoPE survive the hoist numerically (cos ≥ 0.95)?
- Is the elastic-E2B export the full decoder or a slice artifact?
