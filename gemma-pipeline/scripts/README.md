# gemma-pipeline/scripts — Gemma-4-specific graph surgery

These four scripts are the only genuinely new code the Gemma pipeline
needs. Stages 1 and 7–9 of `quantize_to_npu.py` reuse existing,
architecture-agnostic tooling; stages 2–6 need Gemma-aware rewrites
because the Qwen3 scripts (`end-to-end/scripts/rewrite_qwen3_*`,
`pin_shapes_qwen3_4b.py`) hard-assume things Gemma 4 violates.

Build them in order — each depends on the previous one's output.
Validate every step with a CPU cos-vs-source probe (target cos ≥ 0.99,
the bar `end-to-end` holds) before moving on.

## 1. `rewrite_gemma4_htp.py`  (pipeline stages 2 + 3)

Port of `rewrite_qwen3_htp.py`. Two modes:

- `--mode stage` — strip the ops HTP op-config validation rejects:
  the BOOL attention-mask chain, `IsNaN` guards, `Cast→BOOL`. Gemma 4
  has **two** mask chains (sliding-window + global causal), so the
  count of guards is not Qwen3's "28 IsNaN, one per layer" — it is
  driven by `layer_types`. Read the layer-type map, not a constant.
- `--mode fold-pathbmask` — fold the additive causal mask in as a
  constant initializer. For Gemma, fold **both** masks (the 512-window
  sliding mask and the full causal mask) and tag each layer's
  attention with the right one.

New vs Qwen3: Gemma's norm chain (`(1+w)` RMSNorm, QK-norm, pre/post
feed-forward norms), GeGLU activation, and the dual mask. The
`final_logit_softcapping` (`30·tanh(x/30)`) at the lm_head must be
preserved or folded consistently — do not silently drop it.

## 2. `rewrite_gemma4_pathb.py`  (pipeline stage 4)

The hardest script. Port of `rewrite_qwen3_pathb.py` (rotary hoist),
but Gemma 4 needs three things Qwen3 never did:

- **Dual RoPE.** Build *two* cos/sin caches: global (θ=1e6,
  `proportional`) and sliding (θ=1e4, `default`). Route each layer to
  its cache via `layer_types`. The Qwen3 hoist has one cache.
- **Partial rotary.** Global layers have `partial_rotary_factor=0.25`
  — only the first 25% of `head_dim` is rotated, the rest passes
  through. The hoist must split rotary / pass-through dims and
  recombine. Sliding layers are full-rotary.
- **Per-Layer Embeddings (PLE).** The second `vocab×256` embedding
  table feeds a residual into every decoder layer. The rewrite must
  preserve this Gather + per-layer injection — it has no Qwen3
  precedent. This is the highest-uncertainty item; budget a full
  session and a dedicated equivalence probe.

Also handle **KV sharing**: the last `num_kv_shared_layers` (20 on
E2B) read an earlier layer's K/V and emit none of their own. The
hoisted graph's input/output set must reflect that — do not emit 35
independent past/present KV pairs.

`lib/rope.py` from `end-to-end/lib` is a starting point for the cos/sin
math but needs the `proportional` rope_type and `partial_rotary_factor`
added.

## 3. `pin_shapes_gemma4.py`  (pipeline stage 5)

Port of `pin_shapes_qwen3_4b.py`. Pins symbolic ONNX dims to AR=1 and
ctx=N so qairt-converter can lower the graph. Gemma deltas:

- **Per-layer-type head_dim.** Sliding layers use `head_dim=256`,
  global layers `global_head_dim=512`. The KV-cache tensor shapes
  differ by layer type — pin each accordingly.
- **KV-sharing-aware input set.** Only the `num_kv_owning_layers`
  (15 on E2B) carry `past_*`/`present_*` tensors to pin.
- **Rewrite the frozen `attention_mask` initializer to `[1, ctx]`** —
  this is the `end-to-end` session-31 bug fix (`pin_shapes` must
  rewrite the *initializer*, not just symbolic dims). Gemma has two
  masks; rewrite both.

## 4. (optional) `strip_multimodal_wrapper.py`  (pre-stage-1 helper)

Gemma 4's top-level arch is `Gemma4ForConditionalGeneration` — a
multimodal wrapper carrying vision + audio towers. If `optimum-cli`
will not cleanly export just the text decoder, this helper loads the
checkpoint, extracts `model.language_model` (the `gemma4_text`
decoder), and re-saves a text-only HF dir for stage 1 to export.
Try the plain `optimum-cli` export first; only build this if stage 1
fails on the wrapper.

## Validation harness (build alongside)

`probe_gemma4_equivalence.py` — load the rewritten ONNX and the HF
source on CPU, run both on a fixed prompt at pos=0 (zero KV) and pos=5
(synthetic past KV), report logit cosine similarity. The `end-to-end`
pipeline gates each rewrite at cos ≥ 0.99; hold the same bar. Partial
RoPE and PLE are the two places a silent numerical bug will hide.

## Reference

- `end-to-end/scripts/rewrite_qwen3_htp.py`,
  `rewrite_qwen3_pathb.py`, `pin_shapes_qwen3_4b.py` — the Qwen3
  originals to port from.
- `end-to-end/scripts/probe_pathb_equivalence.py` — the probe pattern.
- `../ARCHITECTURE_NOTES.md` — what differs and why.
