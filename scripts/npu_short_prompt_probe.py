"""Phase 5 step 8 gate — short-prompt NPU correctness probe (Path B-mask).

Question: does Path B-mask's `attention_bias` correctly mask out
padding KV slots when the real prompt is much shorter than the
compiled past_len=511? This is the gate for step 8's spec-decode
outer loop, which MUST drive the NPU with short-prompt-shaped inputs
(a natural decode state is "20-token prompt, grow the KV from there").

Without this probe, the step 7 plumbing checkpoint only proves the NPU
works at the artificially-full past_len=511 anchor; short prompts
need the unused KV slots masked out via the FP16-minimum (-65504)
additive bias.

Design:

  1. Encode a short prompt from prompts/humaneval_subset.jsonl (p0 =
     ~20 tokens of a Fibonacci stub).
  2. CPU prefill on the optimum base ONNX (dynamic past_len) ->
     past_kv of length P + greedy-pick next_id at position P.
  3. CPU single-step at position P -> cpu_logits for position P+1.
     This is the ground truth.
  4. NPU single-step:
       - Pad past_kv slot dim: slots 0..P-1 real (from CPU), slots
         P..510 zero-filled. Shape [1, n_kv, 511, head_dim].
       - attention_bias [1, 1, 1, 512]:
           slots 0..P-1 : 0.0       (attend to real past)
           slots P..510 : -65504.0  (block attention to padding)
           slot 511     : 0.0       (self-attention to current token;
                                     the NPU's internal concat appends
                                     the current K/V to slot 511)
       - input_ids = [[next_id]], position_ids = [[P]].
  5. Compare cpu_logits vs npu_logits: argmax, top-5 overlap,
     cosine similarity, max |delta|.

Exit gate (both required):
  * cos >= 0.95 single-step
  * argmax match (both greedy-pick the same token)

If cos clears but argmax flips, top-5 overlap >= 3/5 is soft-acceptable
(quantization noise on a low-margin token). Flag loudly for investigation.

Run:
    .venv\\Scripts\\python.exe scripts\\npu_short_prompt_probe.py
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from npu_load_qwen3_bin import (  # noqa: E402
    CONTEXT_MAX,
    IS_LOCAL_COMPILE,
    LOGITS_OUTPUT_NAME,
    VARIANT,
    QuantSpec,
    _encodings_path,
    dequant_from_uint16,
    load_quant_specs,
    quant_to_uint16,
    rope_tables,
)
from npu_vs_cpu_correctness import (  # noqa: E402
    CONFIG_JSON,
    CPU_ONNX,
    TOKENIZER_JSON,
    compare_logits,
    cpu_build_feed,
    cpu_present_to_past,
    load_cpu_session,
    load_npu_session,
)

# The x86 local-compile variants (w4a16-local, fp16-local) both drop
# position_ids via `--remove_unused_inputs` and expose `logits` /
# `present_N_{key,value}` output names (no qairt-converter rename). The
# w4a16-local variant additionally quantizes every IO except input_ids
# to uint16; fp16-local keeps IO at fp32. The probe keeps an fp32-internal
# representation and quantizes only at the session-feed boundary for
# w4a16-local — symmetric dequant on the way back.
IS_W4A16_LOCAL = VARIANT == "w4a16-local"

HUMANEVAL = REPO_ROOT / "prompts" / "humaneval_subset.jsonl"

# FP16 minimum as specified in Qualcomm's additive-mask convention
# (also matches the Qwen3-4B Genie bundle's attention_bias quant range).
MASK_NEG = np.float32(-65504.0)


def _safe_repr(s: str) -> str:
    return repr(s.encode("ascii", "backslashreplace").decode("ascii"))


def pad_cpu_past_to_npu(
    cpu_past: dict[str, np.ndarray],
    prompt_len: int,
    cfg: dict,
) -> dict[str, np.ndarray]:
    """Extend a P-slot CPU past_kv to the 511-slot NPU past_kv by zero-padding.

    CPU past has shape [1, n_kv, prompt_len, head_dim] per layer; NPU
    wants [1, n_kv, CONTEXT_MAX-1, head_dim]. Slots [prompt_len..510]
    fill with zeros — those slots are what `attention_bias` masks out
    so the exact content doesn't matter.
    """
    n_layers = cfg["num_hidden_layers"]
    n_kv = cfg["num_key_value_heads"]
    head_dim = cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
    past_len = CONTEXT_MAX - 1
    pad_slots = past_len - prompt_len
    if pad_slots < 0:
        raise ValueError(f"prompt_len {prompt_len} exceeds NPU past_len {past_len}")

    npu_past: dict[str, np.ndarray] = {}
    for i in range(n_layers):
        k_real = cpu_past[f"past_key_values.{i}.key"]
        v_real = cpu_past[f"past_key_values.{i}.value"]
        if k_real.shape != (1, n_kv, prompt_len, head_dim):
            raise AssertionError(
                f"layer {i} key shape {k_real.shape} != expected "
                f"(1, {n_kv}, {prompt_len}, {head_dim})"
            )
        zeros_pad = np.zeros((1, n_kv, pad_slots, head_dim), dtype=np.float32)
        npu_past[f"past_key_values_{i}_key"] = np.concatenate(
            [k_real.astype(np.float32), zeros_pad], axis=2
        )
        npu_past[f"past_key_values_{i}_value"] = np.concatenate(
            [v_real.astype(np.float32), zeros_pad], axis=2
        )
    return npu_past


def build_masked_bias(prompt_len: int) -> np.ndarray:
    """Build attention_bias [1,1,1,512] with -65504 over padding KV slots.

    Layout of the NPU's internal seq_k = 512:
      * slots 0..510 correspond to the 511 past_kv entries we provide.
      * slot 511 is appended internally from the current token's K/V.

    For a prompt of length P:
      * slots 0..P-1          : valid past  -> 0.0
      * slots P..CONTEXT_MAX-2 : zero-padded past -> -65504.0
      * slot CONTEXT_MAX-1     : current token -> 0.0
    """
    bias = np.zeros((1, 1, 1, CONTEXT_MAX), dtype=np.float32)
    bias[:, :, :, prompt_len : CONTEXT_MAX - 1] = MASK_NEG
    return bias


def load_prompt(prompt_idx: int) -> str:
    with HUMANEVAL.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == prompt_idx:
                return json.loads(line)["prompt"]
    raise IndexError(f"humaneval prompt index {prompt_idx} out of range")


def cpu_prefill(
    sess: ort.InferenceSession,
    cfg: dict,
    prompt_ids: list[int],
) -> tuple[dict[str, np.ndarray], int]:
    """Run prefill, return (past_kv of length P, greedy next_id at position P)."""
    n_layers = cfg["num_hidden_layers"]
    n_kv = cfg["num_key_value_heads"]
    head_dim = cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
    seq_len = len(prompt_ids)

    out_names = [o.name for o in sess.get_outputs()]
    name_to_idx = {n: i for i, n in enumerate(out_names)}

    empty_kv = {
        f"past_key_values.{i}.{k}": np.zeros((1, n_kv, 0, head_dim), dtype=np.float32)
        for i in range(n_layers)
        for k in ("key", "value")
    }
    feed = cpu_build_feed(
        input_ids=np.array([prompt_ids], dtype=np.int64),
        position_ids=np.arange(seq_len, dtype=np.int64)[None, :],
        past_kv=empty_kv,
        total_seq_len=seq_len,
    )
    outputs = sess.run(None, feed)
    past_kv = cpu_present_to_past(outputs, name_to_idx, n_layers)
    next_id = int(np.argmax(outputs[name_to_idx["logits"]][0, -1]))
    return past_kv, next_id


def cpu_single_step(
    sess: ort.InferenceSession,
    cfg: dict,
    past_kv: dict[str, np.ndarray],
    input_id: int,
    position: int,
) -> np.ndarray:
    """Single decode step on CPU, return logits for position `position + 1`."""
    out_names = [o.name for o in sess.get_outputs()]
    name_to_idx = {n: i for i, n in enumerate(out_names)}
    past_len = next(iter(past_kv.values())).shape[2]
    feed = cpu_build_feed(
        input_ids=np.array([[input_id]], dtype=np.int64),
        position_ids=np.array([[position]], dtype=np.int64),
        past_kv=past_kv,
        total_seq_len=past_len + 1,
    )
    outputs = sess.run(None, feed)
    return outputs[name_to_idx["logits"]][0, -1]


def npu_single_step_short_prompt(
    sess: ort.InferenceSession,
    npu_past: dict[str, np.ndarray],
    input_id: int,
    position: int,
    valid_past_len: int,
    path_key: str = "pathbmask",
    quant_specs: dict[str, QuantSpec] | None = None,
) -> tuple[np.ndarray, list[np.ndarray], list[str]]:
    """Single decode step on NPU Path B-mask / Path B with masked bias.

    Feeds a 511-slot past (zero-padded beyond `valid_past_len`) and an
    attention_bias that -65504's out the padding slots. For path_key
    == "pathb", additionally feeds rope_tables(position) as
    position_ids_cos / position_ids_sin. Returns (logits,
    present_outputs, output_names) so the caller can reshape the
    present KV for the next step.

    `quant_specs` (w4a16-local only): if provided, every fp32 input is
    quantized per its spec before session.run, position_ids is dropped
    from the feed (not an input in that binary), and the uint16 logits
    are dequanted back to fp32 before the return.
    """
    out_names = [o.name for o in sess.get_outputs()]
    logits_idx = out_names.index(LOGITS_OUTPUT_NAME)
    feed: dict[str, np.ndarray] = {
        "input_ids": np.array([[input_id]], dtype=np.int32),
        "attention_bias": build_masked_bias(valid_past_len),
    }
    if not IS_LOCAL_COMPILE:
        # AI-Hub-compiled variants (patha / pathbmask / pathb fp16) expect
        # INT32 position_ids; both x86 local-compile variants used
        # --remove_unused_inputs and dropped it.
        feed["position_ids"] = np.array([[position]], dtype=np.int32)
    if path_key == "pathb":
        cos, sin = rope_tables(position)
        feed["position_ids_cos"] = cos
        feed["position_ids_sin"] = sin
    feed.update(npu_past)
    if quant_specs is not None:
        feed = _quantize_feed(feed, quant_specs)
    outputs = sess.run(None, feed)
    logits_raw = outputs[logits_idx][0, -1]
    if quant_specs is not None and LOGITS_OUTPUT_NAME in quant_specs:
        logits_raw = dequant_from_uint16(logits_raw, quant_specs[LOGITS_OUTPUT_NAME])
    return logits_raw, outputs, out_names


def _quantize_feed(
    feed_fp32: dict[str, np.ndarray],
    quant_specs: dict[str, QuantSpec],
) -> dict[str, np.ndarray]:
    """Return a copy of feed where every name in `quant_specs` is uint16-quantized.

    input_ids has no spec (int32 passthrough); any other name without a
    spec is a bug and we fail fast.
    """
    out: dict[str, np.ndarray] = {}
    for name, arr in feed_fp32.items():
        spec = quant_specs.get(name)
        if spec is None:
            if name != "input_ids":
                raise KeyError(f"no quant spec for w4a16-local input '{name}'")
            out[name] = arr
        else:
            out[name] = quant_to_uint16(arr, spec)
    return out


def npu_rearrange_present_to_past(
    present_outputs: list[np.ndarray],
    out_names: list[str],
    n_layers: int,
    old_valid_past_len: int,
    quant_specs: dict[str, QuantSpec] | None = None,
) -> dict[str, np.ndarray]:
    """Build the next-step past_kv after a short-prompt NPU decode step.

    The NPU's compiled seq_k layout is [511 past slots | 1 current slot].
    After a step at position P (valid_past_len=P), the outputs contain
    K/V of length 512:
      present[0..P-1]  = input past (unchanged)
      present[P..510]  = input zero-padding (unchanged)
      present[511]     = K/V for the token we just fed at position P

    To produce the next past (valid_past_len = P+1):
      new_past[0..P-1] = present[0..P-1]
      new_past[P]      = present[511]   (the token we just fed moves in)
      new_past[P+1..510] = zeros

    Local-compile routing: both fp16-local and w4a16-local expose outputs
    as `present_{i}_{key,value}` (no qairt-converter rename). w4a16-local
    additionally quantizes present with per-tensor scale/offset that
    differs from the input past_kv's scale/offset, so the caller keeps
    an fp32-internal npu_past and we dequant here; quant back to uint16
    happens at the next session-feed boundary. fp16-local passes fp32
    present straight through.
    """
    use_local_names = quant_specs is not None or IS_LOCAL_COMPILE
    use_w4a16 = quant_specs is not None
    next_past: dict[str, np.ndarray] = {}
    new_slot = old_valid_past_len
    for i in range(n_layers):
        if use_local_names:
            k_name = f"present_{i}_key"
            v_name = f"present_{i}_value"
        else:
            k_name = f"output_{2 * i + 1}"
            v_name = f"output_{2 * i + 2}"
        k = present_outputs[out_names.index(k_name)]
        v = present_outputs[out_names.index(v_name)]
        if use_w4a16:
            k = dequant_from_uint16(k, quant_specs[k_name])
            v = dequant_from_uint16(v, quant_specs[v_name])
        # Full 512-slot present from the NPU.
        if k.shape[2] != CONTEXT_MAX or v.shape[2] != CONTEXT_MAX:
            raise AssertionError(
                f"layer {i} present shape {k.shape}/{v.shape}, "
                f"expected seq_k={CONTEXT_MAX}"
            )
        next_k = np.zeros_like(k[:, :, : CONTEXT_MAX - 1, :])
        next_v = np.zeros_like(v[:, :, : CONTEXT_MAX - 1, :])
        if new_slot > 0:
            next_k[:, :, :new_slot, :] = k[:, :, :new_slot, :]
            next_v[:, :, :new_slot, :] = v[:, :, :new_slot, :]
        next_k[:, :, new_slot : new_slot + 1, :] = k[:, :, CONTEXT_MAX - 1 : CONTEXT_MAX, :]
        next_v[:, :, new_slot : new_slot + 1, :] = v[:, :, CONTEXT_MAX - 1 : CONTEXT_MAX, :]
        next_past[f"past_key_values_{i}_key"] = next_k
        next_past[f"past_key_values_{i}_value"] = next_v
    return next_past


def main() -> int:
    global print
    print = functools.partial(print, flush=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt-idx", type=int, default=0,
        help="humaneval fixture row index (default: 0 = fibonacci)",
    )
    parser.add_argument(
        "--path", choices=("pathbmask", "pathb"), default="pathbmask",
        help="compile graph variant (must match the loaded .bin).",
    )
    args = parser.parse_args()

    path_key = args.path
    print(f"=== step 8 gate - short-prompt NPU probe ({path_key}) ===\n")

    if not CPU_ONNX.exists():
        print(f"ERROR: {CPU_ONNX} missing")
        return 2

    with CONFIG_JSON.open() as f:
        cfg = json.load(f)
    tok = Tokenizer.from_file(str(TOKENIZER_JSON))

    prompt = load_prompt(args.prompt_idx)
    prompt_ids = tok.encode(prompt).ids
    prompt_len = len(prompt_ids)
    print(f"prompt (humaneval p{args.prompt_idx}, {prompt_len} tokens):")
    print(f"  {_safe_repr(prompt)}")
    print(f"  ids = {prompt_ids}")

    if prompt_len >= CONTEXT_MAX - 1:
        print(f"ERROR: prompt length {prompt_len} >= NPU past_len {CONTEXT_MAX - 1}")
        return 2

    print("\n--- loading CPU ONNX (FP32, dynamic past_len) ---")
    t0 = time.perf_counter()
    cpu_sess = load_cpu_session(CPU_ONNX)
    print(f"  loaded in {time.perf_counter() - t0:.1f} s")

    print("\n--- loading NPU Path B-mask binary ---")
    t0 = time.perf_counter()
    npu_sess = load_npu_session(cfg, path_key)
    print(f"  loaded in {time.perf_counter() - t0:.1f} s")
    providers = npu_sess.get_providers()
    if not providers or providers[0] != "QNNExecutionProvider":
        print(f"ERROR: NPU session fell back: {providers}")
        return 2

    quant_specs: dict[str, QuantSpec] | None = None
    if IS_W4A16_LOCAL:
        enc_path = _encodings_path(path_key)
        if not enc_path.exists():
            print(f"ERROR: {enc_path} missing (w4a16-local requires the quant encodings)")
            return 2
        runtime_names = (
            [x.name for x in npu_sess.get_inputs()]
            + [x.name for x in npu_sess.get_outputs()]
        )
        quant_specs = load_quant_specs(enc_path, runtime_names)
        print(f"  loaded {len(quant_specs)} quant specs (w4a16-local)")
    elif IS_LOCAL_COMPILE:
        # fp16-local variant: no quant layer, but the wrapper schema
        # differs from AI-Hub fp16 pathb (dropped position_ids, logits
        # output name). describe_inputs/outputs already handle this
        # via IS_LOCAL_COMPILE in npu_load_qwen3_bin.
        print(f"  {VARIANT}: fp32 IO, no quant_specs required")

    print(f"\n--- CPU prefill to past_len={prompt_len} ---")
    t0 = time.perf_counter()
    cpu_past, next_id = cpu_prefill(cpu_sess, cfg, prompt_ids)
    print(f"  prefill elapsed   : {time.perf_counter() - t0:.2f} s")
    print(f"  greedy next token : {next_id}   -> {_safe_repr(tok.decode([next_id]))}")

    print(f"\n--- CPU single-step at position {prompt_len} (ground truth) ---")
    cpu_logits = cpu_single_step(cpu_sess, cfg, cpu_past, next_id, prompt_len)
    cpu_argmax = int(np.argmax(cpu_logits))
    print(f"  cpu logits shape  : {cpu_logits.shape}")
    print(f"  cpu argmax        : {cpu_argmax}   -> {_safe_repr(tok.decode([cpu_argmax]))}")

    print(f"\n--- NPU single-step at position {prompt_len} "
          f"(slots {prompt_len}..{CONTEXT_MAX - 2} padded + masked) ---")
    npu_past = pad_cpu_past_to_npu(cpu_past, prompt_len, cfg)
    t0 = time.perf_counter()
    npu_logits, npu_outputs, npu_out_names = npu_single_step_short_prompt(
        npu_sess, npu_past, next_id, prompt_len,
        valid_past_len=prompt_len, path_key=path_key,
        quant_specs=quant_specs,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    npu_argmax = int(np.argmax(npu_logits))
    print(f"  npu step latency  : {elapsed_ms:.1f} ms")
    print(f"  npu logits shape  : {npu_logits.shape}")
    print(f"  npu argmax        : {npu_argmax}   -> {_safe_repr(tok.decode([npu_argmax]))}")

    print("\n--- comparison ---")
    stats = compare_logits(cpu_logits, npu_logits)
    print(f"  argmax match      : {bool(stats['argmax_match'])}  "
          f"(cpu={stats['cpu_argmax']}, npu={stats['npu_argmax']})")
    print(f"  top-5 overlap     : {stats['top5_overlap']} / 5")
    print(f"    cpu top5        : {stats['cpu_top5']}")
    print(f"    npu top5        : {stats['npu_top5']}")
    print(f"  cosine sim        : {stats['cosine_sim']:.6f}")
    print(f"  max |delta|       : {stats['max_abs_diff']:.4f}")

    cos_ok = stats["cosine_sim"] >= 0.95
    argmax_ok = bool(stats["argmax_match"])
    top5_acceptable = stats["top5_overlap"] >= 3

    # Multi-step probe: validate the rearrangement primitive the outer loop
    # needs. 3 consecutive NPU steps, each growing valid_past_len by 1,
    # against CPU greedy reference. The rearrangement moves slot 511 ->
    # slot P after each step so the "logical" KV layout matches what CPU
    # would have produced.
    n_layers = cfg["num_hidden_layers"]
    n_multi = 3
    print(f"\n--- multi-step extension ({n_multi} consecutive NPU steps) ---")

    # CPU reference: greedy-extend from (cpu_past length P + next_id at P).
    # Predicted token at P+1 is cpu_argmax. Continue greedy for 2 more steps.
    cpu_stream: list[int] = [cpu_argmax]
    cpu_past_iter = {k: v.copy() for k, v in cpu_past.items()}
    # Append next_id (at position P) to cpu_past_iter so past_len -> P+1.
    # Easiest: run CPU single-step with next_id at position P, capture
    # present, then iterate.
    cpu_out_names = [o.name for o in cpu_sess.get_outputs()]
    cpu_name_to_idx = {n: i for i, n in enumerate(cpu_out_names)}
    feed = cpu_build_feed(
        input_ids=np.array([[next_id]], dtype=np.int64),
        position_ids=np.array([[prompt_len]], dtype=np.int64),
        past_kv=cpu_past_iter,
        total_seq_len=prompt_len + 1,
    )
    cpu_out = cpu_sess.run(None, feed)
    cpu_past_iter = cpu_present_to_past(cpu_out, cpu_name_to_idx, n_layers)
    # cpu_stream[0] already = cpu_argmax (greedy at position P+1). Continue.
    cpu_input = cpu_argmax
    cpu_pos = prompt_len + 1
    for _ in range(n_multi - 1):
        feed = cpu_build_feed(
            input_ids=np.array([[cpu_input]], dtype=np.int64),
            position_ids=np.array([[cpu_pos]], dtype=np.int64),
            past_kv=cpu_past_iter,
            total_seq_len=cpu_pos + 1,
        )
        cpu_out = cpu_sess.run(None, feed)
        cpu_past_iter = cpu_present_to_past(cpu_out, cpu_name_to_idx, n_layers)
        next_cpu = int(np.argmax(cpu_out[cpu_name_to_idx["logits"]][0, -1]))
        cpu_stream.append(next_cpu)
        cpu_input = next_cpu
        cpu_pos += 1

    # NPU multi-step with rearrangement.
    npu_stream: list[int] = [npu_argmax]
    next_past = npu_rearrange_present_to_past(
        npu_outputs, npu_out_names, n_layers, old_valid_past_len=prompt_len,
        quant_specs=quant_specs,
    )
    npu_input = npu_argmax
    npu_pos = prompt_len + 1
    npu_valid = prompt_len + 1
    for _ in range(n_multi - 1):
        step_logits, step_outputs, step_out_names = npu_single_step_short_prompt(
            npu_sess, next_past, npu_input, npu_pos,
            valid_past_len=npu_valid, path_key=path_key,
            quant_specs=quant_specs,
        )
        nxt = int(np.argmax(step_logits))
        npu_stream.append(nxt)
        next_past = npu_rearrange_present_to_past(
            step_outputs, step_out_names, n_layers, old_valid_past_len=npu_valid,
            quant_specs=quant_specs,
        )
        npu_input = nxt
        npu_pos += 1
        npu_valid += 1

    multi_matches = [int(a == b) for a, b in zip(cpu_stream, npu_stream)]
    multi_rate = sum(multi_matches) / n_multi
    print(f"  CPU stream    : {cpu_stream}  -> {_safe_repr(tok.decode(cpu_stream))}")
    print(f"  NPU stream    : {npu_stream}  -> {_safe_repr(tok.decode(npu_stream))}")
    print(f"  per-step match: {multi_matches}")
    print(f"  match rate    : {multi_rate * 100:.0f}% ({sum(multi_matches)}/{n_multi})")
    multi_ok = multi_rate >= 0.66

    print("\n=== STATUS ===")
    print(f"  cos >= 0.95            : {cos_ok} ({stats['cosine_sim']:.4f})")
    print(f"  argmax match           : {argmax_ok}")
    print(f"  top-5 overlap >= 3/5   : {top5_acceptable} ({stats['top5_overlap']}/5)")
    print(f"  multi-step rate >= 66% : {multi_ok} ({multi_rate * 100:.0f}%)")
    if cos_ok and argmax_ok and multi_ok:
        print("  overall                : ok (step 8 outer loop unblocked)")
        return 0
    if cos_ok and top5_acceptable:
        print("  overall                : partial - cos holds but argmax flipped; "
              "low-margin token, investigate before step 8")
        return 1
    print("  overall                : FAIL - Path B-mask masking broken, step 8 blocked")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
