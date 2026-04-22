"""Phase 5.5 Lever C - capture calibration samples for AI Hub W4A16 compile.

Runs CPU FP32 ONNX (Qwen3-0.6B, dynamic past_len) on the existing prompt
fixtures, snapshots model inputs at selected decode positions, and saves
them in the exact shape AI Hub's pathbmask compile expects for its
`calibration_data` kwarg on `submit_compile_job`.

Two bundles (research question: does cheap step-0-only calibration match
realistic multi-position calibration?):

    --bundle A    3 decode positions per prompt (steps 5, 25, 60).
                  60 samples total. ~3.4 GB at ctx=256.

    --bundle B    1 decode position per prompt (step 0 only).
                  20 samples total. ~1.1 GB at ctx=256.

Output format: single .npz with each ONNX input key as a stacked array of
shape [n_samples, ...]. Reconstruct the DatasetEntries dict via
`load_dataset_entries(npz_path)` at upload time.

Note: this script does NOT use the NPU. Everything is CPU FP32 ONNX.
Shapes are driven by --ctx (= CONTEXT_MAX) to match what the AI Hub
compile will see in `input_specs`.

Run:
    .venv\\Scripts\\python.exe scripts\\capture_calibration_samples.py --bundle A --ctx 256 --out models\\calibration\\bundle_a_ctx256.npz
    .venv\\Scripts\\python.exe scripts\\capture_calibration_samples.py --bundle B --ctx 256 --out models\\calibration\\bundle_b_ctx256.npz

Add --upload to push to AI Hub as a named dataset after local save.
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
HUMANEVAL = REPO_ROOT / "prompts" / "humaneval_subset.jsonl"
STRUCTURED = REPO_ROOT / "prompts" / "structured_json.jsonl"

# Qwen3-0.6B optimum export — shares lineage with the NPU's source ONNX, so
# the KV state produced here is directly interchangeable with NPU KV state.
# (Matches the reasoning in npu_vs_cpu_correctness.py.)
CPU_ONNX_DIR = REPO_ROOT / "models" / "qwen3-0.6b-optimum"
CPU_ONNX = CPU_ONNX_DIR / "model.onnx"
TOKENIZER_JSON = CPU_ONNX_DIR / "tokenizer.json"
CONFIG_JSON = CPU_ONNX_DIR / "config.json"

# FP16-minimum additive mask, matches Qualcomm's pathbmask convention.
MASK_NEG = np.float32(-65504.0)

BUNDLE_STEPS = {
    "A": [5, 25, 60],
    "B": [0],
}


def _safe_repr(s: str) -> str:
    return repr(s.encode("ascii", "backslashreplace").decode("ascii"))


def load_all_prompts() -> list[tuple[str, str]]:
    """Return [(source, prompt), ...] from humaneval + structured_json."""
    prompts: list[tuple[str, str]] = []
    for path in (HUMANEVAL, STRUCTURED):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                prompts.append((path.stem, json.loads(line)["prompt"]))
    return prompts


def load_cpu_session(onnx_path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"]
    )


def cpu_build_feed(
    input_ids: np.ndarray,
    position_ids: np.ndarray,
    past_kv: dict[str, np.ndarray],
    total_seq_len: int,
) -> dict[str, np.ndarray]:
    feed: dict[str, np.ndarray] = {
        "input_ids": input_ids.astype(np.int64),
        "position_ids": position_ids.astype(np.int64),
        "attention_mask": np.ones((1, total_seq_len), dtype=np.int64),
    }
    feed.update(past_kv)
    return feed


def cpu_present_to_past(
    outputs: list[np.ndarray],
    name_to_idx: dict[str, int],
    n_layers: int,
) -> dict[str, np.ndarray]:
    past: dict[str, np.ndarray] = {}
    for i in range(n_layers):
        past[f"past_key_values.{i}.key"] = outputs[name_to_idx[f"present.{i}.key"]]
        past[f"past_key_values.{i}.value"] = outputs[name_to_idx[f"present.{i}.value"]]
    return past


def cpu_prefill(
    sess: ort.InferenceSession,
    cfg: dict,
    prompt_ids: list[int],
) -> tuple[dict[str, np.ndarray], int]:
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


def cpu_decode_step(
    sess: ort.InferenceSession,
    cfg: dict,
    past_kv: dict[str, np.ndarray],
    input_id: int,
    position: int,
) -> tuple[dict[str, np.ndarray], int]:
    out_names = [o.name for o in sess.get_outputs()]
    name_to_idx = {n: i for i, n in enumerate(out_names)}
    past_len = next(iter(past_kv.values())).shape[2]
    feed = cpu_build_feed(
        input_ids=np.array([[input_id]], dtype=np.int64),
        position_ids=np.array([[position]], dtype=np.int64),
        past_kv=past_kv,
        total_seq_len=past_len + 1,
    )
    out = sess.run(None, feed)
    new_past = cpu_present_to_past(out, name_to_idx, cfg["num_hidden_layers"])
    next_id = int(np.argmax(out[name_to_idx["logits"]][0, -1]))
    return new_past, next_id


def pad_past_to_npu_ctx(
    past_kv_dotted: dict[str, np.ndarray],
    valid_len: int,
    ctx: int,
    cfg: dict,
) -> dict[str, np.ndarray]:
    """Pad real past_kv of length `valid_len` to `ctx-1` slots with zeros.

    Returns a dict with DOTTED names matching ONNX graph input names
    (which is what AI Hub's input_specs + calibration_data expect).
    """
    n_layers = cfg["num_hidden_layers"]
    n_kv = cfg["num_key_value_heads"]
    head_dim = cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
    past_len = ctx - 1
    pad = past_len - valid_len
    if pad < 0:
        raise ValueError(
            f"valid_len {valid_len} exceeds compile past_len {past_len} (ctx={ctx})"
        )

    out: dict[str, np.ndarray] = {}
    zeros = np.zeros((1, n_kv, pad, head_dim), dtype=np.float32)
    for i in range(n_layers):
        k_real = past_kv_dotted[f"past_key_values.{i}.key"].astype(np.float32)
        v_real = past_kv_dotted[f"past_key_values.{i}.value"].astype(np.float32)
        out[f"past_key_values.{i}.key"] = np.concatenate([k_real, zeros], axis=2)
        out[f"past_key_values.{i}.value"] = np.concatenate([v_real, zeros], axis=2)
    return out


def build_attention_bias(valid_past_len: int, ctx: int) -> np.ndarray:
    """Additive bias: real slots 0.0, padded slots -65504, current-slot 0.0."""
    bias = np.zeros((1, 1, 1, ctx), dtype=np.float32)
    bias[:, :, :, valid_past_len : ctx - 1] = MASK_NEG
    return bias


def capture_for_prompt(
    sess: ort.InferenceSession,
    cfg: dict,
    tok: Tokenizer,
    prompt: str,
    decode_steps: list[int],
    ctx: int,
) -> dict[int, dict[str, np.ndarray]]:
    """Prefill prompt, greedy-decode up to max(decode_steps), snapshot inputs
    at each requested step. Returns {step_idx: {input_name: ndarray}}.

    "decode step N" = the Nth single-step decode after prefill. Step 0 inputs:
    past_kv length = prompt_len, position = prompt_len, input = argmax from
    prefill. Step 1 inputs: past_kv length = prompt_len+1, position = prompt_len+1,
    etc.
    """
    prompt_ids = tok.encode(prompt).ids
    P = len(prompt_ids)

    max_step = max(decode_steps)
    if P + max_step > ctx - 1:
        raise ValueError(
            f"prompt_len {P} + max_step {max_step} = {P + max_step} > past_len "
            f"{ctx - 1}; drop step or use larger ctx"
        )

    past_kv, next_id = cpu_prefill(sess, cfg, prompt_ids)

    cur_input = next_id
    cur_pos = P
    cur_past = past_kv

    samples: dict[int, dict[str, np.ndarray]] = {}
    for step in range(max_step + 1):
        valid_past_len = cur_past["past_key_values.0.key"].shape[2]
        if step in decode_steps:
            sample: dict[str, np.ndarray] = {
                "input_ids": np.array([[cur_input]], dtype=np.int64),
                "position_ids": np.array([[cur_pos]], dtype=np.int64),
                "attention_bias": build_attention_bias(valid_past_len, ctx),
            }
            sample.update(pad_past_to_npu_ctx(cur_past, valid_past_len, ctx, cfg))
            samples[step] = sample
        if step == max_step:
            break
        cur_past, nxt = cpu_decode_step(sess, cfg, cur_past, cur_input, cur_pos)
        cur_input = nxt
        cur_pos += 1

    return samples


def save_bundle(
    accum: dict[str, list[np.ndarray]],
    out_path: Path,
    ctx: int,
    bundle: str,
    decode_steps: list[int],
    sample_meta: list[dict],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_samples = len(next(iter(accum.values())))
    for k, lst in accum.items():
        if len(lst) != n_samples:
            raise AssertionError(
                f"input '{k}' has {len(lst)} samples, expected {n_samples}"
            )

    stacked = {k: np.stack(v, axis=0) for k, v in accum.items()}
    total_gb = sum(a.nbytes for a in stacked.values()) / (1024**3)

    print(f"\nstacking {n_samples} samples across {len(stacked)} inputs ...")
    for k in sorted(stacked.keys()):
        if k.startswith("past_key_values.") and not k.endswith(".0.key"):
            continue
        print(f"  {k:40s} {stacked[k].shape}  {stacked[k].dtype}")
    if any(k.startswith("past_key_values.") for k in stacked):
        pkv_keys = [k for k in stacked if k.startswith("past_key_values.")]
        print(f"  ... and {len(pkv_keys) - 1} more past_kv entries")

    print(f"\nsaving {out_path} ({total_gb:.2f} GB uncompressed) ...")
    t0 = time.perf_counter()
    np.savez(str(out_path), **stacked)
    print(f"  wrote in {time.perf_counter() - t0:.1f} s")

    manifest = {
        "ctx": ctx,
        "bundle": bundle,
        "decode_steps": decode_steps,
        "n_samples": n_samples,
        "input_shapes": {k: list(stacked[k].shape) for k in sorted(stacked)},
        "input_dtypes": {k: str(stacked[k].dtype) for k in sorted(stacked)},
        "samples": sample_meta,
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"manifest: {manifest_path}")


def load_dataset_entries(npz_path: Path) -> dict[str, list[np.ndarray]]:
    """Invert save_bundle: [n_samples, ...] stacked array -> list of ndarrays."""
    loaded = np.load(str(npz_path))
    entries: dict[str, list[np.ndarray]] = {}
    for k in loaded.files:
        arr = loaded[k]
        entries[k] = [arr[i] for i in range(arr.shape[0])]
    return entries


def upload_to_ai_hub(npz_path: Path, name: str) -> None:
    import qai_hub as hub

    print(f"\nloading {npz_path} back into DatasetEntries ...")
    entries = load_dataset_entries(npz_path)
    n_samples = len(next(iter(entries.values())))
    print(f"  {len(entries)} inputs × {n_samples} samples")

    print(f"uploading as dataset '{name}' ...")
    t0 = time.perf_counter()
    ds = hub.upload_dataset(entries, name=name)
    print(f"  uploaded in {time.perf_counter() - t0:.1f} s, dataset_id={ds.dataset_id}")

    ref_path = npz_path.with_suffix(".dataset_ref.json")
    with ref_path.open("w") as f:
        json.dump({"dataset_id": ds.dataset_id, "dataset_name": name}, f, indent=2)
    print(f"dataset ref: {ref_path}")


def main() -> int:
    global print
    print = functools.partial(print, flush=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", choices=("A", "B"), required=True)
    parser.add_argument("--ctx", type=int, default=256)
    parser.add_argument("--out", type=Path, required=True, help="output .npz path")
    parser.add_argument(
        "--upload", action="store_true",
        help="also upload to AI Hub as a named dataset after local save",
    )
    parser.add_argument(
        "--dataset-name", default=None,
        help="dataset name on AI Hub (default: derived from --out stem)",
    )
    args = parser.parse_args()

    decode_steps = BUNDLE_STEPS[args.bundle]
    print(f"=== Lever C calibration capture ===")
    print(f"  bundle       : {args.bundle}")
    print(f"  ctx          : {args.ctx}  (past_len={args.ctx - 1})")
    print(f"  decode_steps : {decode_steps}")
    print(f"  out          : {args.out}")

    if not CPU_ONNX.exists():
        print(f"ERROR: {CPU_ONNX} missing")
        return 2

    with CONFIG_JSON.open() as f:
        cfg = json.load(f)
    tok = Tokenizer.from_file(str(TOKENIZER_JSON))

    prompts = load_all_prompts()
    print(f"  prompts      : {len(prompts)} from humaneval + structured_json")
    expected = len(prompts) * len(decode_steps)
    print(f"  total samples: {expected}")

    print(f"\n--- loading CPU ONNX ---")
    t0 = time.perf_counter()
    sess = load_cpu_session(CPU_ONNX)
    print(f"  loaded in {time.perf_counter() - t0:.1f} s")

    accum: dict[str, list[np.ndarray]] = {}
    sample_meta: list[dict] = []

    t_start = time.perf_counter()
    for i, (src, prompt) in enumerate(prompts):
        t0 = time.perf_counter()
        per_prompt = capture_for_prompt(sess, cfg, tok, prompt, decode_steps, args.ctx)
        elapsed = time.perf_counter() - t0
        P = len(tok.encode(prompt).ids)
        for step_idx in sorted(per_prompt):
            sample = per_prompt[step_idx]
            for key, val in sample.items():
                accum.setdefault(key, []).append(val)
            sample_meta.append({
                "source": src,
                "prompt_idx_in_source": i % 10,
                "prompt_len": P,
                "decode_step": step_idx,
                "prompt_preview": _safe_repr(prompt[:60]),
            })
        print(f"  prompt {i+1:2d}/{len(prompts)} [{src[:10]:10s}] P={P:3d} "
              f"captured={sorted(per_prompt)} ({elapsed:.1f}s)")

    total_elapsed = time.perf_counter() - t_start
    print(f"\n  capture total: {total_elapsed:.1f} s "
          f"({total_elapsed / expected:.2f} s/sample)")

    save_bundle(accum, args.out, args.ctx, args.bundle, decode_steps, sample_meta)

    if args.upload:
        name = args.dataset_name or args.out.stem
        upload_to_ai_hub(args.out, name)

    print(f"\n=== STATUS: ok ===")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
