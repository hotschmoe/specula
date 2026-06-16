"""CPU fp reference for the Qwen3-14B w8a16 NPU bundle.

Runs the original HF Qwen3-14B (bf16) on CPU for the bundle's
`sample_prompt.txt`, capturing the first-decode logits (the logits at the
last prompt position, which predict the first generated token). The NPU
engine's first-decode logits are compared against this by cosine similarity
to tell whether the no-calibration w8a16 bundle is numerically usable.

Also emits a short greedy continuation so we have a coherent-text anchor.

Run with the transformers-4.57 venv (matches the bundle's build-time
transformers, which emits the live `attention_bias`):

    .venv-arm-export/Scripts/python.exe npu_engine/ref_cpu_14b.py \
        --hf models/Qwen3-14B \
        --bundle models/qwen3_14b-w8a16-specula-x2e \
        --gen 32

Output: results/pathb_eval/ref_cpu_14b.npz
        (first_decode_logits[vocab], prompt_ids, gen_ids, gen_text)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

REPO = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO / "results" / "pathb_eval"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", type=Path, default=REPO / "models" / "Qwen3-14B")
    ap.add_argument("--bundle", type=Path,
                    default=REPO / "models" / "qwen3_14b-w8a16-specula-x2e")
    ap.add_argument("--prompt", type=Path, default=None,
                    help="defaults to <bundle>/sample_prompt.txt")
    ap.add_argument("--max-prompt", type=int, default=64,
                    help="cap prompt tokens (ctx=512, keep room for gen)")
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--tag", default="ref_cpu_14b")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    prompt_path = args.prompt or (args.bundle / "sample_prompt.txt")
    prompt = prompt_path.read_text(encoding="utf-8")
    print(f"=== CPU fp reference: {args.hf.name} ===")
    print(f"  prompt file: {prompt_path}")

    tok = AutoTokenizer.from_pretrained(str(args.hf))
    ids = tok(prompt, return_tensors="pt").input_ids[:, : args.max_prompt]
    print(f"  prompt tokens: {ids.shape[1]}")

    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(args.hf), torch_dtype=torch.bfloat16, attn_implementation="eager")
    model.eval()
    print(f"  model loaded in {time.perf_counter() - t0:.1f}s "
          f"({sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params)")

    # First-decode logits = logits at the last prompt position.
    with torch.no_grad():
        t1 = time.perf_counter()
        out = model(ids, use_cache=True)
        past = out.past_key_values
        first_logits = out.logits[0, -1].float().numpy().copy()
        print(f"  prefill ({ids.shape[1]} tok) in {time.perf_counter() - t1:.1f}s")

        # Greedy continuation for a coherent-text anchor.
        gen_ids = []
        cur = int(np.argmax(first_logits))
        for i in range(args.gen):
            gen_ids.append(cur)
            step = model(torch.tensor([[cur]]), past_key_values=past,
                         use_cache=True)
            past = step.past_key_values
            cur = int(step.logits[0, -1].argmax())
            if i % 8 == 0:
                print(f"  gen step {i}")

    gen_text = tok.decode(gen_ids)
    top5 = np.argsort(first_logits)[-5:][::-1]
    print(f"\n  first-decode argmax: {int(top5[0])} "
          f"({tok.decode([int(top5[0])])!r})")
    print(f"  top-5: {[(int(t), tok.decode([int(t)])) for t in top5]}")
    print(f"  continuation: {gen_text!r}")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_DIR / f"{args.tag}.npz"
    np.savez_compressed(
        out_path,
        first_decode_logits=first_logits,
        prompt_ids=ids[0].numpy().astype(np.int64),
        gen_ids=np.array(gen_ids, dtype=np.int64),
        gen_text=gen_text)
    print(f"\n  saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
