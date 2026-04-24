"""Generate a PP512 prompt: text that tokenizes to exactly 512 tokens
under the Qualcomm Qwen3-4B bundle's tokenizer.

Writes:
    results/qwen3_4b_baseline/pp512_prompt.txt       - chat-wrapped text
    results/qwen3_4b_baseline/pp512_prompt_tokens.txt - one token id per line

Both llama.cpp and Genie use the same Qwen3 BPE tokenizer (Qwen3 4B ships
one vocab); the files tokenize identically in either runtime.

Run:
    .venv/Scripts/python.exe scripts/gen_pp512_prompt.py
"""
from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer


REPO_ROOT = Path(__file__).resolve().parent.parent
TOKENIZER_PATH = (
    REPO_ROOT / "models" / "qualcomm-qwen3-4b-ref"
    / "qwen3_4b-genie-w4a16-qualcomm_snapdragon_x2_elite" / "tokenizer.json"
)
OUT_DIR = REPO_ROOT / "results" / "qwen3_4b_baseline"
OUT_TXT = OUT_DIR / "pp512_prompt.txt"
OUT_IDS = OUT_DIR / "pp512_prompt_tokens.txt"

# Filler content: a concrete, model-family-neutral technical paragraph
# repeated and then trimmed to exactly 512 BPE tokens. Using concrete
# content (rather than lorem ipsum) ensures the tokenizer produces a
# realistic distribution of subwords, matching a real long-prompt
# workload.
FILLER = (
    "Speculative decoding combines a small draft model with a large "
    "target model. The draft proposes several tokens per step; the "
    "target verifies them in a single batched forward pass. If the "
    "draft's next-token distribution agrees with the target's, the "
    "tokens are accepted without further target computation. On "
    "Snapdragon X2 Elite Extreme, the Hexagon NPU runs the draft while "
    "the CPU runs the target through llama.cpp. The Adreno GPU handles "
    "prompt processing through OpenCL-tuned matmul kernels. Unified "
    "LPDDR5X memory at 228 GB per second lets all three compute "
    "islands share weights without copying. This reduces the cost of "
    "the heterogeneous handoff to a cache-line flush rather than a "
    "direct memory access. Quantization plays a central role: the "
    "4-bit weight 16-bit activation format shrinks the NPU memory "
    "footprint to roughly a quarter of the bfloat16 baseline while "
    "the HMX vector engine runs the dequantized tensors efficiently. "
    "When the weights are quantized with mixed precision where "
    "attention value and output projections remain at 8-bit, the "
    "accepted rate climbs back toward the floating-point baseline. "
    "The combined effect across prefill and decode is a speedup that "
    "scales with the accept rate. "
)

CHAT_WRAP = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n{body}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def main() -> int:
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))

    # Build text until it tokenizes to >= 512 tokens, then trim.
    body = FILLER
    while True:
        wrapped = CHAT_WRAP.format(body=body)
        ids = tok.encode(wrapped, add_special_tokens=False).ids
        if len(ids) >= 512:
            break
        body += FILLER

    # Trim to exactly 512 tokens. Decode the first 512 back to text;
    # the chat header may have been partially consumed, so we rebuild
    # cleanly from the decoded slice and let the downstream runtimes
    # re-tokenize to the same 512-token length (BPE is deterministic
    # given the decoded string).
    trimmed_ids = ids[:512]
    trimmed_text = tok.decode(trimmed_ids, skip_special_tokens=False)

    # Sanity: re-encode the trimmed text and confirm token count.
    recheck_ids = tok.encode(trimmed_text, add_special_tokens=False).ids
    print(f"target tokens : 512")
    print(f"decoded bytes : {len(trimmed_text.encode('utf-8'))}")
    print(f"recheck tokens: {len(recheck_ids)}")
    if len(recheck_ids) != 512:
        print("WARNING: re-encode count drifted — trimming to shortest safe length")
        # Fall back: decode from len-N where N makes re-encode == 512.
        # In practice BPE decode→encode is idempotent on clean boundaries,
        # but subword fragments at the trim seam can shift count by ±1.
        # Shave one token and retry a few times.
        for i in range(1, 6):
            alt = tok.decode(trimmed_ids[:-i], skip_special_tokens=False)
            if len(tok.encode(alt, add_special_tokens=False).ids) == 512 - i:
                trimmed_text = alt
                recheck_ids = tok.encode(alt, add_special_tokens=False).ids
                print(f"  settled at {len(recheck_ids)} tokens after trimming {i} ids")
                break

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text(trimmed_text, encoding="utf-8")
    OUT_IDS.write_text("\n".join(str(i) for i in recheck_ids), encoding="utf-8")
    print(f"wrote {OUT_TXT} ({len(trimmed_text)} chars, {len(recheck_ids)} tokens)")
    print(f"wrote {OUT_IDS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
