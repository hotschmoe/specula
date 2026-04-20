"""Phase 5 step 3a - fetch Qwen3-0.6B ONNX from onnx-community.

Skips the optimum/torch export path (torch has no cp312 win_arm64 wheel,
see scripts/npu_draft_sidecar.py session notes). The onnx-community repo
publishes a canonical FP32 export with KV cache as IO tensors, which is
exactly what AI Hub wants as input for step 4.

Downloads:
  onnx/model.onnx       - 300 MB graph
  onnx/model.onnx_data  - 2 GB external-data weights
  config.json           - model architecture metadata
  tokenizer.json        - HF tokenizer (BPE)
  tokenizer_config.json, special_tokens_map.json, vocab.json, merges.txt

Target dir: models/qwen3-0.6b-onnx/

Idempotent: huggingface_hub caches and skips completed files on re-run.
"""

import sys
from pathlib import Path

from huggingface_hub import snapshot_download


MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
TARGET = MODELS_DIR / "qwen3-0.6b-onnx"


def main() -> int:
    TARGET.mkdir(parents=True, exist_ok=True)
    print(f"target : {TARGET}")
    print("downloading onnx-community/Qwen3-0.6B-ONNX ...")

    local = snapshot_download(
        repo_id="onnx-community/Qwen3-0.6B-ONNX",
        local_dir=str(TARGET),
        allow_patterns=[
            "onnx/model.onnx",
            "onnx/model.onnx_data",
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "chat_template.jinja",
            "added_tokens.json",
            "README.md",
        ],
    )
    print(f"local  : {local}")

    model_path = TARGET / "onnx" / "model.onnx"
    data_path = TARGET / "onnx" / "model.onnx_data"
    for p in (model_path, data_path):
        if not p.exists():
            print(f"MISSING: {p}")
            return 2
        print(f"  {p.name:20s}  {p.stat().st_size / (1024*1024):8.1f} MB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
