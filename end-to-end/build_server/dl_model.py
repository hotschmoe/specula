from huggingface_hub import snapshot_download
p = snapshot_download(
    "Qwen/Qwen3-14B",
    local_dir="/workspace/models/Qwen3-14B",
    allow_patterns=["*.safetensors", "*.json", "*.txt", "tokenizer*", "merges*", "vocab*"],
)
print("DOWNLOAD_DONE", p)
