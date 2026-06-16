import sys
from pathlib import Path
sys.path.insert(0, "/workspace/end-to-end")
from lib.bundle import assemble_genie_bundle
from lib.model_config import load_model_info

RUN = Path("/workspace/runs/qwen3_14b_w8a16")
BIN = RUN / "09_bin"
mi = load_model_info("Qwen/Qwen3-14B", Path("/workspace/models/Qwen3-14B"), precision="w8a16")
parts = sorted(BIN.glob("part*.bin"), key=lambda p: int(p.stem.replace("part","")))
print("parts (%d):" % len(parts), [p.name for p in parts])
bdir = RUN / "10_bundle" / "qwen3_14b-w8a16-specula-x2e"
assemble_genie_bundle(
    bin_paths=parts,
    bin_info_paths=[None]*len(parts),
    encodings_paths=[],
    tokenizer_dir=Path("/workspace/models/Qwen3-14B"),
    bundle_dir=bdir,
    bundle_name="qwen3_14b_w8a16",
    metadata={"model_id":"Qwen/Qwen3-14B","precision":"w8a16",
              "pipeline":"specula no-AIMET local build (Threadripper)","ctx":512,"num_parts":len(parts)},
    tar_out=RUN / "10_bundle" / "qwen3_14b-w8a16-specula-x2e.tar",
    model_info=mi, ctx=512, dsp_arch="v81", soc_model=88, rope_theta=mi.rope_theta,
)
print("BUNDLE_DONE", bdir)
