import sys, time
from pathlib import Path
sys.path.insert(0, "/workspace/end-to-end")
import onnx
from onnx import TensorProto
from lib import split as S
from lib.model_config import load_model_info

RUN = Path("/workspace/runs/qwen3_14b_w8a16")
src_onnx = RUN / "05_pathb_ctx512" / "model.onnx"
out_root = RUN / "06_split"
NUM_PARTS = 8
CTX = 512

mi = load_model_info("Qwen/Qwen3-14B", Path("/workspace/models/Qwen3-14B"), precision="w8a16")
print("model: L=%d H=%d kv=%d hd=%d vocab=%d" % (mi.num_hidden_layers, mi.hidden_size, mi.num_key_value_heads, mi.head_dim, mi.vocab_size))
m = onnx.load(str(src_onnx), load_external_data=False)
print("graph: %d nodes, %d inits" % (len(m.graph.node), len(m.graph.initializer)))
shared_mask = S.detect_shared_attn_mask(m)
print("shared attn mask: %s" % shared_mask)
specs = S.build_part_specs(num_layers=mi.num_hidden_layers, hidden_size=mi.hidden_size,
    vocab_size=mi.vocab_size, num_kv_heads=mi.num_key_value_heads, head_dim=mi.head_dim,
    ctx=CTX, num_parts=NUM_PARTS, shared_mask=shared_mask)
# This pathb (transformers 4.57 fold-pathbmask) uses a LIVE additive
# attention_bias graph input spliced into every layer Add, not the old
# internal folded mask the shared_mask path assumes. Declare it as a direct
# input on every decoder part (part2..N), like cos/sin.
for spec in specs[1:]:
    spec.inputs.append(("attention_bias", TensorProto.FLOAT, [1, 1, 1, CTX]))
print("part specs: %s" % [s.name for s in specs])
for spec in specs:
    dst = out_root / spec.name
    t0 = time.time()
    info = S.extract_part(m, spec, src_onnx.parent, dst)
    nn = info.get("n_nodes", "?"); dgb = info.get("data_size_gb", 0)
    print("  %s: %s nodes, %.2f GB, %.0fs" % (spec.name, nn, dgb, time.time() - t0))
print("SPLIT_DONE")
