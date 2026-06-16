import sys
from pathlib import Path
sys.path.insert(0, "/workspace/end-to-end")
import onnx
from onnx import TensorProto
from lib.split import PartSpec, extract_part

RUN = Path("/workspace/runs/qwen3_14b_w8a16")
src_onnx = RUN / "05_pathb_ctx512" / "model.onnx"
out_root = RUN / "06_split"
H=5120; KV=8; HD=128; CTX=512; VOCAB=151936; PAST=511

def kv_in(s,e):
    it=[("position_ids_cos",TensorProto.FLOAT,[1,1,HD]),("position_ids_sin",TensorProto.FLOAT,[1,1,HD])]
    for li in range(s,e+1):
        it.append(("past_key_values.%d.key"%li,TensorProto.FLOAT,[1,KV,PAST,HD]))
        it.append(("past_key_values.%d.value"%li,TensorProto.FLOAT,[1,KV,PAST,HD]))
    it.append(("attention_bias",TensorProto.FLOAT,[1,1,1,CTX]))
    return it
def kv_out(s,e):
    it=[]
    for li in range(s,e+1):
        it.append(("present.%d.key"%li,TensorProto.FLOAT,[1,KV,CTX,HD]))
        it.append(("present.%d.value"%li,TensorProto.FLOAT,[1,KV,CTX,HD]))
    return it

# part8: layers 30-39 (decoder only, NO lm_head)
part8 = PartSpec(name="part8",
    inputs=[("/model/layers.29/Add_1_output_0",TensorProto.FLOAT,[1,1,H])]+kv_in(30,39),
    outputs=[("/model/layers.39/Add_1_output_0",TensorProto.FLOAT,[1,1,H])]+kv_out(30,39))
# part9: final_norm + lm_head only (no attention -> no rotary -> no 127 issue)
part9 = PartSpec(name="part9",
    inputs=[("/model/layers.39/Add_1_output_0",TensorProto.FLOAT,[1,1,H])],
    outputs=[("logits",TensorProto.FLOAT,[1,1,VOCAB])])

m = onnx.load(str(src_onnx), load_external_data=False)
for spec in (part8, part9):
    info = extract_part(m, spec, src_onnx.parent, out_root/spec.name)
    print(spec.name, info.get("n_nodes"), "nodes,", round(info.get("data_size_gb",0),2), "GB")
print("RESPLIT_DONE")
