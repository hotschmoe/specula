import sys
from pathlib import Path
sys.path.insert(0, "/workspace/end-to-end")
import onnx
from onnx import TensorProto
from lib.split import PartSpec, extract_part
RUN = Path("/workspace/runs/qwen3_14b_w8a16")
src_onnx = RUN / "05_pathb_ctx512" / "model.onnx"
out_root = RUN / "06_split"
H=5120; KV=8; HD=128; CTX=512; PAST=511
def kv_in(s,e):
    it=[("position_ids_cos",TensorProto.FLOAT,[1,1,HD]),("position_ids_sin",TensorProto.FLOAT,[1,1,HD])]
    for li in range(s,e+1):
        it+=[("past_key_values.%d.key"%li,TensorProto.FLOAT,[1,KV,PAST,HD]),("past_key_values.%d.value"%li,TensorProto.FLOAT,[1,KV,PAST,HD])]
    it.append(("attention_bias",TensorProto.FLOAT,[1,1,1,CTX])); return it
def kv_out(s,e):
    it=[]
    for li in range(s,e+1):
        it+=[("present.%d.key"%li,TensorProto.FLOAT,[1,KV,CTX,HD]),("present.%d.value"%li,TensorProto.FLOAT,[1,KV,CTX,HD])]
    return it
def dec(name, s, e):
    return PartSpec(name=name,
        inputs=[("/model/layers.%d/Add_1_output_0"%(s-1),TensorProto.FLOAT,[1,1,H])]+kv_in(s,e),
        outputs=[("/model/layers.%d/Add_1_output_0"%e,TensorProto.FLOAT,[1,1,H])]+kv_out(s,e))
m = onnx.load(str(src_onnx), load_external_data=False)
for spec in (dec("part8",30,34), dec("part9",35,39)):
    info = extract_part(m, spec, src_onnx.parent, out_root/spec.name)
    print(spec.name, info.get("n_nodes"), "nodes,", round(info.get("data_size_gb",0),2),"GB")
print("RESPLIT2_DONE")
