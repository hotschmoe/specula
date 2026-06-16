"""Emit a qairt-converter --config YAML forcing float32 on every IO tensor.

With calibration (--input_list), the quantizer otherwise quantizes IO to
uint16, breaking the FLOAT_32 seam contract (and the attention_bias mask).
--config has higher precedence than --preserve_io_datatype, so we pin every
input/output DataType to float32 here. Run in the QAIRT container.

    python3 make_io_config.py <part_dir>/model.onnx <out.yaml>
"""
import sys
import onnx

m = onnx.load(sys.argv[1], load_external_data=False)
ins = [i.name for i in m.graph.input]
outs = [o.name for o in m.graph.output]


def block(name):
    return (f"  - Name: {name}\n"
            f"    Desired Model Parameters:\n"
            f"        DataType: float32\n")


with open(sys.argv[2], "w") as f:
    f.write("Input Tensor Configuration:\n")
    for n in ins:
        f.write(block(n))
    f.write("Output Tensor Configuration:\n")
    for n in outs:
        f.write(block(n))
print(f"wrote {sys.argv[2]}: {len(ins)} inputs + {len(outs)} outputs -> float32")
