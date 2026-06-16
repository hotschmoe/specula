"""Strip the MTP (multi-token-prediction) head from a Qwen3.6 GGUF.

The Qwen3.6-27B-MTP GGUF ships an extra block (the highest blk index, e.g.
blk.64 with `nextn.*` tensors) that is the speculative-decoding head, not part
of the N-layer model. llama.cpp's qwen35 loader strictly requires every file
tensor to be used (`wrong number of tensors; expected X, got Y`), so we drop
the whole top block and set `<arch>.block_count` to the real layer count.

    .venv-arm-export/Scripts/python.exe scripts/strip_mtp_head.py \
        models/Qwen3.6-27B-MTP-Q4_0.gguf models/Qwen3.6-27B-Q4_0.gguf
"""
import sys
sys.path.insert(0, r"C:\Users\hotschmoe\Documents\GitHub\llama.cpp\gguf-py")
import gguf  # noqa: E402

inp, outp = sys.argv[1], sys.argv[2]
r = gguf.GGUFReader(inp)

arch = None
for f in r.fields.values():
    if f.name == "general.architecture":
        arch = bytes(f.parts[f.data[0]]).decode()
assert arch, "no general.architecture"

# Highest block index = the MTP head to drop.
blk_idx = sorted({int(t.name.split(".")[1]) for t in r.tensors
                  if t.name.startswith("blk.")})
top = blk_idx[-1]
new_count = top  # real layers are 0..top-1
drop_prefix = f"blk.{top}."
keep = [t for t in r.tensors if not t.name.startswith(drop_prefix)]
print(f"arch={arch} top_block={top} -> block_count={new_count}; "
      f"dropping {len(r.tensors) - len(keep)} tensors, keeping {len(keep)}")

w = gguf.GGUFWriter(outp, arch)
for field in r.fields.values():
    if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
        continue
    vt = field.types[0]
    sub = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
    val = field.contents()
    if field.name == f"{arch}.block_count":
        val = new_count
    if val is not None:
        w.add_key_value(field.name, val, vt, sub_type=sub)

for t in keep:
    w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_ti_data_to_file()
for t in keep:
    w.write_tensor_data(t.data, tensor_endianess=r.endianess)
w.close()
print(f"wrote {outp}")
