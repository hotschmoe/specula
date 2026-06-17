#!/usr/bin/env python3
"""Generate qnn-net-run inputs + backend config to profile one graph of a
Qualcomm context binary, for per-op HTP timing.

Reads the qnn-context-binary-utility JSON dump, picks a target graph, writes
zero-valued raw input tensors (timing is data-independent), an input_list.txt,
and a backend-extensions config JSON pointing at the bundle's htp ext config.
Prints the qnn-net-run command to run.
"""
import json
import os
import sys
import numpy as np

DISS = r"C:\Users\hotschmoe\Documents\GitHub\specula\results\qualcomm_dissect"
OUT = r"C:\Users\hotschmoe\Documents\GitHub\specula\results\qualcomm_dissect\prof"
BUNDLE = r"C:\Users\hotschmoe\Documents\GitHub\specula\models\qualcomm-qwen3-4b-ref\qwen3_4b-genie-w4a16-qualcomm_snapdragon_x2_elite"

DT = {  # QNN dtype -> (numpy dtype, bytes)
    "INT_32": (np.int32, 4),
    "UFIXED_POINT_16": (np.uint16, 2),
    "UFIXED_POINT_8": (np.uint8, 1),
    "FLOAT_32": (np.float32, 4),
    "FLOAT_16": (np.float16, 2),
}


def tinfo(t):
    ti = t.get("info", t)
    return (ti.get("name"), (ti.get("dataType") or "").replace("QNN_DATATYPE_", ""),
            ti.get("dimensions"))


def main():
    part = sys.argv[1] if len(sys.argv) > 1 else "part2"
    graphsub = sys.argv[2] if len(sys.argv) > 2 else "prompt_ar128_cl512"
    d = json.load(open(os.path.join(DISS, part + ".json")))
    graphs = d["info"]["graphs"]
    g = next(x["info"] for x in graphs if graphsub in x["info"]["graphName"])
    gname = g["graphName"]
    os.makedirs(OUT, exist_ok=True)
    raw = os.path.join(OUT, "in")
    os.makedirs(raw, exist_ok=True)

    entries = []
    for t in g["graphInputs"]:
        name, dtype, dims = tinfo(t)
        npdt, nb = DT[dtype]
        n = 1
        for x in dims:
            n *= x
        arr = np.zeros(n, dtype=npdt)
        fn = os.path.join(raw, name.replace("/", "_").replace(".", "_") + ".raw")
        arr.tofile(fn)
        entries.append(f"{name}:={fn}")
    # input_list.txt — QNN format: one inference per line, space-separated name:=path
    il = os.path.join(OUT, "input_list.txt")
    with open(il, "w") as f:
        f.write(" ".join(entries) + "\n")

    # backend extensions config for qnn-net-run -> points at the bundle htp ext config
    ext = {
        "backend_extensions": {
            "shared_library_path": "QnnHtpNetRunExtensions.dll",
            "config_file_path": os.path.join(BUNDLE, "htp_backend_ext_config.json"),
        }
    }
    extf = os.path.join(OUT, "netrun_ext.json")
    json.dump(ext, open(extf, "w"), indent=1)

    print(f"graph        = {gname}  ({len(entries)} inputs)")
    print(f"input_list   = {il}")
    print(f"config_file  = {extf}")
    print(f"context_bin  = {os.path.join(BUNDLE, part.replace('part','qwen3_4b_part_') + '_of_4.bin')}")
    print(f"output_dir   = {os.path.join(OUT, 'out')}")


if __name__ == "__main__":
    main()
