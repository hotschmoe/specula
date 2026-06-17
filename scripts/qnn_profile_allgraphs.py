#!/usr/bin/env python3
"""Generate per-graph zero inputs + input_lists for ALL graphs in a QNN context
binary, and print the comma-joined --input_list for qnn-net-run."""
import json
import os
import sys
import numpy as np

DISS = r"C:\Users\hotschmoe\Documents\GitHub\specula\results\qualcomm_dissect"
OUT = r"C:\Users\hotschmoe\Documents\GitHub\specula\results\qualcomm_dissect\prof"
DT = {"INT_32": (np.int32, 4), "UFIXED_POINT_16": (np.uint16, 2),
      "UFIXED_POINT_8": (np.uint8, 1), "FLOAT_32": (np.float32, 4), "FLOAT_16": (np.float16, 2)}


def tinfo(t):
    ti = t.get("info", t)
    return ti.get("name"), (ti.get("dataType") or "").replace("QNN_DATATYPE_", ""), ti.get("dimensions")


def main():
    part = sys.argv[1] if len(sys.argv) > 1 else "part2"
    d = json.load(open(os.path.join(DISS, part + ".json")))
    graphs = d["info"]["graphs"]
    ils = []
    for gi, gw in enumerate(graphs):
        g = gw["info"]
        gname = g["graphName"]
        gd = os.path.join(OUT, "all", gname)
        os.makedirs(gd, exist_ok=True)
        entries = []
        for t in g["graphInputs"]:
            name, dtype, dims = tinfo(t)
            npdt, _ = DT[dtype]
            n = 1
            for x in dims:
                n *= x
            fn = os.path.join(gd, name.replace("/", "_").replace(".", "_") + ".raw")
            np.zeros(n, dtype=npdt).tofile(fn)
            entries.append(f"{name}:={fn}")
        ilf = os.path.join(gd, "input_list.txt")
        open(ilf, "w").write(" ".join(entries) + "\n")
        ils.append(ilf)
    print(",".join(ils))


if __name__ == "__main__":
    main()
