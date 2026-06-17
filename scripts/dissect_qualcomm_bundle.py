#!/usr/bin/env python3
"""Phase 1 static dissection of the Qualcomm Qwen3-4B w4a16 bundle.

Parses the qnn-context-binary-utility JSON dumps (one per .bin part) and emits a
structured summary: graphs (ctx tiers x ar modes), shared-weight / op-data / IO
sizes, and the full IO tensor contract (dtype + quant + dims) for the AR128
prefill graph of each part.  Used to build the deep-dive doc.
"""
import json
import sys
import glob
import os

DISS = r"C:\Users\hotschmoe\Documents\GitHub\specula\results\qualcomm_dissect"


def tinfo(t):
    ti = t.get("info", t)
    q = ti.get("quantizeParams", {}) or {}
    enc = q.get("quantizationEncoding", "")
    sc = None
    off = None
    sod = q.get("scaleOffsetEncoding") or q.get("scaleOffset") or {}
    if isinstance(sod, dict):
        sc, off = sod.get("scale"), sod.get("offset")
    return dict(name=ti.get("name"), dtype=(ti.get("dataType") or "").replace("QNN_DATATYPE_", ""),
                dims=ti.get("dimensions"), enc=enc.replace("QNN_QUANTIZATION_ENCODING_", ""),
                scale=sc, offset=off, ttype=ti.get("type"))


def main():
    parts = sorted(glob.glob(os.path.join(DISS, "part*.json")))
    for pf in parts:
        d = json.load(open(pf))
        info = d["info"]
        name = os.path.basename(pf)
        graphs = info["graphs"]
        print("=" * 78)
        print(f"{name}  blobSize={info['contextBlobSize']/1e6:.1f}MB  numGraphs={info['numGraphs']}  "
              f"buildId={info['buildId']}  soc={info['socModel']}")
        # blob info from graph 0 (shared across graphs)
        g0 = graphs[0]["info"]
        bi = g0.get("graphBlobInfoV2", {})
        print(f"  sharedWeights={bi.get('sharedWeightsSize',0)/1e6:.1f}MB  opData={bi.get('opDataSize',0)/1e6:.2f}MB "
              f"ioTensor={bi.get('ioTensorSize',0)/1e6:.2f}MB  nativeKChan={bi.get('nativeKChannelSize')} "
              f"nativeVChan={bi.get('nativeVChannelSize')}")
        print("  graphs:", ", ".join(g["info"]["graphName"] for g in graphs))
        # pick the AR128 cl512 prefill graph
        pre = next((g["info"] for g in graphs if "ar128_cl512" in g["info"]["graphName"]), g0)
        ins = [tinfo(t) for t in pre["graphInputs"]]
        outs = [tinfo(t) for t in pre["graphOutputs"]]
        print(f"  --- {pre['graphName']}: {len(ins)} inputs, {len(outs)} outputs ---")
        # group inputs by dtype
        from collections import Counter
        din = Counter((i["dtype"], i["enc"]) for i in ins)
        dout = Counter((o["dtype"], o["enc"]) for o in outs)
        print("    input dtypes:", dict(din))
        print("    output dtypes:", dict(dout))
        # show the non-KV inputs (the interesting ones)
        for i in ins:
            if "past_" not in (i["name"] or ""):
                print(f"      IN  {i['name']:24s} {i['dtype']:18s} {i['enc']:14s} dims={i['dims']} scale={i['scale']}")
        for o in outs:
            if "present" not in (o["name"] or "") and "past" not in (o["name"] or ""):
                print(f"      OUT {o['name']:24s} {o['dtype']:18s} {o['enc']:14s} dims={o['dims']} scale={o['scale']}")
        # one KV example
        kv = next((i for i in ins if "past_key" in (i["name"] or "")), None)
        if kv:
            print(f"      KV  {kv['name']:24s} {kv['dtype']:18s} {kv['enc']:14s} dims={kv['dims']} scale={kv['scale']}")


if __name__ == "__main__":
    main()
