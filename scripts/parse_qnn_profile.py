#!/usr/bin/env python3
"""Parse qnn-profile-viewer CSV -> per-op-type cycle breakdown per graph.

Rows of interest:
  ts,EXECUTE,<cycles>,CYCLES,BACKEND,SUB-EVENT,<opname>:OpId_N (cycles)
Graph boundaries:
  ts,EXECUTE,<us>,US,NETRUN,ROOT,Graph N: <graphName>
"""
import re
import sys
from collections import defaultdict

CSV = sys.argv[1] if len(sys.argv) > 1 else \
    r"C:\Users\hotschmoe\Documents\GitHub\specula\results\qualcomm_dissect\prof\out3\prof.csv"
WANT = sys.argv[2] if len(sys.argv) > 2 else "prompt_ar128_cl512"


def optype(name):
    # name like "/model/.../self_attn/MatMul:OpId_123 (cycles)" -> "MatMul"
    base = name.split(":")[0]
    leaf = base.rsplit("/", 1)[-1]
    # strip trailing _NN index
    leaf = re.sub(r"_\d+$", "", leaf)
    return leaf or base


def main():
    cur = None
    total = defaultdict(lambda: [0, 0])  # graph -> not used
    bytype = defaultdict(lambda: [0, 0])  # optype -> [cycles, count]
    accel_total = 0
    graph_us = {}
    for line in open(CSV, encoding="utf-8", errors="ignore"):
        parts = line.rstrip("\n").split(",")
        if len(parts) < 7:
            continue
        ev_id = parts[6]
        if parts[1] == "EXECUTE" and parts[5] == "ROOT" and ev_id.startswith("Graph "):
            cur = ev_id.split(": ", 1)[-1]
            graph_us[cur] = int(parts[2])
            continue
        if cur and WANT in (cur or "") and parts[3] == "CYCLES" and parts[5] == "SUB-EVENT":
            cyc = int(parts[2])
            ot = optype(ev_id)
            bytype[ot][0] += cyc
            bytype[ot][1] += 1
        if cur and WANT in (cur or "") and "Accelerator (execute) time (cycles)" in ev_id:
            accel_total = int(parts[2])

    print(f"graph match: {WANT}")
    print(f"accelerator execute total: {accel_total:,} cycles")
    tot = sum(v[0] for v in bytype.values())
    print(f"sum of per-op cycles: {tot:,}\n")
    print(f"{'op type':28s} {'cycles':>14s} {'%':>6s} {'count':>6s} {'cyc/op':>10s}")
    for ot, (cyc, cnt) in sorted(bytype.items(), key=lambda kv: -kv[1][0]):
        pct = 100.0 * cyc / tot if tot else 0
        print(f"{ot:28s} {cyc:>14,} {pct:>5.1f}% {cnt:>6d} {cyc//max(cnt,1):>10,}")


if __name__ == "__main__":
    main()
