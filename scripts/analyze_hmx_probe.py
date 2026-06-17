#!/usr/bin/env python3
"""Analyze the raw HMX single-tile probe dump (tests/test-hmx-probe.cpp).

The DSP probe (hmx_w4a16_probe) wrote NSLOT=8 raw 32x32 readout tiles (1024
floats each, in HMX memory order) to a host buffer; the host reader saved them
to a .bin of 8192 float32.  Each slot ran a KNOWN single tile so the true
acc[r][c] is known; by reading the raw memory-order tile back we recover the
HMX readout permutation and answer the existence questions.

Slot layout (must match hmx-matmul-ops.c):
  0 w4a16-mixed all-ones          expect acc[r][c]=32 everywhere
  1 w4a16-mixed act row r=(r+1)   acc[r][c]=32*(r+1)        -> ROW axis
  2 w4a16-mixed wt colpair p=p    acc[r][c]=32*floor(c/2)   -> COL axis (pair)
  3 w4a8-int    all-ones          expect 32 everywhere
  4 w4a8-int    act row r=(r+1)   acc[r][c]=32*(r+1)        -> ROW axis (uh_2x2)
  5 w4a8-int    wt colpair p=p    acc[r][c]=32*floor(c/2)   -> COL axis (pair)
  6 w4a8-int    nibble 0x21       acc[r][c]=32*{1 or 2}     -> nibble->col parity
  7 w4a16-mixed nibble 0x21       same, mixed path
"""
import sys
import numpy as np

TILE = 1024

SLOT_DESC = [
    "w4a16-mixed ubit all-ones (expect 32)",
    "w4a16-mixed ubit act row r=(r+1)  -> readout ROW axis",
    "w4a16-mixed ubit wt colpair p=p   -> readout COL axis",
    "w4a8-int    all-ones (expect 32)",
    "w4a8-int    act row r=(r+1)  -> int readout ROW axis",
    "w4a8-int    wt colpair p=p   -> int readout COL axis",
    "w4a8-int    nibble 0x21      -> nibble->col parity",
    "w4a16-mixed ubit nibble 0x21 -> nibble->col parity",
    "w4a16-mixed N    all-ones (Q6_weight_n; expect 32 if dequant)",
    "w4a16-mixed N    act row r=(r+1)",
    "w4a16-mixed sbit all-ones (Q6_weight_sbit; expect 32)",
    "fp16xfp16   PROVEN all-ones (Rosetta; acc=32)",
    "w4a8-int    uh_2x2 Rt=1023 (coverage vs slot3?)",
    "w4a8-int    uh_2x1 readout variant",
    "w4a8-int    2KB Rt=2047 all-ones uh_2x1 (uniform 32?)",
    "w4a8-int    2KB Rt=2047 spatial-ramp 8x8x32 uh_2x1",
    "w4a8-int    2KB depth-ramp (d+1) -> scale (528/66/10)",
    "w4a8-int    2KB single depth d=5 -> scale confirm",
    "w4a8-int    2KB weight colramp [n][k] -> COL->n map",
]
NSLOT = len(SLOT_DESC)


def load(path):
    a = np.fromfile(path, dtype=np.float32)
    assert a.size >= NSLOT * TILE, f"expected >={NSLOT*TILE} floats, got {a.size}"
    return a[: NSLOT * TILE].reshape(NSLOT, TILE)


def dump_slot(name, tile):
    finite = np.isfinite(tile)
    n_nan = int((~finite).sum())
    vals = tile[finite]
    nz = vals[vals != 0.0]
    n_written = int(nz.size)
    distinct = sorted(set(np.round(nz, 3).tolist()))
    # is every nonzero value a near-multiple of 32?
    mult32 = bool(nz.size) and bool(np.all(np.abs(nz / 32.0 - np.round(nz / 32.0)) < 0.05))
    print(f"  [{name}]")
    print(f"     written(nonzero)={n_written}/{TILE}  nan/inf={n_nan}  "
          f"all_nonzero_mult_of_32={mult32}")
    show = distinct[:24]
    print(f"     distinct nonzero (<=24): {show}")
    return n_written, distinct, mult32


def existence(name, tile, expect=32.0):
    n_written, distinct, mult32 = dump_slot(name, tile)
    near = int(np.sum(np.abs(tile - expect) < 0.5))
    # MAC "works" if the written values are uniform ~= expect (uniform inputs)
    ok = near >= 1 and all(abs(d - expect) < 0.5 for d in distinct)
    print(f"     ~={expect:.0f} count={near}  -> existence {'PASS' if ok else 'fail'}")
    return ok


def resolve_readout(row_tile, col_tile, nib_tile, label):
    """Recover offset -> (r,c) from the row/col/nibble probe slots."""
    print(f"\n=== readout resolution: {label} ===")
    # row index from 32*(r+1); col-pair from 32*floor(c/2); parity from nibble.
    r_of = np.full(TILE, -1, dtype=int)
    cp_of = np.full(TILE, -1, dtype=int)
    ok = True
    for o in range(TILE):
        rv = row_tile[o] / 32.0
        cv = col_tile[o] / 32.0
        if not (np.isfinite(rv) and np.isfinite(cv)):
            ok = False
            continue
        r = int(round(rv)) - 1
        cp = int(round(cv))
        if 0 <= r < 32:
            r_of[o] = r
        if 0 <= cp < 16:
            cp_of[o] = cp
    nrows = len(set(r_of[r_of >= 0]))
    ncps = len(set(cp_of[cp_of >= 0]))
    print(f"  distinct rows recovered: {nrows}/32   distinct col-pairs: {ncps}/16")
    # nibble parity: nib_tile holds 32*(1 or 2). within a col-pair the two cols
    # should differ if the two nibbles map to even/odd col.
    nib_vals = sorted(set(int(round(v / 32.0)) for v in nib_tile if np.isfinite(v)))
    print(f"  nibble-probe distinct weight-values (expect {{1,2}}): {nib_vals}")

    # Build the (r, col_pair) -> offset table and print as a compact grid so the
    # readout permutation is human-readable.  Full per-column parity needs the
    # nibble slot, but (r, col-pair) already pins the formula structure.
    if nrows >= 30 and ncps >= 14:
        print("  offset(r, col_pair) grid [r=0..7 rows shown, cp=0..15]:")
        inv = {}
        for o in range(TILE):
            if r_of[o] >= 0 and cp_of[o] >= 0:
                inv[(r_of[o], cp_of[o])] = o
        for r in range(8):
            row = " ".join(f"{inv.get((r, cp), -1):4d}" for cp in range(16))
            print(f"    r={r:2d}: {row}")
        # Try to fit candidate closed forms for offset(r, c) using col = 2*cp
        # (even column).  Compare against the kernel's current guess + agent's.
        print("  candidate-formula check (even col = 2*cp):")
        check_formulas(inv)
    else:
        print("  (insufficient distinct rows/col-pairs to resolve cleanly)")
    return ok


def check_formulas(inv):
    # inv maps (r, cp) -> memory offset, with even column c = 2*cp.
    # Kernel current guess hmx_uh_2x2_elem_off(row, col):
    def kern(row, col):
        d = col & 31
        h = row & 1
        w = (row >> 1) & 1
        hpair = row >> 2
        return (w & 1) + 2 * (h & 1) + 4 * d + 512 * hpair

    # Agent's R4Crouton2x2 reading: offset=((h//2)*2+(w//2))*128+((h%2)*2+(w%2))*32+d
    # with (h,w,d) = (row, col, depth?) -- ambiguous; try row->h, col->w mapping
    # is nonsensical for 32-wide col, so try col=d variant too.
    def agentA(row, col):  # h=row, w=col? (col up to 31 breaks 2x2) -- skip
        return None

    names = {"kernel_guess": kern}
    for nm, fn in names.items():
        hits = 0
        tot = 0
        for (r, cp), o in inv.items():
            col = 2 * cp
            pred = fn(r, col)
            if pred is None:
                continue
            tot += 1
            if pred == o:
                hits += 1
        if tot:
            print(f"    {nm}: {hits}/{tot} offsets match")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/hmx_probe_raw.bin"
    t = load(path)
    print(f"loaded {path}\n")
    print("=== ALL SLOTS (written = nonzero positions; rest memset to 0) ===")
    for s in range(NSLOT):
        dump_slot(f"slot{s} {SLOT_DESC[s]}", t[s])
    print("\n=== EXISTENCE TESTS (permutation-independent) ===")
    mixed_ok = existence("slot0 w4a16-mixed all-ones", t[0])
    int_ok = existence("slot3 w4a8-int   all-ones", t[3])

    print("\n=== HEADLINE ===")
    print(f"  native w4a16 (int4 x fp16) HMX primitive EXISTS: "
          f"{'YES' if mixed_ok else 'NO / not cleanly'}")
    print(f"  w4a8 (int4 x uint8) integer HMX works:           "
          f"{'YES' if int_ok else 'NO / not cleanly'}")

    if int_ok:
        resolve_readout(t[4], t[5], t[6], "w4a8 integer (uh_2x2)")
    if mixed_ok:
        resolve_readout(t[1], t[2], t[7], "w4a16 mixed (hf readout)")


if __name__ == "__main__":
    main()
