# On-device HMX single-tile probe — findings (2026-06-16)

Empirical reverse-engineering of the undocumented Hexagon **HMX matrix ISA**
on the Snapdragon X2 Elite (arch v81), to settle two questions:

1. Does a **native w4a16** primitive exist — int4-weight × fp16-activation with
   the weight decompressed to its integer value in hardware (Qualcomm-blessed,
   accuracy-free)?
2. Is the **w4a8** integer path (int4-weight × uint8-activation → int32) a
   working MAC, and what are its tile/readout layouts?

## Method — the probe harness

The HMX matrix ISA (operand pairing, readout layout, `Rt` semantics) is
undocumented, so this is empirical. Rather than guess layouts and risk DSP
crashes, the probe (`ggml/src/ggml-hexagon/htp/hmx-matmul-ops.c::hmx_w4a16_probe`,
branch `npu-int4a16-hmx`):

- builds **KNOWN** 32×32 tiles directly in VTCM (so results don't depend on the
  still-unvalidated Q4_0 packing — uniform `0x11` weights = every int4 nibble
  is 1, nibble-order-independent);
- runs ONE HMX intrinsic sequence per subtest on a single output tile;
- `memset`s the readout VTCM to 0 first, so **written positions are nonzero and
  unwritten positions read back as 0** (distinguishable: all legit values ≥ 4);
- copies the **RAW** readout tile (HMX memory order, no offset interpretation)
  to a host-visible `dst`.

All layout brute-forcing happens **offline on the host**
(`scripts/analyze_hmx_probe.py`) — one on-device run resolves many unknowns
with no recompile per hypothesis. Triggered via an `op_matmul` hijack under
`-DGGML_HEXAGON_W4A16_PROBE=ON`; the host reader is
`tests/test-hmx-probe.cpp` (runs one big Q4_0 `mul_mat` on HTP0, dumps raw).

Mechanism note: **DSP-side FARF is NOT visible on stdout** on this Windows
host (goes to OutputDebugString) — confirmed empirically — hence the
dump-raw-to-dst design instead of FARF reporting.

## Result 1 — w4a8 integer MAC: **WORKS** ✅

int4-weight (`Q6_weight_ubit`) × uint8-activation (`Q6_activation_ub`) →
int32 accumulator (`Q6_mxclracc`) → `Q6_mxmem_AR_after_uh_2x2` (uint16) readout
**computes a correct matrix product**. Evidence (single 32-wide K-tile, naive
`off=r*32+c` view of the raw readout):

| subtest | true acc[r][c] | observed (raw) |
|---|---|---|
| all-ones | 32 (uniform) | **4.0 uniform** — uniform in → uniform out ✓ |
| act row ramp r→(r+1) | 32·(r+1) | 8,24,40,56,… — **monotone in row** ✓ |
| wt colpair p→p | 32·⌊c/2⌋ | 8,12 alternating in col-pairs — **col structure** ✓ |

So the integer HMX MAC is real and structure-preserving. **Open layout
details** (the remaining w4a8 work):

- **Only 256 of 1024 positions are written** by a single `uh_2x2` call (in the
  naive view: even rows 0..14, all cols). A full 32×32 readout needs either
  multiple `uh_2x2` calls or the `:retain`/`2x1` variants — `uh_2x2` emits a
  2×2-folded sub-block, not the whole tile.
- **A fixed scale** (~÷8, or equivalently only ~4 of the 32 K-lanes contract)
  and **row/col folding** appear — strongly implying the **input tile layouts**
  (uint8 activation, int4 weight) are NOT the naive row-major fill the probe
  used; they need the authoritative QAIRT crouton / `R4Weights8x4Layout`
  packing. The folding is that mismatch showing through.

Next step for w4a8: feed inputs in the real `R4Weights8x4Layout` (weight) +
uint8-activation-crouton layouts and re-probe; the readout decode then drops
out cleanly against the `fp16` Rosetta tile (slot 11, known-correct layout).

## Result 2 — native w4a16: **DOES NOT EXIST** ❌ (definitive)

**Harness validation first:** the proven `fp16 × fp16` all-ones tile (slot 11)
returns exactly **32.0** through the identical probe path (clear → activation →
weight → `Q6_mxmem_AR_after_hf` → dst dump). So the MAC, readout, and dump are
all correct — any anomaly in the mixed slots is real hardware behaviour, not a
probe artefact.

Against that anchor, **none of the three int4 weight loaders** dequantize the
int4 weight to its integer value when paired with an fp16 activation + fp16
accumulator (all-ones, true acc = 32):

| weight loader | observed | expected |
|---|---|---|
| `Q6_weight_ubit` (unsigned int4) | 0.0099 | 32 |
| `Q6_weight_n` (normalized)       | 0.0049 | 32 |
| `Q6_weight_sbit` (signed int4)   | 0.0099 | 32 |
| `Q6_weight_hf` (fp16, Rosetta)   | **32.0** ✓ | 32 |

Every int4 loader is **~3000–6500× too small**, and `ubit` is **non-linear in
the integer weight value** (the `0x21` nibble probe → uniform 0.16, not two
distinct values for weights 1 vs 2). The int4 weight is reinterpreted as a tiny
fixed-point / subnormal quantity, not its dequantized integer. No fixed scale
recovers a usable, weight-linear matmul.

**Conclusion: there is no native int4-weight × fp16-activation HMX primitive on
v81.** The earlier "plausible native w4a16" hypothesis (from the perf signature
+ intrinsic-name coexistence, REOPENED in the plan) is now **empirically
disproven on silicon** with a validated harness. The only thing the HMX weight
unit will do with an int4 load in an fp16 pass is a tiny non-integer
reinterpretation. (Lone open thread, non-viable in practice: `Q6_weight_n` is
linear in the *activation* but its weight-linearity was not separately probed —
moot given the ~6500× magnitude error with no clean scale.)

## Verdict / direction

- **Native w4a16 (int4×fp16) is disproven** — do not pursue an "accuracy-free
  native primitive." Close the REOPENED question.
- **w4a8 (int4 × uint8 → int32) is the viable integer-HMX route** — the MAC is
  confirmed working. Remaining work: use the authoritative QAIRT tile layouts
  (`R4Weights8x4Layout` weight pack + uint8-activation crouton) instead of the
  naive row-major fill, assemble the full 32×32 via the partial `uh_2x2`
  readout (single call writes 256/1024), resolve the fixed scale, then add the
  per-32-block Q4_0 fp16 scale. The fp16 Rosetta tile (slot 11) anchors the
  readout decode.

## w4a8 fold — precise characterization (from the naive-layout probe)

A single `Q6_mxmem_AR_after_uh_2x2` call on the integer accumulator (naive
`off=r*32+c` view of the raw 1024-float dump):

- **Writes 256 of 1024 positions**: naive rows {0,2,4,…,14} only (even, <16),
  all 32 naive cols. Odd rows and rows ≥16 stay 0.
- **Row map + scale** (slot4 ramp, value = 8·(naive_r+1)) is consistent with
  `value = acc/8` where `acc = 32·(true_r+1)` and `true_r = 2·naive_r+1`
  (i.e. one call captures only true rows ≡ 1 (mod 4) — 8 of 32). Equivalent
  reading: only ~4 of 32 K-lanes contract (acc≈4 for all-ones) → same numbers.
- **Col fold** (slot5 colpair): naive cols collapse to a `[8,8,12,12]`
  repeat — only ~2 distinct true col-pairs captured.

Every one of these (1/4 rows, fixed scale, col fold, K-subset) is a symptom of
the **naive row-major tile fill not matching HMX's required layouts**. The fix
is not more probing of the naive fill — it is to feed inputs in the documented
QAIRT layouts.

## Scoped next steps (w4a8, in order)

1. **Implement the authoritative input layouts** in the probe fill functions:
   `R4Weights8x4Layout` for the int4 weight tile and the uint8-activation
   crouton (`QNN/HTP/core/memory_layout.h` + `tile_extract.h`). Re-probe
   all-ones: success = uniform value across the full set the readout writes,
   with the correct magnitude (32, not 4).
2. **Resolve the `uh_2x2` partial readout**: determine the call count / variant
   (`uh_2x2` vs `uh_2x1`, `:retain`) needed to emit all 1024 outputs, and the
   `(r,c)→offset` permutation. Anchor with the fp16 Rosetta tile (slot 11,
   known crouton).
3. **Pin the fixed scale** (the ÷8 / K-subset factor) once layouts are right.
4. Fold the **per-32-block Q4_0 fp16 scale** (the structural Unknown #3) via
   per-block partial readouts or the `mxmem2` column-bias.
5. Then flip `HTP_W4A16_VALIDATED`, validate one real Q4_0 tile vs CPU
   (`test-backend-ops -o MUL_MAT` on HTP, single small shape), then full
   `llama-bench` numerics (cos vs fp16 path) + PP vs the 175/2224 references.

Artifacts: `results/hmx_probe_raw*.bin`, `scripts/analyze_hmx_probe.py`,
probe + harness on branch `npu-int4a16-hmx`
(`-DGGML_HEXAGON_W4A16_PROBE=ON` → `tests/test-hmx-probe.exe`).
