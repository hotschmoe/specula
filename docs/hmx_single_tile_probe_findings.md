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

## w4a8 readout mechanism — RESOLVED (probe v3)

The integer readout's partial coverage is a property of the readout intrinsic +
its `Rt`, now characterized on-device (int all-ones, naive input):

| readout | positions written | row coverage |
|---|---|---|
| `Q6_mxmem_AR_after_uh_2x2`, Rt=0    | 256  | 8 rows (even 0..14) |
| `Q6_mxmem_AR_after_uh_2x2`, Rt=1023 | 512  | 16 rows (even 0..30) |
| `Q6_mxmem_AR_after_uh_2x1`, Rt=0    | **1024** | **all 32 rows** |
| `Q6_mxmem_AR_after_hf` (fp16 ref)   | 1024 | all 32 rows |

So **`uh_2x1` reads out the full 32×32 integer tile** (and `Rt` widens `uh_2x2`
coverage). Use `uh_2x1` (or paired `uh_2x2` calls) for the integer readout. The
remaining value error (uniform input still yields a 4-vs-240 split by row half)
is the **naive input tile layout** folding — not the readout. That isolates the
last unknown cleanly to the input layouts.

## w4a8 INTEGER DATA PATH — confirmed in-engine (probe v4)

The decisive correction (from QAIRT `tile_extract.h`): an **HMX tile is always
2 KB**, and a **uint8 tile is 8×8×32 = 2048 elements in 'flat' order** (not the
32×32 / 1024-byte / `Rt=1023` the first draft assumed; 16-bit fp16 is the
*different* 8h×4w×32 crouton = 1024 elem × 2 B). The earlier fold/scale/partial
coverage was entirely this tile-size mismatch.

Re-probed in the live backend (the probe dispatches through `op_matmul` on a
real HTP session — same path the model uses) with **full 2 KB tiles + `Rt=2047`
+ `uh_2x1` readout**:

| config | result |
|---|---|
| naive 32×32, `Rt=1023`, `uh_2x2` Rt=0 | 256/1024 written, folded |
| naive 32×32, `Rt=1023`, `uh_2x1`      | 1024 written but split 4 / 240 |
| **2 KB, `Rt=2047`, `uh_2x1`, all-ones** | **1024 written, UNIFORM 4.0** ✓ |
| **2 KB, `Rt=2047`, `uh_2x1`, ramp**     | clean, fully decodable (below) |

So the integer data path is: **uint8 activation in 8×8×32 flat (M = h·8+w
spatial, d = depth/K), int4 weight (`R4Weights8x4`), `Q6_activation_ub` +
`Q6_weight_ubit` with `Rt=2047`, `uh_2x1` readout → full clean 1024-coverage.**

Decoded mapping (ramp `act[M]=M+1`, recovered value = `4·(M+1)`):
- **readout row `r` → activation spatial `M = 4·(r//2)+1`** (rows pair up;
  value constant across all output columns), distinct M = {1,5,9,…,61}.
- A single activation+weight+readout pass covers a **strided subset** of the
  output — exactly like the proven fp16 `core_dot_chunk_fp16`, which composes
  the full result with nested `r`/`c`/`k` tile loops. The full w4a8 matmul must
  loop passes the same way.
- **A fixed ÷8 readout scale** (all-ones K=32 → acc 32 → reads 4); fold into
  the rescale (or it may be a `uh_2x1` vs `uh_2x2` property — pin by varying K).

This is the "exact data path to target," verified on silicon. It is NOT the
fp16 path's 32×32 geometry — the integer path is denser (8-bit 8×8×32 tiles).

## Scoped next steps (w4a8, in order)

1. **Build the full integer matmul on the confirmed geometry** (2 KB / `Rt=2047`
   / `uh_2x1`), looping passes like `core_dot_chunk_fp16`; pack int4 weights in
   `R4Weights8x4` and uint8 activations in 8×8×32 flat:
   `R4Weights8x4Layout` for the int4 weight tile and the uint8-activation
   crouton (`QNN/HTP/core/memory_layout.h` + `tile_extract.h`). Re-probe
   all-ones: success = uniform value across the full set the readout writes,
   with the correct magnitude (32, not 4).
2. ~~Resolve the `uh_2x2` partial readout~~ **DONE** — use `Q6_mxmem_AR_after_uh_2x1`
   for full 32-row coverage (see table above). Still to pin: the exact
   `(r,c)→offset` permutation, anchored against the fp16 Rosetta (slot 11).
3. **Pin the fixed scale** (the ÷8 / K-subset factor) once layouts are right.
4. Fold the **per-32-block Q4_0 fp16 scale** (the structural Unknown #3) via
   per-block partial readouts or the `mxmem2` column-bias.
5. Then flip `HTP_W4A16_VALIDATED`, validate one real Q4_0 tile vs CPU
   (`test-backend-ops -o MUL_MAT` on HTP, single small shape), then full
   `llama-bench` numerics (cos vs fp16 path) + PP vs the 175/2224 references.

Artifacts: `results/hmx_probe_raw*.bin`, `scripts/analyze_hmx_probe.py`,
probe + harness on branch `npu-int4a16-hmx`
(`-DGGML_HEXAGON_W4A16_PROBE=ON` → `tests/test-hmx-probe.exe`).
