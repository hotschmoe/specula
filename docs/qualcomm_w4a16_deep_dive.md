# Qualcomm Qwen3-4B w4a16 bundle — deep dive (how it hits 2200+ pp)

Goal: dissect, end to end, the blessed Qualcomm Qwen3-4B w4a16 Genie bundle that
reaches **PP ~2224 t/s** on the X2 Elite NPU — every step of data movement and
compute — so we understand exactly where the performance comes from and what an
open engine (llama.cpp) must replicate. Living document; built in phases.

Engine landscape (which thing hits 2200+):
- **ORT-QNN** (our `npu_engine/`, `bench_qwen3_4b_ortqnn.py`) — measured **PP 2229 / TG 27.8**.
- **Genie** (`genie-t2t-run`) — ~1725 pp; DSP transport flaky on this box.
- **qnn-net-run** (QAIRT standalone) — runs the same `.bin` contexts directly,
  bypassing ORT; used here for clean per-op profiling.
All three execute the **same pre-compiled HTP context binaries** (`.bin`); the
performance is a property of those binaries + the QnnHtp runtime config, not the
host wrapper. Tools (all native ARM64 at `C:\Qualcomm\AIStack\QAIRT\2.45.40.260406\bin\aarch64-windows-msvc\`):
`qnn-context-binary-utility` (graph dump), `qnn-net-run` (run+profile),
`qnn-profile-viewer`, `hexagon-llvm-objdump`/`-dis` (HTP disasm), `genie-t2t-run`.

---

## Phase 1 — static structure (the "what")

### Model (config.json)
Qwen3-4B: **36 layers**, hidden **2560**, intermediate **9728**, **32 attn heads**,
**8 KV heads** (GQA), head_dim **128**, vocab **151936**, rope_theta **1e6**,
`tie_word_embeddings=true` (lm_head shares the embedding table). RMSNorm eps 1e-6,
SiLU MLP. Trained bf16.

### Bundle topology (4 context binaries, `qnn-context-binary-utility`)
Built with **QAIRT v2.42** (buildId `v2.42.0.251225...`), socModel **88** (X2 Elite),
contextBlobVersion 3.3.4. Each `.bin` holds **10 graphs**: 5 context tiers
(512/1024/2048/3072/4096) × {`prompt_ar128_*` prefill, `token_ar1_*` decode}.

| part | size | sharedWeights | opData | ioTensor | contents |
|---|---|---|---|---|---|
| 1 | 778 MB | **777.9 MB** | 0 | 0.66 MB | embedding (vocab×hidden uint16 = 151936·2560·2 B = 778 MB) |
| 2 | 669 MB | **615.0 MB** | 7.27 MB | 14.06 MB | decoder layers 0–11 (w4) |
| 3 | 669 MB | **615.0 MB** | 7.27 MB | 14.06 MB | decoder layers 12–23 (w4) |
| 4 | 1070 MB | **1006.4 MB** | 9.94 MB | 52.3 MB | decoder layers 24–35 (w4) + lm_head (uint16, tied) |

- **Weights are w4** in the decoder: 615 MB / 12 layers = **51.25 MB/layer** =
  ~102.5M nibble-params/layer × 0.5 B — matches a 4B/36 dense layer at 4-bit.
- **Embedding + lm_head are uint16** (NOT 4-bit): 778 MB = vocab·hidden·2 B. They
  are a gather / final projection where precision matters and 4-bit gives little
  size win relative to accuracy cost.
- `nativeKChannelSize=256`, `nativeVChannelSize=64` (HMX channel tiling hints).

### The data types — "w4a16" decoded
- **Weights**: 4-bit (uint4, scale-offset), decoder only.
- **Activations** (hidden states flowing between parts): **`UFIXED_POINT_16`
  (uint16 fixed-point, scale+offset)** — *not fp16*. e.g. embed out scale 7.2e-6
  offset -30800; layer-11 out scale 0.2177. So the "a16" is **quantized uint16**,
  and the matmul is **int4-weight × uint16-activation** (how that maps to HMX,
  which has only ub/hf/f8 activation loaders, is the Phase-2 question).
- **KV cache**: **`UFIXED_POINT_8` (uint8, scale-offset)**, per-layer scale.
  `past_key_*` [8,1,128,384], `past_value_*` [8,1,384,128] (8 KV heads, head_dim
  128, 384 = past positions for cl512/ar128). present_* outputs are uint8 too.
  → KV is **¼ the bandwidth of fp32, ½ of fp16** — a major decode-bandwidth win.
- **attention_mask**: uint16 [1,1,128,512] (scale 0.00153, offset -65535 → additive
  −∞-style mask). **position_ids_cos/sin**: uint16 [1,1,128,64] (rotary **hoisted
  out** of the graph as inputs — matches our prior finding; avoids the inline
  rotary MatMul that fails QNN op-validation).
- Cross-part seam: hidden state passed part→part as uint16 [1,128,2560].

### Runtime config (genie_config.json + htp_backend_ext_config.json) — the perf knobs
- `dsp_arch: v81`, **`perf_profile: burst`** (max DCVS voltage/clock corners),
  `rpc_control_latency: 100`.
- **`mem_type: shared_buffer`** — ION/shared memory; CPU and DSP share IO buffers
  → **zero-copy** activation/KV IO (no DDR→DDR memcpy across the FastRPC boundary).
- **`weight_sharing_enabled: true`** — the 615 MB weight blob is shared across all
  10 graphs in a part (AR1 + AR128 + all 5 ctx tiers reference ONE copy). No
  per-graph weight duplication → fits the HTP budget and loads once.
- **`use-mmap: true`** — weights memory-mapped (fast load, page-cache shared).
- **`poll: true`** — host busy-polls for DSP completion instead of interrupt
  wakeup → lower per-step latency (matters for AR1 decode).
- `spill-fill-bufsize: 0` (no spill-fill needed — the working set fits VTCM budget).
- `n-threads: 3`, `cpu-mask: 0xe0` (host threads pinned to the 3 prime cores).

### Phase-1 takeaways (perf levers visible already)
1. **AR128 batched prefill** — the `prompt_ar128` graphs process 128 tokens/call,
   amortizing weight load over 128 rows (the prefill throughput multiplier).
2. **uint8 KV + zero-copy shared_buffer + mmap** — minimal data-movement overhead.
3. **Weight sharing across graphs** — one resident weight set serves prefill+decode
   and every ctx tier.
4. **w4 weights** — ¼ the weight DDR→VTCM bandwidth of fp16 (prefill is
   bandwidth-bound). This is the lever llama.cpp's fp16-dequant path gives up.
5. **Burst + poll** — clocks pinned high, no wakeup latency.

Open for Phase 2/3: the exact int4×uint16 matmul instructions (dequant? split
uint16? HMX mode), and the measured per-op time/bandwidth split.

---

## Phase 3 — dynamic per-op profiling (the "how fast")

**How it was measured.** Standalone `qnn-net-run` on `qwen3_4b_part_2_of_4.bin`
(layers 0–11) with `--profiling_level detailed --perf_profile burst`, decoded by
`qnn-profile-viewer`. Two unlocks were needed:
1. **DSP transport** — QAIRT's own `QnnHtp.dll` hits the `0x80000406` signed-PD
   wall (same as Genie/llama.cpp). Fix: point `--backend` *and*
   `ADSP_LIBRARY_PATH` at **ORT's bundled signed skel** (`.venv-ort21/.../onnxruntime_qnn/`,
   which ships `libQnnHtpV81Skel.so` + `libqnnhtpv81.cat`). Then qnn-net-run runs
   on-device exactly like ORT-QNN. (ORT's own profiling is unusable here — *any*
   profiling option, even `enable_profiling`, makes ORT 2.1.0 drop the QNN EP to
   CPU. qnn-net-run is the per-op path.)
2. **Inputs** — `--use_native_input_files` (else qnn-net-run treats raw files as
   float32 and the tensor sizes mismatch). Zero inputs (timing is data-independent).

Scripts: `scripts/qnn_profile_allgraphs.py` (gen inputs), `scripts/parse_qnn_profile.py`
(per-op-type breakdown). Caveat: detailed profiling adds per-op overhead, so
*absolute* times are inflated; the **relative** per-op split and the
**prefill-vs-decode ratio** are the signal.

### INIT (one-time, per part)
- **RPC load binary 730 ms / QNN-accel load 700 ms** — loading the 615 MB w4
  weight blob into HTP-accessible memory (mmap + shared_buffer keeps this a
  one-time cost, shared across all 10 graphs).
- VTCM acquire 3.2 ms; **8 HVX threads** used per graph.

### Per-op cycle breakdown — AR128 prefill (`prompt_ar128_cl512_2_of_4`, 12 layers × 128 tok)
Accelerator execute total **100.7M cycles**.

| op type | cycles | % | count | cyc/op | what it is |
|---|---|---|---|---|---|
| **MatMul** | 24.2M | **24.0%** | 792 | 30.5k | attention Q·Kᵀ + softmax·V (activation×activation, per head) |
| **Mul** | 22.7M | **22.5%** | 1945 | 11.7k | RoPE, SiLU gate, dequant/requant scaling |
| **Conv** | 15.1M | **15.0%** | 600 | 25.2k | **the w4a16 weight projections** (q/k/v/o/gate/up/down as 1×1 Conv) |
| Softmax | 9.7M | 9.6% | 384 | 25.2k | attention softmax (32 heads × 12 layers) |
| Add | 6.4M | 6.4% | 888 | | residuals / bias |
| RMSNorm | ~5.4M | ~5.4% | 24 | ~225k | input + post-attn layernorms (one big op each) |
| Slice/Sub/Transpose/Concat/Div | ~9M | ~9% | | | KV assembly, RoPE split, reshapes |

**Key insight:** in batched prefill the **w4 weight matmul (Conv) is only ~15%**.
Attention (MatMul 24% + Softmax 10% = 34%) and elementwise (Mul 22%) dominate —
because at AR128 the weight load/compute is amortised over 128 tokens, while the
O(tok²) attention and O(tok) elementwise scale up. So Qualcomm's prefill speed is
**not** a magic matmul; it's that the w4 projection is already cheap+amortised and
the rest is well-tiled.

### Per-op cycle breakdown — AR1 decode (`token_ar1_cl512_2_of_4`, 12 layers × 1 tok)
Accelerator execute total **36.4M cycles**.

| op type | cycles | % | count | cyc/op |
|---|---|---|---|---|
| MatMul | 14.2M | 39.2% | 792 | 18.0k |
| Conv (w4 weights) | 6.5M | 17.8% | 600 | 10.8k |
| Mul | 3.9M | 10.7% | 1945 | 2.0k |
| Slice | 3.6M | 9.8% | 1152 | 3.1k |
| Add | 2.7M | 7.5% | 888 | 3.1k |

### The AR128 win, quantified
- Prefill (128 tok) **100.7M cyc** vs decode (1 tok) **36.4M cyc** → processing
  **128 tokens costs only 2.8× one token** ⇒ **~46× more cycle-efficient per token.**
- Same op *count* in both (792 MatMul, 600 Conv, 1945 Mul …) — but at AR1 each op
  does 1 token of work against a **fixed ~10–18k-cycle/op floor** (kernel launch +
  VTCM setup + weight DMA). At AR128 that floor is amortised over 128 tokens.
- ⇒ **The 2224-pp prefill throughput is fundamentally per-op-overhead + weight-DMA
  amortisation over the 128-wide batch**, on top of an HMX-efficient w4 Conv. AR1
  decode (~26 t/s) is per-op-overhead-bound, which is why decode is slow regardless
  of quant.

### Clean (non-profiled) timing + bandwidth — part2 (12 layers), burst, qnn-net-run
Steady-state HTP `Accelerator (execute)` time, median of 100 runs (no per-op
profiling overhead):

| graph | cl512 | cl1024 | cl2048 | cl4096 |
|---|---|---|---|---|
| `prompt_ar128` (128 tok) | 11.75 ms | 12.50 ms | 15.88 ms | 21.10 ms |
| `token_ar1` (1 tok) | 6.97 ms | 7.12 ms | 7.95 ms | 10.08 ms |

- prefill **91.8 µs/token** (12 layers, cl512) vs decode **6968 µs/token** →
  **decode is ~76× slower per token** (clean confirmation of the AR128 win; even
  larger than the profiled 46× because profiling inflates prefill more). Times
  grow with ctx tier (attention is O(ctx)).
- Extrapolated decoder (36 layers ≈ 3× part2): prefill ~3630 t/s, decode ~48 t/s
  — consistent with the measured end-to-end **2229 pp / 27.8 tg** once embed +
  lm_head + host-seam overhead are added.
- **Bandwidth (weights resident, 615 MB w4 per part):** decode reads the part's
  weights in 6.97 ms = **88 GB/s** (≈39% of the 228 GB/s LPDDR5X) for ONE token →
  decode is heavily weight-bandwidth-bound. Prefill spreads the same 615 MB over
  128 tokens = **0.41 GB/s/token effective** → prefill is compute-bound, not
  bandwidth-bound. This is *the* quantitative reason w4 + AR128 wins prefill and
  why decode stays ~27 t/s regardless.

## Phase 2 — HTP instruction-level dissection (the "how") — CRACKED

The `.bin` contains no custom matmul code — QNN HTP references **pre-compiled op
kernels in `libQnnHtpV81Skel.so`** (11 MB; src tree `QAISW/FirstParty/QNN/HTP/
.../ops/int/matmul.cc`, `fp/fp16_matmul.cc`). The skel disassembles with
`hexagon-llvm-objdump --mv81 --mhvx` (my first pass missed `--mhvx`, hence the
`<unknown>`s): HMX **control** ops decode (`mxclracc.hf`, …); the matrix-MAC
opcodes (class `0x92` = `mxmem`/`mxmpy`, `0xa6` = `mx*` convert/store) still need
the internal assembler, but their structure is unambiguous from the symbols and
the surrounding HVX.

### The w4a16 matmul, decoded — int4 is STORAGE-ONLY; the HMX runs fp16
Two-stage, exactly mirroring what an open engine would do (just fused + tiled):

**Stage 1 — int4→fp16 decompress in HVX** (`expand_bq_s4_to_pkweights_fp16`,
`QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION`). Decoded inner loop:
```
v16 = vmem(r1++#1)                       ; load packed 4-bit weights
v2  = vand(v16, 0x0f000f00)              ; mask a nibble field
v16.uw = vrotr(v16.uw, ...)              ; rotate to next nibble
v5.b = vlut32(v3.b, v12.b, r6)           ; LUT: int4 code -> value (codebook)
v7:6.qf32 = vmpy(v4.hf, v0.hf)           ; * per-block fp16 scale (v0)
v13.hf = v7:6.qf32                        ; -> fp16
vmem(r0++#1) = v13                        ; store to pkWeightsF16
```
i.e. **`vand` (nibble mask) + `vrotr` (nibble select) + `vlut32` (codebook) +
`vmpy.hf` (× per-block scale) → fp16**, written to a `pkWeightsF16` (packed-fp16,
Crouton-tiled) buffer. (The int8 sibling `expand_bq_pkweights_s8` is the same
shape with `vmpy(v0.ub, v2.b)`.)

**Stage 2 — fp16 HMX matmul** (`hmx_convf16_1x1_stride1` et al.):
```
mxclracc.hf                 ; clear fp16 accumulator
loop: <0x92 mxmem/mxmpy>    ; stream fp16 weight+activation tiles, MAC-accumulate
      <0xa6 mx* convert>    ; accumulator -> output
```

**Decisive conclusion:** Qualcomm's "w4a16" matmul **dequantizes int4 weights to
fp16 (in HVX) and runs the HMX systolic array in fp16** — the SAME arithmetic as
llama.cpp, and consistent with our on-device probe that found **no native int4×fp16
HMX primitive** (`hmx_single_tile_probe_findings.md`). int4 is purely a
storage/bandwidth format, expanded just-in-time. There is `set_hmx_params_convw4b1x1`
(w4) / `convw2b1x1` (w2) param setup, but the MAC kernel is `hmx_convf16_*`.

**So the speed is NOT a magic int4 matmul.** The difference vs llama.cpp is purely
*how well* the dequant + matmul are fused/tiled/scheduled: Qualcomm expands int4→
fp16 **into Crouton-packed VTCM tiles streamed straight into the HMX** (the
`pkWeightsF16` path), within one finalised, statically-scheduled, weight-resident
graph — vs llama.cpp's separate `q4_0_to_fp16_lut` pass + per-op dispatch.

### Can we reuse the skel in llama.cpp? No.
The skel exposes only a **proprietary FastRPC IDL** (`qnn_skel_handle_invoke`,
`file:///libqnn_skel.so?...&_modver=1.0`), not a callable kernel ABI or a
`QnnOpPackage`. (a) Reversing the IDL = re-implementing the QnnHtp runtime;
(b) embedding QnnHtp + a 1-op context binary = just running QNN (= ORT-QNN, with
the known ~7-session/~10 GB ceiling, and two runtimes contending for one HMX);
(c) extracting the kernel object is blocked by Crouton/VTCM/HMX-env coupling, no
stable C++ ABI, and a FirstParty-vs-GPL **license conflict**. ggml-hexagon wrote
its own HMX kernels precisely to avoid this. **The reusable asset is the
algorithm above, not the binary.**

## Phase 4 — delta vs llama.cpp + synthesis

### Measured throughput (same X2 Elite, this project)
| path | prefill | decode | quant | data movement |
|---|---|---|---|---|
| **Qualcomm w4a16, ORT-QNN, AR128** | **2229 pp** | 27.8 tg | w4 / uint16 act / uint8 KV | shared_buffer zero-copy, mmap weights |
| llama.cpp HTP, Q4_0 | ~102–175 pp (plateau) | ~18 tg | Q4_0 → **fp16 dequant** then fp16 HMX | per-op DDR↔VTCM, LUT dequant pre-pass |

The ~13× prefill gap is **not** a magic matmul (Phase 3 shows the w4 Conv is only
~15% of prefill, and the probe in the sibling doc proved there is no native
int4×fp16 HMX primitive). It decomposes as:

1. **Per-op overhead amortisation (the big one).** Each layer is ~hundreds of
   ops (Phase 3: 792 MatMul + 600 Conv + 1945 Mul + … across 12 layers). Every op
   has a fixed ~10–18k-cycle floor (kernel launch + VTCM acquire + weight DMA).
   Qualcomm's **AR128** graph pays that floor once per op for **128 tokens**
   (~46× more cycle-efficient/token than AR1). llama.cpp batches too but
   **plateaus ~170 pp** — its per-op dispatch + graph-build overhead and VTCM
   churn keep it from amortising as well as the **fully-compiled, statically-
   scheduled QNN context** (one finalised HTP program for the whole 12-layer
   part, weights resident, IO shared-buffer).
2. **Weight bandwidth.** Qualcomm keeps weights **w4 in VTCM** (¼ the DDR→VTCM
   traffic) and decompresses inline; llama.cpp's HTP path runs a separate
   `q4_0_to_fp16_lut` pass that **materialises full fp16 weight tiles** → 2×
   weight bandwidth + an extra pass. Prefill is bandwidth-sensitive.
3. **Zero-copy IO + uint8 KV + burst + poll.** shared_buffer removes the FastRPC
   IO copy; uint8 KV is ¼/½ the KV traffic; burst pins clocks; poll removes
   wakeup latency. llama.cpp's backend does more explicit DDR↔VTCM movement.
4. **Whole-graph fusion/scheduling.** The QNN compiler fuses RoPE/norm/dequant/
   matmul and schedules DMA double-buffering across the finalised layer graph;
   llama.cpp executes ggml ops one-at-a-time with the scheduler between them.

### What this means for an open engine (actionable)
- **The single biggest lever is reducing per-op overhead / increasing fusion +
  static scheduling**, not the quant scheme — Qualcomm's win is mostly that a
  whole 12-layer part is ONE finalised, weight-resident, shared-buffer HTP
  program executed over a 128-wide batch.
- **Cheap, no-new-kernel win for llama.cpp:** keep Q4_0 weights packed into VTCM
  and decompress **inline/fused** (drop the `q4_0_to_fp16_lut` full-tile pre-pass)
  → recover much of lever #2 with the existing fp16 HMX matmul.
- **uint8 KV + shared-buffer IO** in the llama.cpp HTP backend → lever #3.
- w4a8 (integer HMX, sibling doc) is a *later* increment — it adds throughput +
  ¼ activation bandwidth but is gated on the undocumented HMX integer ISA.

---

## Executive summary — how Qualcomm hits 2200+ pp
1. **w4 weights, uint16 (fixed-point) activations, uint8 KV**, uint16 tied
   embed/lm_head. The matmul is int4-weight × uint16-act, decompressed inline
   (no native int4×fp16 primitive exists — proven in the sibling probe doc).
2. **AR128 batched prefill** processes 128 tokens per graph call, amortising the
   ~hundreds-of-ops-per-layer fixed overhead + weight DMA → **~46× more
   cycle-efficient per token than AR1 decode** (the dominant lever).
3. **One finalised, statically-scheduled HTP program per 12-layer part**, with
   **weight sharing** across all 10 graphs (prefill+decode × 5 ctx tiers), so the
   615 MB w4 weights load once (730 ms) and stay resident.
4. **Zero-copy data movement**: shared_buffer IO, mmap weights, uint8 KV, burst
   clocks, busy-poll completion.
5. In prefill the **w4 matmul (Conv) is only ~15%** of cycles; attention (34%)
   and elementwise (22%) dominate — i.e. the quant/matmul is already cheap and
   amortised, and the rest is well-tiled and fused. The gap to llama.cpp is
   overhead/fusion/bandwidth, not arithmetic.

Reproduce: `scripts/dissect_qualcomm_bundle.py` (Phase 1),
`scripts/qnn_profile_allgraphs.py` + `qnn-net-run` + `qnn-profile-viewer` +
`scripts/parse_qnn_profile.py` (Phase 3); DSP transport via ORT's signed skel.

## Phase 4 — delta vs llama.cpp + synthesis
_(in progress)_
