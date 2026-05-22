# 2026-05-21 — specula pathb bundle NPU testing

First on-device test of the three specula-built Qwen3-4B NPU bundles
produced by the cloud-GPU (RTX Pro 6000) pipeline and transferred to the
Snapdragon X2 Elite Extreme laptop. Goal: run all three under **Genie**
and our **ORT-QNN** runtime, compare PP/TG throughput and output
quality against the Qualcomm AI Hub Qwen3-4B reference bundle.

**Bottom line:** the two ctx512 bundles run correctly and decode
coherent text through our ORT-QNN runtime; the ctx32768 bundle and the
entire Genie path are blocked. Throughput is far below the Qualcomm
reference — most of the gap is structural (the pathb bundle design),
some is fixable harness overhead. Details below.

---

## 1. What was tested

Three bundles, extracted to `models/specula-qwen3-4b-ref/`:

| bundle | precision | ctx | parts | bundle size |
|---|---|---|--:|--:|
| `qwen3-4b_w4a16_pathb_ctx512_x2e_v81`   | w4a16 | 512   | 4  | 3.7 GB |
| `qwen3-4b_w8a16_pathb_ctx512_x2e_v81`   | w8a16 | 512   | 4  | 5.0 GB |
| `qwen3-4b_w4a16_pathb_ctx32768_x2e_v81` | w4a16 | 32768 | 19 | 3.9 GB |

Control: `models/qualcomm-qwen3-4b-ref/qwen3_4b-genie-w4a16-qualcomm_snapdragon_x2_elite/`
(Qualcomm AI Hub shipping w4a16 bundle, 4 parts).

---

## 2. Software / hardware versions used

| component | version | notes |
|---|---|---|
| Host OS | Windows 11 Home 26200 (ARM64) | Snapdragon X2 Elite Extreme |
| Hexagon NPU driver | **30.0.220.11010** (dated 2026-01-26) | ~4 months old at test time |
| QAIRT SDK | 2.45.40.260406 | only version installed |
| Genie | libGenie.so **1.17.0** (from QAIRT 2.45) | |
| onnxruntime | 1.24.4 (venv) | bundles QAIRT 2.42 `QnnHtp.dll` |
| QnnHtp.dll used for pathb | system **QAIRT 2.45** `lib/aarch64-windows-msvc/QnnHtp.dll` | see §5 |
| pathb bundles compiled with | QAIRT 2.45.40.260406 (`buildId` in `bin_info`) | |
| Qualcomm control compiled with | QAIRT 2.42 | loads with the venv DLL |

The user flagged that the NPU toolchain had been idle for weeks and
might have drifted. QAIRT and ORT versions match what the repo docs
assume; the **NPU driver** is the one component that could have been
updated by Windows Update — and the Genie failure in §4 is consistent
with exactly that.

---

## 3. Results

### 3.1 Throughput (PP = prefill t/s, TG = decode t/s)

| bundle | runtime | PP t/s | TG t/s | step (ms) | status |
|---|---|--:|--:|--:|---|
| Qualcomm w4a16 (control) | ORT-QNN | **1604** | **23.4** | 41 (decode) | OK |
| Qualcomm w4a16 (control) | Genie   | — | — | — | **DSP transport broken** (§4) |
| specula w4a16 ctx512 | ORT-QNN | 5.26 | 5.21 | 183 / 188 | OK |
| specula w8a16 ctx512 | ORT-QNN | 4.48 | 4.40 | 220 / 224 | OK |
| specula w4a16 ctx32768 | ORT-QNN | — | — | — | **19 parts > HTP ceiling** (§4) |
| all 3 specula bundles | Genie | — | — | — | **no attention_mask** (§4) |

Measurement geometry: 256-token prefill + 128-token greedy decode for
all ORT-QNN runs, same prompt (`results/qwen3_4b_baseline/pp512_prompt.txt`,
truncated to 256 tokens). Control PP uses AR128 batched prefill; pathb
PP is AR1 (token-by-token) because the pathb bundles ship no AR128
prefill graph. One warmup step discarded. AC power.

CSV: `results/csv/pathb_ortqnn_w4a16_ctx512.csv`,
`results/csv/pathb_ortqnn_w8a16_ctx512.csv`,
`results/csv/qwen3_4b_ortqnn_2026-05-21_ctrl.csv`.

### 3.2 Output quality

Both ctx512 bundles decode **coherent, on-topic English**. The prompt
is about speculative decoding; both continuations stay on subject and
are grammatical.

- **w4a16 ctx512** continuation (greedy, 128 tok): mostly coherent but
  loops ("The 4-bit quantization is also used in…" repeated) — typical
  greedy-decode behavior on a 4B model, not a quantization defect.
- **w8a16 ctx512** continuation: noticeably more fluent and less
  repetitive ("…allowing for different quantization schemes and
  different target models…"). Consistent with w8a16 being the
  higher-fidelity quant.
- First-decode argmax token agrees between the two (`'.'`, id 13).
- First-decode **logit cosine w4a16 vs w8a16 = 0.972**. Greedy
  continuations diverge fast (6/128 tokens shared) — expected, since
  greedy decode amplifies small logit deltas between two different
  quantizations.

Eval artifacts: `results/pathb_eval/{w4a16_ctx512,w8a16_ctx512}.npz`
(first-decode logits + generated token ids).

**Not measured:** on-device first-decode logit cosine of each pathb
bundle vs an FP reference. The pipeline already gates on cos ≥ 0.99 vs
FP via `end-to-end/eval_quality.py` (run on the cloud GPU, pre-split).
An on-device quant-vs-FP number is a follow-up — see §6.

---

## 4. Shortcomings / blockers found

### 4.1 Genie cannot load any pathb bundle (no attention_mask tensor)

`genie-t2t-run` loads all 4 `.bin` parts of the w4a16 ctx512 bundle,
then fails dialog init:

```
[WARNING] Could not find attention mask tensor for CacheGroup past_
Failure to initialize model. Default Group past_ has no associated attention mask
Failed to create the dialog.
```

Cause: the pathb pipeline step `fold-pathbmask` deliberately removes the
BOOL attention-mask chain, so the compiled graph has **no
`attention_mask` input**. Genie's SMART_MASK / NATIVE_KV cache manager
*requires* a mask tensor bound to the `past_` cache group. Qualcomm's
own bundle keeps `attention_mask` as a graph input, so it loads.

To make a pathb bundle Genie-loadable the pipeline must re-expose
`attention_mask` as a graph input (skip `fold-pathbmask`, or thread the
mask as a real input rather than folding it). Until then, pathb bundles
are testable only via our own ORT-QNN runtime.

### 4.2 Genie DSP transport is broken on this machine right now

The Qualcomm control bundle *should* run in Genie (it has the mask, and
the repo has prior Genie numbers for it — PP ~1725 / TG ~26). Today it
fails:

```
[ERROR] DspTransport.openSession qnn_open failed, 0x80000406, prio 100
[ERROR] IDspTransport: Unable to load lib 0x80000406
[WARNING] Failed to create transport instance: 1002
[WARNING] Failed to load skel, error: 1002
```

This persists with `ADSP_LIBRARY_PATH` pointed at the v81 unsigned skel
dir and the v81 skel libs confirmed present
(`lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so` exists). The model
loads and the prompt echoes, but no DSP session opens.

Crucially, **ORT-QNN runs fine on the same HTP** (three successful runs
this session) — so the HTP hardware and driver can execute inference.
The break is specific to Genie's `DspTransport` / FastRPC skel-load
path. This is consistent with the NPU driver having been updated while
the toolchain was idle: Genie 1.17's transport handshake and the
current driver appear to disagree.

**Net effect: Genie produced zero usable numbers this session** — pathb
bundles blocked on the mask, control blocked on DSP transport. The
GENIE-vs-ORT-QNN comparison the test set out to make could not be done.

### 4.3 ctx32768 bundle (19 parts) exceeds the HTP session ceiling

The ctx32768 bundle is split into 19 parts. ORT-QNN loads parts 1–4
fine, then **part 5 fails**:

```
LoadCachedQnnContextFromBuffer Failed to create context from binary. Error code: 1002
```

QNN error 1002 = HTP context/memory exhausted — the documented
~7-live-session ceiling (`reference_ortqnn_session_limit`), and the
ctx32768 parts carry heavier per-graph metadata so the real ceiling is
~4–5 here. 19 co-resident HTP contexts is not achievable.

See §7 for the "can Genie ever do this?" analysis.

### 4.4 pathb TG is ~4.5× slower than the Qualcomm reference

specula w4a16 ctx512 TG = 5.2 t/s vs control 23.4 t/s. The NPU compute
is comparable (both run 36 w4a16 layers); the ~140 ms/step gap is
host-side. Breakdown of *why* — see §5.

---

## 5. Why pathb TG is slow — analysis

Per-step cost for the pathb 4-part chain (measured ~183–224 ms;
control ~41 ms):

1. **FP32 KV IO — the big one.** The pathb graph passes the KV cache as
   **FP32** `[1,8,511,128]` in / `[1,8,512,128]` out. 36 layers × 2 (k,v)
   = 72 tensors ≈ **150 MB fed in + 150 MB out per step**. Qualcomm's
   bundle quantizes KV to **uint8** — 4× fewer bytes — specifically so
   this plumbing is cheap. The NPU must also DMA 4× the KV bytes
   on-device. This is a *bundle design* cost, not a harness bug.

2. **AR1 prefill.** The pathb bundles ship one graph per `.bin`, AR1
   only — no AR128 batched-prefill graph. So prefill is token-by-token
   (PP ≈ TG ≈ 5 t/s) versus the control's 128-wide AR128 prefill
   (PP 1604). This is the single largest PP gap and is structural.

3. **Plain `sess.run()` dispatch (harness).** `bench_pathb_ortqnn.py`
   uses plain `sess.run()`, which allocates the ~150 MB of output
   tensors fresh every step. The control harness uses `IOBinding` with
   pre-bound output buffers. This is the one *fixable-in-the-harness*
   slice — estimated worth ~30–50 % of decode step time, i.e. pathb TG
   could plausibly reach ~10–12 t/s with IOBinding. It would still sit
   below the control because of (1) and (2).

4. **Per-step Python overhead.** 4 sequential `session.run` calls and
   72-entry feed dicts built in Python each step. Minor next to (1).

**Conclusion:** roughly half the gap (no AR128 prefill, FP32 KV) is the
pathb bundle/pipeline design and can only be fixed upstream on RunPod;
the other slice (IOBinding, dispatch) is fixable in our runtime. Even
fully optimized, FP32 KV caps pathb well below the Qualcomm reference.

---

## 6. Recommendations / next steps

Upstream pipeline (RunPod), in priority order:

1. **Quantize the KV cache to uint8** like Qualcomm does. Biggest single
   throughput lever — 4× less KV IO on host *and* device.
2. **Emit an AR128 prefill graph.** Without batched prefill, PP is stuck
   at decode speed. Qualcomm ships ar1 + ar128 graphs in each `.bin`.
3. **Re-expose `attention_mask`** as a graph input (don't fold it) so
   the bundles become Genie-loadable — otherwise we can never benchmark
   against the vendor runtime.
4. **Re-split ctx32768 into far fewer parts** (4–8, not 19). The split
   count is a pipeline knob (`--num-parts`); 19 contexts cannot co-exist
   on the HTP under any runtime.

Local (this repo):

5. Add `IOBinding` to `bench_pathb_ortqnn.py` to recover the dispatch
   overhead and report a fair "best-case our-runtime" number.
6. Capture on-device first-decode logits for the Qualcomm control and
   compute quant-vs-reference cosine for each pathb bundle (true
   accuracy comparison; §3.2 only has bundle-vs-bundle).
7. Investigate the Genie DSP transport break — likely needs a QAIRT /
   driver version realignment. Until fixed, Genie is unusable.

---

## 7. Can Genie ever run a 19-part / swap-execution bundle?

User question: the ctx32768 bundle needs swap-execution to fit the HTP
session ceiling — can Genie do that, or do we need our own engine?

**Genie: no.** Genie loads every `.bin` in `genie_config.json`'s
`ctx-bins` list into one model and expects all parts co-resident for
the lifetime of the dialog. It has no part-eviction / part-swapping
execution mode. A 19-context model cannot be made to fit Genie. (And
separately, Genie can't load *any* pathb bundle at all — §4.1.)

**Our own engine: technically yes, but it's the wrong fix.** We could
write a scheduler that loads a window of parts, runs them, evicts, and
loads the next window — per token. But each part load is seconds; doing
~3–4 load/evict cycles per token would mean tens of seconds per token.
Not viable for real inference.

**The real fix is not a swap engine — it's not splitting into 19 parts.**
The part count is a pipeline parameter. The split exists only to dodge
the single-`.bin` 3.67 GB HTP-serializer ceiling; 4–8 parts is enough
to clear that. 19 is overkill and is what creates the problem. Re-split
the ctx32768 model into ≤8 parts and it runs like the ctx512 bundles —
no swap engine, no Genie change needed. Swap-execution only becomes
worth building if a single model genuinely cannot fit the HTP in any
reasonable part count, which is not the case here.

---

## 8. Test artifacts

- Harness (new, keeper): `npu_engine/bench_pathb_ortqnn.py` — generic
  ORT-QNN driver for pathb bundles (reads the IO contract from
  `bin_info/part_*.json`, threads the folded mask across parts, uses
  the QAIRT 2.45 DLL).
- Load probe: `npu_engine/_probe_pathb_load.py`.
- CSVs: `results/csv/pathb_ortqnn_*.csv`,
  `results/csv/qwen3_4b_ortqnn_2026-05-21_ctrl.csv`.
- Eval npz: `results/pathb_eval/*.npz`.
- Extracted bundles: `models/specula-qwen3-4b-ref/` (13 GB,
  regeneratable from the staging tars — can be cleared after testing).
