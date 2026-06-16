# Next session — run the 14B w8a16 bundle on the Hexagon NPU

Kickoff brief for a clean session. The **build** is done; this is the
**runtime/deploy** workstream: load and decode the 10-part Qwen3-14B w8a16
bundle on the X2E Hexagon via ORT-QNN.

## Where things stand (read these first)

- `current_status.md` (top) — the autonomous build-server run.
- `docs/threadripper_build_server.md` — how the bundle was built + every
  14B-scale finding (incl. the split-balancing rules).
- The bundle is **on the X2E**, sha256-verified:
  `models/qwen3_14b-w8a16-specula-x2e/` — `*_part_1..10_of_10.bin` (28 GB) +
  `genie_config.json` + `htp_backend_ext_config.json` + tokenizer/config +
  `metadata.json`. Also on the box (`runs/.../10_bundle/`) + `06_split` kept
  for part rebuilds.

## The goal

Load the 10 parts on the Hexagon and run a coherent decode (then a logit
cos-sim check vs the CPU/ORT fp reference). Not fast — **loadable + correct**
first.

## The central obstacle — session ceiling

The runtime loads each `.bin` as an HTP context. **ORT-QNN tops out at ~7 HTP
sessions** ([[reference_ortqnn_session_limit]]); we have **10 parts**. Naive
"one session per part" will fail with **QNN 1002** at the 8th.

The known fix ([[reference_ortqnn_session_limit]],
`docs/npu_engine_prefill_sidequest.md`): a **combined wrapper** — one ORT-QNN
session per `.bin` but multiple **EPContext nodes** referencing graphs in the
same binary, so N parts collapse to ≤7 sessions. (Note the *rejected* naive
combined-wrapper from the 4B sidequest — it failed on duplicate input names
across AR1/AR128; here we only have AR1 decode graphs, so re-check whether the
simpler form works.) Alternatively, the **mode-swap sidecar**
(`npu_engine/sidecar.py`) holds a subset loaded and swaps — but that's for
AR1/AR128 mode switching, not for >7 *parts*; the parts must coexist.

## Part topology (how to wire the runtime)

10 parts, threaded by the residual **seam** + per-attention-layer **KV cache**:
- **part1**: `input_ids[1,1]` → embed → `Gather_output_0[1,1,5120]` (seam)
- **part2..9** (decoder, 5 layers each, layers 0..39): inputs = prev seam
  `[1,1,5120]` + `position_ids_cos/sin[1,1,128]` + per-layer
  `past_key_values.{L}.key/value[1,8,511,128]` + `attention_bias[1,1,1,512]`;
  outputs = next seam + per-layer `present.{L}.key/value[1,8,512,128]`
- **part10**: seam `[1,1,5120]` → final_norm + lm_head → `logits[1,1,151936]`

Exact specs: `end-to-end/build_server/split_14b.py` + `split_tail2.py` (and
`lib/split.py::build_part_specs`). The runtime feeds part k's seam+present into
part k+1, maintains the KV ring buffer, and computes cos/sin + attention_bias
per step (see `lib/cal.py` for the exact rope/mask construction; ctx=512).

## Key files

- `npu_engine/bench_pathb_ortqnn.py` — the ORT-QNN engine/bench harness (the
  4B runner; generalize to 10 parts + the combined-wrapper).
- `npu_engine/sidecar.py` — long-lived mode-swap engine.
- `docs/npu_engine_prefill_sidequest.md` — session-ceiling analysis + the
  rejected/accepted wrapper approaches.
- ORT-QNN must match QAIRT 2.45.40 ([[reference_ort_qnn_qairt_match]]) — use
  the 2.1.0 ORT-QNN on the X2E.

## Suggested first steps

1. Load **part1 alone** via ORT-QNN on the Hexagon → confirm a single `.bin`
   loads + runs (embed lookup). Smallest possible loadability check.
2. Load **2–3 decoder parts** chained (seam + KV threading) → confirm the
   multi-part wiring + correct hidden states (cos vs fp).
3. Solve the **>7-session** problem (combined wrapper / EPContext) to get all
   10 loaded at once.
4. Full decode + logit cos-sim vs CPU reference; then a coherent-text smoke.

## Watch out for

- The X2E NPU driver / Genie DSP transport break ([[reference_genie_dsp_transport_broken]])
  — use **ORT-QNN**, not Genie (the bundle is genie-shaped but we run our own
  engine).
- w8a16 was basic-PTQ with **no calibration** — accuracy may be rough; a logit
  cos check tells us if it's usable or needs the calib pass.
- The same engine will serve the **w4a16** bundle (`docs/qwen3_14b_w4a16_plan.md`)
  and eventually the **27B** — build it general over part count + topology.
