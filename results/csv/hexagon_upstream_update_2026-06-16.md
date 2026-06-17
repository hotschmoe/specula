# Hexagon backend upstream update — 4.5x prefill (2026-06-16)

llama.cpp hotschmoe-npu-work: 45cac7c (2026-04-17) -> 74ade52 (master), 34 hexagon commits.
Qwen3-4B-Q4_0, HTP0, -ngl 99 -t 16, burst, AC.

| build | config | pp512 | pp1024 | pp2048 | tg |
|-------|--------|------:|-------:|-------:|---:|
| base 45cac7c | -fa 0 | 187.4 | - | - | 19.9 |
| master 74ade52 | -fa 0 | 499.1 | 418.5 | 315.8 | - |
| master 74ade52 | **-fa 1** | **844.7** | 783.0 | 686.5 | 21.6 |

Key: HMX flash attention (#22347) makes -fa 1 win (was -fa 0). Gap to Qualcomm
w4a16 bundle (2229 pp ORT-QNN): 11.9x -> 2.64x. Pin -fa 1 on HTP.
Levers in the update: #23368 (HMX matmul rework), #23835 (op fusion), #22347
(HMX FA), #22334 (max clock), #23989 (MUL_MAT/FA opt).

## Profiling the remaining 2.64x (GGML_HEXAGON_PROFILE=1, -v)

pp512 forward = ONE op-batch of 687 ops, htp-ops-usec 595883 (=596ms, matches 844 t/s).
Op-type counts confirm all upstream wins are ENGAGED:
- MUL_MAT 504 (7 projections x 72) | RMS_NORM+MUL 290 (FUSION working, #23835)
- FLASH_ATTN_EXT 72 (HMX flash attn #22347) | SWIGLU 72 | ROPE/SET_ROWS/ADD 144 each
Per-op usec NOT exposed in mode 1 (metadata only; batch total only). The 2.64x
residue is architectural (687 separate ggml ops vs QNN's one finalized graph),
not a single fixable op.

## Qwen3-14B-Q4_0 (8.5GB) on new build, 4 sessions, -fa 1

| config | pp512 | tg16 |
|--------|------:|-----:|
| NDEV=4 HTP0-3 -ngl 34 | 156.8 | 12.17 |
| NDEV=4 HTP0-3 -ngl 99 | 153.2 | 11.38 |

-ngl 34 (small hybrid offload) slightly beats all-on-HTP. Both crash on TEARDOWN
(dspqueue_write 0x0e, 4-session churn — known fragility) but results are valid.
Q4_0 GGUF IS the w4a16-equivalent (int4 weights -> fp16 HMX); no special download.

## Qwen3.6-35B-A3B-MXFP4_MOE (20.2 GiB) — max-NPU-residency for background/battery UX

New build, -fa 1, GGML_HEXAGON_NDEV=4, HTP0-3. Goal: max NPU residency (silent,
power-efficient, leaves CPU/GPU free), speed secondary.

| -ngl | pp | tg16 | NPU resident | overflow |
|-----:|---:|-----:|-------------:|---------:|
| 24 | pp512 138.8 | 15.42 | (less) | more on GPU/CPU (faster) |
| 30 | pp128 79.9 | 15.55 | | |
| 40 | pp128 65.2 | - | | |
| **99 (max NPU)** | pp128 63.7 | 13.4 | **~12 GB (HTP-REPACK)** | **~8 GB CPU_REPACK** |

- At -ngl 99: **~12 GB of the 20 GB model is NPU-resident**, ~8 GB spills to CPU
  (the MXFP4 MoE experts). 41/41 offloadable layers on HTP. tg16 13.4 (A3B = ~3B
  active/token, so decode beats the 14B's 12.2).
- Each HTP session maps 3.35 GB vmem (v81 default) -> ~13.4 GB HTP-mappable (more
  than the old ~8 GB assumption).
- Tradeoff confirmed: more -ngl = more NPU residency but slower pp (HTP prefill <
  GPU for MoE). User priority = residency -> -ngl 99.
- Overflow currently -> CPU. Redirect to GPU (keep CPU free) needs --override-tensor
  routing the *_exps tensors to GPUOpenCL (adding GPUOpenCL to --device alone
  didn't move them). TODO.

## Teardown crash FIXED (host-side, ggml-hexagon.cpp:2180)

Root cause: 4-session heavy load fills the DSP request queue; host flush_batch
used finite DSPQUEUE_TIMEOUT and GGML_ABORT'd on AEE_EWOULDBLOCK (0x0e, queue
full). Fix: on EWOULDBLOCK, drain completed responses (flush_pending) + retry
(matches the op_queue->push backpressure pattern). Host-only change (no skel
rebuild). Required for using these models in a coding harness.
