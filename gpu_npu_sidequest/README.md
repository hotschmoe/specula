# gpu_npu_sidequest — workspace

Scratch + artifacts for the GPU↔NPU placement side quest.
Plan: `docs/gpu_npu_placement_sidequest.md`.

## Layout

- `scripts/`  — microbench + harness scripts written for this side quest
- `results/`  — CSV outputs (final CSVs also copied to `results/csv/`)
- `logs/`     — raw run logs
- `findings/` — per-phase markdown writeups

## Shared CSV schema

All Phase 0 CSVs (`results/phase0_<island>.csv`) use these columns:

| column      | meaning |
|-------------|---------|
| island      | `gpu` \| `npu` \| `cpu` |
| op_shape    | `PP` (batched k-token forward) \| `TG` (autoregressive 1-token) |
| k           | token count of the forward pass (1 for TG) |
| model       | e.g. `qwen3-4b-q4km` |
| backend     | `llama.cpp-opencl` \| `ort-qnn` \| `llama.cpp-cpu` |
| power       | `AC` \| `battery` |
| tok_per_s   | throughput (for PP: k / pass_time) |
| ms_per_pass | wall time of one k-token forward pass |
| notes       | free text |

`op_shape=PP` swept over k is the verify-shape microbench; `op_shape=TG`
is the draft-shape anchor. The k where the GPU and NPU per-pass-cost
curves cross is the Phase 0 deliverable.
