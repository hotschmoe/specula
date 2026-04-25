"""Custom ORT-QNN runtime for Qualcomm Hexagon NPU.

Hosts the chained 4-partition driver for the Qwen3-4B (and successors)
Genie w4a16 bundles, plus the bench/probe/comparison scripts that
exercise it. Designed to grow into the full speculative-decode sidecar
that runs prefill + draft on NPU and verify on CPU.

Modules:
  qualcomm_qwen3_4b_oracle   — chained ORT-QNN runtime + KV stitching
                                (the canonical engine; everything else
                                imports from here)
  bench_qwen3_4b_ortqnn      — single-stream PP/TG bench
  bench_concurrency4_npu_ortqnn — N=4 concurrent-stream bench
  specula_qwen3_4b_oracle    — companion oracle for spec-decode validation
  compare_local_vs_qualcomm_oracle — logit-cos parity check
  probe_qualcomm_qwen3_4b    — minimal load+run smoke test (single partition)
"""
