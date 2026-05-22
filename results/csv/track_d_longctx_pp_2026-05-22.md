# Track D - long-ctx PP curve, Qwen3.6-35B-A3B MXFP4, OpenCL, -t 16, fa 0
#
# PARTIAL - sweep stopped early 2026-05-22 (laptop needed). A1 complete
# except the ngl99/ub2048/pp32768 cell (dropped - investigate).
# 128K PP and TG-at-depth pending: run scripts/track_d_overnight_2026-05-22.ps1
# -> results/csv/track_d_longctx_pp_2026-05-22_tail.md

## PP 8192/32768 x ngl{0,99} x ub{512,2048}
| model                          |       size |     params | backend    | ngl | threads | n_ubatch |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | -------: | --------------: | -------------------: |
| qwen35moe 35B.A3B MXFP4 MoE    |  20.21 GiB |    34.66 B | OpenCL     |   0 |      16 |      512 |          pp8192 |        148.80 ┬▒ 0.00 |
| qwen35moe 35B.A3B MXFP4 MoE    |  20.21 GiB |    34.66 B | OpenCL     |   0 |      16 |      512 |         pp32768 |         89.97 ┬▒ 0.00 |
| qwen35moe 35B.A3B MXFP4 MoE    |  20.21 GiB |    34.66 B | OpenCL     |   0 |      16 |     2048 |          pp8192 |        129.42 ┬▒ 0.00 |
| qwen35moe 35B.A3B MXFP4 MoE    |  20.21 GiB |    34.66 B | OpenCL     |   0 |      16 |     2048 |         pp32768 |         81.82 ┬▒ 0.00 |
| qwen35moe 35B.A3B MXFP4 MoE    |  20.21 GiB |    34.66 B | OpenCL     |  99 |      16 |      512 |          pp8192 |        159.54 ┬▒ 0.00 |
| qwen35moe 35B.A3B MXFP4 MoE    |  20.21 GiB |    34.66 B | OpenCL     |  99 |      16 |      512 |         pp32768 |        114.62 ┬▒ 0.00 |
| qwen35moe 35B.A3B MXFP4 MoE    |  20.21 GiB |    34.66 B | OpenCL     |  99 |      16 |     2048 |          pp8192 |        174.40 ┬▒ 0.00 |

## PP 131072 x ngl{0,99}, ub 2048
