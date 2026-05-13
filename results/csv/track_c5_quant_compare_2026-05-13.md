# Quant comparison — Qwen3-4B (CPU-kleidiai @ -t 16, OpenCL ngl=99 @ -t 16)

## Qwen3-4B-Q4_K_M.gguf — CPU-kleidiai
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen3 4B Q4_K - Medium         |   2.32 GiB |     4.02 B | CPU        |      16 |           pp512 |        280.20 ± 6.12 |
| qwen3 4B Q4_K - Medium         |   2.32 GiB |     4.02 B | CPU        |      16 |           tg128 |         31.66 ± 1.13 |

build: 856c3adac (9128)
## Qwen3-4B-Q4_K_M.gguf — OpenCL ngl=99
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen3 4B Q4_K - Medium         |   2.32 GiB |     4.02 B | OpenCL     |  99 |      16 |           pp512 |      240.16 ± 175.19 |
| qwen3 4B Q4_K - Medium         |   2.32 GiB |     4.02 B | OpenCL     |  99 |      16 |           tg128 |        11.58 ± 11.55 |
## Qwen3-4B-Q4_K_M.gguf — OpenCL ngl=0
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen3 4B Q4_K - Medium         |   2.32 GiB |     4.02 B | OpenCL     |   0 |      16 |           pp512 |        296.27 ± 3.17 |
| qwen3 4B Q4_K - Medium         |   2.32 GiB |     4.02 B | OpenCL     |   0 |      16 |           tg128 |         44.60 ± 0.04 |

## Qwen3-4B-Q4_0.gguf — CPU-kleidiai
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen3 4B Q4_0                  |   2.21 GiB |     4.02 B | CPU        |      16 |           pp512 |        269.71 ± 1.63 |
| qwen3 4B Q4_0                  |   2.21 GiB |     4.02 B | CPU        |      16 |           tg128 |         42.55 ± 1.19 |

build: 856c3adac (9128)
## Qwen3-4B-Q4_0.gguf — OpenCL ngl=99
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen3 4B Q4_0                  |   2.21 GiB |     4.02 B | OpenCL     |  99 |      16 |           pp512 |        581.21 ± 0.76 |
| qwen3 4B Q4_0                  |   2.21 GiB |     4.02 B | OpenCL     |  99 |      16 |           tg128 |         26.84 ± 0.25 |
## Qwen3-4B-Q4_0.gguf — OpenCL ngl=0
| model                          |       size |     params | backend    | ngl | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --------------: | -------------------: |
| qwen3 4B Q4_0                  |   2.21 GiB |     4.02 B | OpenCL     |   0 |      16 |           pp512 |        384.50 ± 5.16 |
| qwen3 4B Q4_0                  |   2.21 GiB |     4.02 B | OpenCL     |   0 |      16 |           tg128 |         50.50 ± 0.20 |
