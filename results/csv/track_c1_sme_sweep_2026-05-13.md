# SME env sweep — Qwen3-4B Q4_K_M on build-cpu-kleidiai

## GGML_KLEIDIAI_SME=0
| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen3 4B Q4_K - Medium         |   2.32 GiB |     4.02 B | CPU        |       8 |           pp512 |        177.32 ± 2.54 |
| qwen3 4B Q4_K - Medium         |   2.32 GiB |     4.02 B | CPU        |       8 |           tg128 |         39.61 ± 0.35 |

build: 856c3adac (9128)

## GGML_KLEIDIAI_SME=1

## GGML_KLEIDIAI_SME=2

## GGML_KLEIDIAI_SME=4

## GGML_KLEIDIAI_SME=8

## GGML_KLEIDIAI_SME=16
