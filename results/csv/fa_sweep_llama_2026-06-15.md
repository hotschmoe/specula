# FA on/off sweep — Llama-3.2-3B Q4_0 (cross-family, non-Qwen)

## CPU -ngl0 -t16
| model                          |       size |     params | backend    | threads |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --: | --------------: | -------------------: |
| llama 3B Q4_0                  |   1.78 GiB |     3.21 B | CPU        |      16 |   0 |           pp512 |       537.38 ± 12.54 |
| llama 3B Q4_0                  |   1.78 GiB |     3.21 B | CPU        |      16 |   0 |           tg128 |         64.19 ± 0.46 |
| llama 3B Q4_0                  |   1.78 GiB |     3.21 B | CPU        |      16 |   1 |           pp512 |        292.99 ± 2.34 |
| llama 3B Q4_0                  |   1.78 GiB |     3.21 B | CPU        |      16 |   1 |           tg128 |         68.86 ± 0.07 |

build: e37abd6b5 (9617)

## OpenCL -ngl99 -ub512 -t16
| model                          |       size |     params | backend    | ngl | threads |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ------: | --: | --------------: | -------------------: |
| llama 3B Q4_0                  |   1.78 GiB |     3.21 B | OpenCL     |  99 |      16 |   0 |           pp512 |        790.03 ± 3.15 |
| llama 3B Q4_0                  |   1.78 GiB |     3.21 B | OpenCL     |  99 |      16 |   0 |           tg128 |         38.18 ± 0.14 |
| llama 3B Q4_0                  |   1.78 GiB |     3.21 B | OpenCL     |  99 |      16 |   1 |           pp512 |        407.85 ± 2.03 |
| llama 3B Q4_0                  |   1.78 GiB |     3.21 B | OpenCL     |  99 |      16 |   1 |           tg128 |         36.73 ± 0.07 |
