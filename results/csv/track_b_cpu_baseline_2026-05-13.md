| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen35moe 35B.A3B MXFP4 MoE    |  20.21 GiB |    34.66 B | CPU        |      18 |           pp512 |        186.88 ± 0.00 |
| qwen35moe 35B.A3B MXFP4 MoE    |  20.21 GiB |    34.66 B | CPU        |      18 |           tg128 |         31.45 ± 0.00 |

build: 856c3adac (9128)
