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
