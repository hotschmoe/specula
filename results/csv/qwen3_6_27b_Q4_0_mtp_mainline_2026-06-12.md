# Qwen3.6-27B-MTP-Q4_0 MTP sweep via llama-server (MAINLINE)
Build: mainline e37abd6b5 (was PR #22673 build in session 27)
Model: C:\Users\hotschmoe\Documents\GitHub\specula\models\Qwen3.6-27B-MTP-Q4_0.gguf

| config | n_max | PP t/s | TG t/s | draft_n | accepted | accept % |
|---|---:|---:|---:|---:|---:|---:|
| ngl=0 t=18 no-MTP | 0 | 47.44 | 7.75 | 0 | 0 | 0 |
| ngl=0 t=18 MTP n4 | 4 | 52.93 | 12.40 | 311 | 177 | 56.9 |
| ngl=0 t=18 MTP n8 | 8 | 45.39 | 10.09 | 512 | 190 | 37.1 |
| ngl=0 t=18 MTP n12 | 12 | 47.99 | 7.46 | 746 | 192 | 25.7 |
