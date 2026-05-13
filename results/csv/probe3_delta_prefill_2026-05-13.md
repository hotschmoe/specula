# Probe 3: delta-prefill behavior via cache_prompt on OpenCL -ngl 0
Build: C:\Users\hotschmoe\Documents\GitHub\specula\llama.cpp\build-opencl\bin\llama-server.exe
Model: 35B-A3B MXFP4_MOE, -ngl 0 -t 18, -c 16384

| turn | prompt_n | prompt_ms | PP t/s | predicted_ms | TG t/s | wall s | cache_n |
|---|---:|---:|---:|---:|---:|---:|---:|
| turn1_cold | 5457 | 84246 | 64.8 | 2317 | 27.63 | 86.8 | 0 |
| turn2_cached | 524 | 11919 | 44.0 | 1535 | 28.02 | 13.5 | 4941 |
| turn3_rerun | 4 | 69 | 57.8 | 1515 | 28.39 | 1.6 | 5461 |
