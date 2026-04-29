# Qualcomm Qwen2.5-7B w8a16 oracle trace

- bundle: `models\qualcomm-qwen2_5-7b-ref\qwen2_5_7b_instruct-genie-w8a16-qualcomm_snapdragon_x2_elite`
- mode: AR1 / CL4096, greedy argmax
- prompt: 5 tokens, generation steps: 8
- prefill mean latency: 54.1 ms/step
- generation mean latency: 45.5 ms/step
- logits quant: scale=0.001054224674589932, offset=-31902

## Generated tokens

0. id=624  tok='.Ċ'
1. id=32  tok='A'
2. id=13  tok='.'
3. id=12095  tok='ĠParis'
4. id=198  tok='Ċ'
5. id=33  tok='B'
6. id=13  tok='.'
7. id=55201  tok='ĠLyon'

## Decoded continuation

```
.
A. Paris
B. Lyon
```

## Saved

- C:\Users\hotschmoe\Documents\GitHub\specula\results\qualcomm_qwen2_5_7b_oracle.npz
- C:\Users\hotschmoe\Documents\GitHub\specula\results\qualcomm_qwen2_5_7b_oracle.md
