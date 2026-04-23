# Qualcomm Qwen3-4B w4a16 oracle trace

- bundle: `models\qualcomm-qwen3-4b-ref\qwen3_4b-genie-w4a16-qualcomm_snapdragon_x2_elite`
- mode: AR1 / CL512, greedy argmax
- prompt: 30 tokens, generation steps: 32
- prefill mean latency: 39.5 ms/step
- generation mean latency: 39.4 ms/step
- logits quant: scale=0.0013899723052105387, offset=-26872

## Generated tokens

0. id=198  tok='Ċ'
1. id=32313  tok='Okay'
2. id=11  tok=','
3. id=279  tok='Ġthe'
4. id=1196  tok='Ġuser'
5. id=374  tok='Ġis'
6. id=10161  tok='Ġasking'
7. id=330  tok='Ġ"'
8. id=3838  tok='What'
9. id=374  tok='Ġis'
10. id=23249  tok='Ġgravity'
11. id=7521  tok='?"'
12. id=323  tok='Ġand'
13. id=6801  tok='Ġwants'
14. id=279  tok='Ġthe'
15. id=4226  tok='Ġanswer'
16. id=1212  tok='Ġunder'
17. id=5779  tok='Ġten'
18. id=4244  tok='Ġwords'
19. id=13  tok='.'
20. id=6771  tok='ĠLet'
21. id=752  tok='Ġme'
22. id=1744  tok='Ġthink'
23. id=382  tok='.ĊĊ'
24. id=5338  tok='First'
25. id=11  tok=','
26. id=358  tok='ĠI'
27. id=1184  tok='Ġneed'
28. id=311  tok='Ġto'
29. id=10339  tok='Ġexplain'
30. id=23249  tok='Ġgravity'
31. id=304  tok='Ġin'

## Decoded continuation

```

Okay, the user is asking "What is gravity?" and wants the answer under ten words. Let me think.

First, I need to explain gravity in
```

## Saved

- C:\Users\hotschmoe\Documents\GitHub\specula\results\qualcomm_qwen3_4b_oracle.npz
- C:\Users\hotschmoe\Documents\GitHub\specula\results\qualcomm_qwen3_4b_oracle.md
