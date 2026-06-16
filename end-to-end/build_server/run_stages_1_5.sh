#!/bin/bash
set -e
WS=/workspace; PY=$WS/.venv-box/bin/python
RUN=$WS/runs/qwen3_14b_w8a16; MODEL=$WS/models/Qwen3-14B
mkdir -p $RUN
[ -f $RUN/01_optimum/model.onnx ] && echo "skip 1 (done)" || { echo "=== stage 1 export ==="; $PY $WS/end-to-end/scripts_helper/optimum_export_4b.py --model-path $MODEL --out-dir $RUN/01_optimum; }
[ -f $RUN/02_staged/model.onnx ] && echo "skip 2 (done)" || { echo "=== stage 2 htp stage ==="; $PY $WS/scripts/rewrite_qwen3_htp.py --mode stage --optimum-dir $RUN/01_optimum --staged-dir $RUN/02_staged; }
[ -f $RUN/03_pathbmask/model.onnx ] && echo "skip 3 (done)" || { echo "=== stage 3 fold-pathbmask ==="; $PY $WS/scripts/rewrite_qwen3_htp.py --mode fold-pathbmask --optimum-dir $RUN/01_optimum --staged-dir $RUN/02_staged --pathbmask-dir $RUN/03_pathbmask; }
[ -f $RUN/04_pathb/model.onnx ] && echo "skip 4 (done)" || { echo "=== stage 4 rotary hoist ==="; $PY $WS/scripts/rewrite_qwen3_pathb.py --src-dir $RUN/03_pathbmask --dst-dir $RUN/04_pathb; }
[ -f $RUN/05_pathb_ctx512/model.onnx ] && echo "skip 5 (done)" || { echo "=== stage 5 pin shapes ==="; $PY $WS/scripts/pin_shapes_qwen3_4b.py --src-dir $RUN/04_pathb --dst-dir $RUN/05_pathb_ctx512 --ctx 512 --seq-q 1 --num-kv-heads 8 --head-dim 128 --vocab-size 151936; }
echo STAGES_1_5_DONE
du -sh $RUN/0* 2>/dev/null
