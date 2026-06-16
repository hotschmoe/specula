#!/bin/bash
# Re-quantize 14B decoder parts WITH calibration -> genuine int8 weights.
# Runs INSIDE specula-qairt:2.45 (SSD mounted at /workspace). IO stays fp32
# (--preserve_io_datatype) so seams remain FLOAT_32 and part1/part10 bins are
# reused unchanged. Only the missing piece is added: --input_list <cal>.
#
#   docker run --rm -v /mnt/vm_8tb/specula-build:/workspace specula-qairt:2.45 \
#     bash /workspace/end-to-end/build_server/requant_14b.sh 2 9
set -u
Q=/workspace/qairt/Qualcomm/AIStack/QAIRT/2.45.40.260406
export PATH=$Q/bin/x86_64-linux-clang:$PATH
export LD_LIBRARY_PATH=$Q/lib/x86_64-linux-clang
export PYTHONPATH=$Q/lib/python
export QAIRT_TMP_DIR=/workspace/tmp
mkdir -p $QAIRT_TMP_DIR

RUN=/workspace/runs/qwen3_14b_w8a16
SPLIT=$RUN/06_split; DLC=$RUN/07b_dlc; QDLC=$RUN/08b_qdlc
BIN=$RUN/09b_bin_calib; CAL=$RUN/cal_raw; LOG=$RUN/qairt_logs_calib
CFG=/workspace/qnn_v81_box.json
mkdir -p $DLC $QDLC $BIN $LOG

A=${1:-2}; B=${2:-9}
for k in $(seq $A $B); do
  p=part$k
  echo "===== $p ====="
  if [ ! -f $CAL/$p/input_list.txt ]; then echo "$p NO_CALIB ($CAL/$p)"; continue; fi
  # 1. convert fp32 ONNX -> DLC (preserve fp32 IO; no quantization_overrides)
  qairt-converter --input_network $SPLIT/$p/model.onnx --output_path $DLC/$p.dlc \
      --preserve_io_datatype > $LOG/${p}_convert.log 2>&1
  if [ ! -f $DLC/$p.dlc ]; then echo "$p CONVERT_FAIL"; tail -4 $LOG/${p}_convert.log; continue; fi
  echo "  converted ($(du -h $DLC/$p.dlc|cut -f1))"
  # 2. quantize WITH calibration -> int8 weights + int16 activations
  qairt-quantizer --input_dlc $DLC/$p.dlc --output_dlc $QDLC/$p.dlc \
      --input_list $CAL/$p/input_list.txt \
      --weights_bitwidth 8 --act_bitwidth 16 --bias_bitwidth 8 \
      --use_per_channel_quantization > $LOG/${p}_quant.log 2>&1
  if [ ! -f $QDLC/$p.dlc ]; then echo "$p QUANT_FAIL"; tail -8 $LOG/${p}_quant.log; continue; fi
  echo "  quantized ($(du -h $QDLC/$p.dlc|cut -f1))"
  # 3. context-bin
  qnn-context-binary-generator --backend $Q/lib/x86_64-linux-clang/libQnnHtp.so \
      --dlc_path $QDLC/$p.dlc --binary_file $p --output_dir $BIN \
      --config_file $CFG > $LOG/${p}_ctxbin.log 2>&1
  if [ ! -f $BIN/$p.bin ]; then echo "$p CTXBIN_FAIL"; tail -8 $LOG/${p}_ctxbin.log; continue; fi
  echo "$p OK ($(du -h $BIN/$p.bin|cut -f1))"
done
echo "REQUANT_DONE"
ls -la $BIN/*.bin 2>/dev/null | awk '{print "  ", $5, $9}'
