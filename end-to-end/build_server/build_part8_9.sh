#!/bin/bash
Q=/workspace/qairt/Qualcomm/AIStack/QAIRT/2.45.40.260406
export PATH=$Q/bin/x86_64-linux-clang:$PATH
export LD_LIBRARY_PATH=$Q/lib/x86_64-linux-clang
export PYTHONPATH=$Q/lib/python
export QAIRT_TMP_DIR=/workspace/tmp; mkdir -p /workspace/tmp
RUN=/workspace/runs/qwen3_14b_w8a16
DLC=$RUN/07_dlc; QDLC=$RUN/08_qdlc; BIN=$RUN/09_bin; LOG=$RUN/qairt_logs
CFG=/workspace/qnn_v81_box.json
for p in part8 part9; do
  echo "===== $p ====="
  python $Q/bin/x86_64-linux-clang/qairt-converter --input_network $RUN/06_split/$p/model.onnx --output_path $DLC/${p}.dlc --preserve_io_datatype > $LOG/${p}_convert.log 2>&1
  if [ ! -f $DLC/${p}.dlc ]; then echo "$p CONVERT_FAIL"; tail -4 $LOG/${p}_convert.log; continue; fi
  echo "  converted ($(du -h $DLC/${p}.dlc|cut -f1))"
  qairt-quantizer --input_dlc $DLC/${p}.dlc --output_dlc $QDLC/${p}.dlc --weights_bitwidth 8 --act_bitwidth 16 --bias_bitwidth 8 --use_per_channel_quantization > $LOG/${p}_quant.log 2>&1
  if [ ! -f $QDLC/${p}.dlc ]; then echo "$p QUANT_FAIL"; tail -4 $LOG/${p}_quant.log; continue; fi
  echo "  quantized ($(du -h $QDLC/${p}.dlc|cut -f1))"
  $Q/bin/x86_64-linux-clang/qnn-context-binary-generator --backend $Q/lib/x86_64-linux-clang/libQnnHtp.so --dlc_path $QDLC/${p}.dlc --binary_file ${p} --output_dir $BIN --config_file $CFG > $LOG/${p}_ctxbin.log 2>&1
  if [ ! -f $BIN/${p}.bin ]; then echo "$p CTXBIN_FAIL"; tail -4 $LOG/${p}_ctxbin.log; continue; fi
  echo "$p OK ($(du -h $BIN/${p}.bin|cut -f1))"
done
echo "PART89_DONE"
ls -la $BIN/*.bin | awk "{printf \"  %s %.2fGB\n\", \$NF, \$5/1e9}"
