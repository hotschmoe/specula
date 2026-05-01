#!/usr/bin/env bash
# SQ4 M1 — pathb ONNX + AIMET encodings → DLC → HTP context binary.
#
# Reads:
#   $OUT/qwen3_0p6b_pathb_w8a16.onnx     (aimet_onnx export, no QDQ in graph)
#   $OUT/qwen3_0p6b_pathb_w8a16.encodings (sibling .encodings file)
# Writes:
#   $OUT/qwen3_0p6b_pathb_w8a16.dlc
#   $OUT/qwen3_0p6b_pathb_w8a16.serialized.bin (HTP context binary)
#
# Run from inside the AIMET venv. Set OUT=<aimet output dir> first.
set -euo pipefail

: "${OUT:?set OUT to the AIMET output dir, e.g. /root/sq4_intermediates/m1_pathb_w8a16_smoke}"
: "${PREFIX:=qwen3_0p6b_pathb_w8a16}"

QAIRT_SDK_ROOT="${QAIRT_SDK_ROOT:-/workspace/sdks/qairt-2.45.40.260406}"
VENV="${VENV:-/workspace/venvs/aimet-2.26-cu121-py310}"

# Activate venv + QAIRT path/lib/python additions
source "$VENV/bin/activate"
export PATH="$QAIRT_SDK_ROOT/bin/x86_64-linux-clang:$VENV/bin:$PATH"
export LD_LIBRARY_PATH="$QAIRT_SDK_ROOT/lib/x86_64-linux-clang:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$QAIRT_SDK_ROOT/lib/python:${PYTHONPATH:-}"

ONNX="$OUT/$PREFIX.onnx"
ENC="$OUT/$PREFIX.encodings"
DLC="$OUT/$PREFIX.dlc"
BIN_PREFIX="$OUT/$PREFIX"

[[ -f "$ONNX" ]] || { echo "missing $ONNX"; exit 2; }
[[ -f "$ENC" ]]  || { echo "missing $ENC"; exit 2; }

echo "[A] qairt-converter: ONNX + encodings → DLC"
echo "    in  $ONNX"
echo "    enc $ENC"
echo "    out $DLC"
# Note on SoC targeting: qairt-converter 2.45's --target_soc_model only
# accepts a small allow-list (SM8845/SM8850/SM8850L per BackendInfo) that
# doesn't include the X2 Elite. We don't pin SoC at the converter — the
# DLC stays generic; HTP arch gets pinned at qnn-context-binary-generator
# via --config_file (qnn_v75_config.json sets dsp_arch=v75 to match X2E).
qairt-converter \
  --input_network "$ONNX" \
  --output_path   "$DLC" \
  --quantization_overrides "$ENC" \
  --preserve_io_datatype \
  2>&1 | tee "$OUT/qairt_converter.log"

echo
echo "[B] qnn-context-binary-generator: DLC → HTP .bin"
# X2 Elite NPU is HTP v75. The default libQnnHtp.so backend targets v68
# (the mobile-arch floor), which fails op-config validation on the
# AIMET-quant ops introduced at v73+ ("Value 68, expected >= 73"). The
# v75 config below pins the device arch via QNN_HTP_DEVICE_CONFIG_OPTION_ARCH.
QNN_CONFIG="${QNN_CONFIG:-/workspace/specula/last_side_quest/sq4_cloud_adventure/qnn_v75_config.json}"
qnn-context-binary-generator \
  --backend     "$QAIRT_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so" \
  --dlc_path    "$DLC" \
  --binary_file "$PREFIX" \
  --output_dir  "$OUT" \
  --config_file "$QNN_CONFIG" \
  2>&1 | tee "$OUT/qnn_context.log"

echo
echo "DONE. artifacts:"
ls -la "$OUT/"
