"""SQ5 smoke test — verify the ctx-tier generalization in
npu_engine/qualcomm_qwen3_4b_oracle.py works WITHOUT touching the NPU.

Validates four things per tier in {512, 1024, 2048, 3072, 4096}:
  1. build_part_cfg(metadata, ar=1, ctx=N) finds the cl=N components
     in metadata.yaml.
  2. The same for ar=128 (AR128 prefill graphs).
  3. KVStore(num_layers=36, ctx_len=N, with_ar128_input=True) builds
     the expected master + AR128 buffer shapes.
  4. The mask helpers produce [1,1,1,N] / [1,1,128,N]-shaped arrays.

If any tier fails, the engine cannot run at that tier — do not
proceed to a real NPU smoke test until this passes.

Run:
    .venv/Scripts/python.exe last_side_quest/sq5_long_context_npu/smoke_metadata.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "npu_engine"))

import yaml  # noqa: E402

from qualcomm_qwen3_4b_oracle import (  # noqa: E402
    AR128_BATCH,
    BUNDLE_DIR,
    HEAD_DIM,
    NUM_KV_HEADS,
    NUM_LAYERS,
    KVStore,
    attention_mask_quantized,
    attention_mask_quantized_ar128,
    build_part_cfg,
    wrapper_path,
)

CTX_TIERS = (512, 1024, 2048, 3072, 4096)


def main() -> int:
    metadata = yaml.safe_load((BUNDLE_DIR / "metadata.yaml").read_text())
    components = metadata["components"]
    failures: list[str] = []

    print(f"=== SQ5 ctx-tier smoke test (no NPU) ===")
    print(f"bundle: {BUNDLE_DIR}")
    print(f"metadata components total: {len(components)}")
    print()

    for ctx in CTX_TIERS:
        print(f"--- cl={ctx} ---")

        # 1. build_part_cfg(ar=1)
        try:
            cfg_ar1 = build_part_cfg(metadata, ar=1, ctx=ctx)
        except KeyError as e:
            failures.append(f"cl={ctx} ar=1: KeyError {e}")
            print(f"  build_part_cfg(ar=1):    FAIL {e}")
            continue
        gn1 = cfg_ar1[2]["graph_name"]
        n_inputs_ar1 = len(cfg_ar1[2]["inputs"])
        print(f"  build_part_cfg(ar=1):    OK  graph_name={gn1}, part2 inputs={n_inputs_ar1}")

        # 2. build_part_cfg(ar=128)
        try:
            cfg_ar128 = build_part_cfg(metadata, ar=AR128_BATCH, ctx=ctx)
        except KeyError as e:
            failures.append(f"cl={ctx} ar=128: KeyError {e}")
            print(f"  build_part_cfg(ar=128):  FAIL {e}")
            continue
        gn128 = cfg_ar128[2]["graph_name"]
        print(f"  build_part_cfg(ar=128):  OK  graph_name={gn128}")

        # Confirm past_kv shape from metadata reflects the tier.
        past_key_in = next(
            io for io in cfg_ar1[2]["inputs"]
            if io["name"].startswith("past_key_") and io["name"].endswith("_in")
        )
        expected_past = ctx - 1
        actual_past = past_key_in["shape"][-1]
        ar1_shape_ok = actual_past == expected_past
        print(f"  AR1 past_key_*_in shape: {past_key_in['shape']}  "
              f"(expected last-dim {expected_past}: {'OK' if ar1_shape_ok else 'MISMATCH'})")
        if not ar1_shape_ok:
            failures.append(
                f"cl={ctx} ar=1: past_key shape last-dim {actual_past} != {expected_past}")

        # 3. KVStore allocation
        kv = KVStore(NUM_LAYERS, with_ar128_input=True, ctx_len=ctx)
        master_shape = kv.keys[0].shape
        ar128_shape = kv.keys_ar128_in[0].shape
        master_ok = master_shape == (NUM_KV_HEADS, 1, HEAD_DIM, ctx - 1)
        ar128_ok = ar128_shape == (NUM_KV_HEADS, 1, HEAD_DIM, ctx - AR128_BATCH)
        print(f"  KVStore master keys[0]:  {master_shape}  "
              f"(expect {(NUM_KV_HEADS, 1, HEAD_DIM, ctx - 1)}: {'OK' if master_ok else 'FAIL'})")
        print(f"  KVStore ar128 keys[0]:   {ar128_shape}  "
              f"(expect {(NUM_KV_HEADS, 1, HEAD_DIM, ctx - AR128_BATCH)}: {'OK' if ar128_ok else 'FAIL'})")
        if not master_ok:
            failures.append(f"cl={ctx} kv master shape {master_shape}")
        if not ar128_ok:
            failures.append(f"cl={ctx} kv ar128 shape {ar128_shape}")

        # 4. Mask helpers
        m1 = attention_mask_quantized(0, 0.001, -16384, ctx_len=ctx)
        m128 = attention_mask_quantized_ar128(0, 0.001, -16384, ctx_len=ctx)
        m1_ok = m1.shape == (1, 1, 1, ctx)
        m128_ok = m128.shape == (1, 1, AR128_BATCH, ctx)
        print(f"  mask AR1 shape:          {m1.shape}  "
              f"({'OK' if m1_ok else 'FAIL'})")
        print(f"  mask AR128 shape:        {m128.shape}  "
              f"({'OK' if m128_ok else 'FAIL'})")
        if not m1_ok:
            failures.append(f"cl={ctx} mask ar1 shape {m1.shape}")
        if not m128_ok:
            failures.append(f"cl={ctx} mask ar128 shape {m128.shape}")

        # 5. wrapper_path naming (ctx=512 keeps legacy filename)
        wp = wrapper_path(BUNDLE_DIR, 2, "_ar128", ctx)
        if ctx == 512:
            expect = BUNDLE_DIR / "oracle_part2_ar128.wrapper.onnx"
        else:
            expect = BUNDLE_DIR / f"oracle_part2_ar128_cl{ctx}.wrapper.onnx"
        wp_ok = wp == expect
        print(f"  wrapper_path:            {wp.name}  ({'OK' if wp_ok else 'FAIL'})")
        if not wp_ok:
            failures.append(f"cl={ctx} wrapper_path {wp} != {expect}")

        # Memory accounting: total bytes for 36 layers × 2 KV × master + ar128
        master_bytes = (
            NUM_LAYERS * 2  # keys + values
            * NUM_KV_HEADS * 1 * HEAD_DIM * (ctx - 1)
        )
        ar128_bytes = (
            NUM_LAYERS * 2
            * NUM_KV_HEADS * 1 * HEAD_DIM * (ctx - AR128_BATCH)
        )
        print(f"  KV memory budget:        master {master_bytes/1024/1024:.0f} MB"
              f" + ar128 {ar128_bytes/1024/1024:.0f} MB"
              f" = {(master_bytes + ar128_bytes)/1024/1024:.0f} MB")
        print()

    if failures:
        print(f"=== SMOKE FAILED — {len(failures)} issues ===")
        for f in failures:
            print(f"  - {f}")
        return 1

    print(f"=== SMOKE PASSED — all {len(CTX_TIERS)} ctx tiers OK ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
