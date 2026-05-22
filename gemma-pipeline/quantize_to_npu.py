"""End-to-end Gemma 4 -> Hexagon HTP NPU bundle (orchestrator).

Sibling of `end-to-end/quantize_to_npu.py` (the Qwen3 pipeline). Same
contract — one entry point, idempotent stages, max-quality defaults —
but retargeted at Gemma 4, whose decoder needs its own graph-surgery
scripts (see ARCHITECTURE_NOTES.md and scripts/README.md).

Stages:

    1. optimum-cli export onnx               (text decoder only)
    2. rewrite_gemma4_htp.py --mode stage          [NOT YET BUILT]
    3. rewrite_gemma4_htp.py --mode fold-pathbmask [NOT YET BUILT]
    4. rewrite_gemma4_pathb.py  (dual+partial rotary hoist, PLE) [NOT YET BUILT]
    5. pin_shapes_gemma4.py     (pin AR=1, ctx=N)               [NOT YET BUILT]
    6. AIMET aimet_onnx PTQ + SEQ_MSE (+ AdaScale)   CUDA-only
    7. qairt-converter ONNX+encodings -> DLC
    8. qnn-context-binary-generator DLC -> HTP .bin (v75)
    9. bundle .bin + tokenizer + metadata, tar

This orchestrator is INTENTIONALLY honest: stages 2-6 depend on the
Gemma-specific rewrite scripts and the CUDA AIMET environment. Until
those exist this script runs stage 1, prints the resolved model plan,
and stops with an explicit, actionable error at the first unbuilt
stage. It does not fake success.

Why not just port the Qwen3 pipeline? Because the Qwen3 rewrites
(rewrite_qwen3_*, pin_shapes_qwen3_4b) hard-assume single-theta RoPE,
no per-layer embeddings, no KV sharing, and uniform full attention.
Gemma 4 violates all four. See ARCHITECTURE_NOTES.md.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from lib.model_config import load_model_info, summary_str  # noqa: E402

REPO_ROOT = HERE.parent
QNN_V75_CONFIG = HERE / "configs" / "qnn_v75_config.json"

# The four Gemma-specific scripts this pipeline needs. Spec lives in
# scripts/README.md. Until a file exists, the stage that needs it stops.
GEMMA_REWRITES = {
    2: HERE / "scripts" / "rewrite_gemma4_htp.py",
    3: HERE / "scripts" / "rewrite_gemma4_htp.py",
    4: HERE / "scripts" / "rewrite_gemma4_pathb.py",
    5: HERE / "scripts" / "pin_shapes_gemma4.py",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", default="google/gemma-4-E2B-it",
                   help="HF model id. Default: the smallest Gemma 4.")
    p.add_argument("--model-path", type=Path, default=None,
                   help="Local HF dir. Default: /workspace/models/<basename>.")
    p.add_argument("--model-family", default=None,
                   help="Override family resolution (gemma4 | gemma3).")
    p.add_argument("--workdir", type=Path, required=True,
                   help="Per-run workspace; all stage outputs land here.")
    p.add_argument("--precision", choices=("w4a16", "w8a16"), default="w4a16")
    p.add_argument("--ctx", type=int, default=32768,
                   help="Pinned attention window. Gemma 4 SWA makes 32768 a "
                        "realistic target (vs the Qwen3 4096 ceiling).")
    p.add_argument("--num-cal-samples", type=int, default=128)
    p.add_argument("--use-seq-mse", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use-ada-scale", action=argparse.BooleanOptionalAction, default=False,
                   help="Off by default for Gemma: the AdaScale ReduceMean-v18 "
                        "crash (end-to-end/README.md) is more likely given "
                        "Gemma's extra norms. Enable once the v18 converter "
                        "is patched.")
    p.add_argument("--vo-pin-w8", action=argparse.BooleanOptionalAction, default=None,
                   help="Pin V/O proj weights to w8 (w4a16 V/O-collapse "
                        "mitigation). Default: on for w4a16.")
    p.add_argument("--uint8-kv", action=argparse.BooleanOptionalAction, default=True,
                   help="Quantize KV-cache I/O to 8-bit. Default ON for Gemma "
                        "long-ctx — quarters the global-layer KV slice.")
    p.add_argument("--venv", type=Path, default=Path("/workspace/venvs/aimet-2.26-cu121-py310"),
                   help="AIMET venv root (CUDA). Only stage 6 needs it.")
    p.add_argument("--qairt-root", type=Path,
                   default=Path("/workspace/sdks/qairt-2.45.40.260406"))
    p.add_argument("--qnn-config", type=Path, default=QNN_V75_CONFIG)
    p.add_argument("--force-stage", type=int, default=None)
    p.add_argument("--stop-after-stage", type=int, default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="Resolve model info, print the plan, exit. No work.")
    return p.parse_args()


def _fatal(msg: str) -> int:
    print(f"\nFATAL: {msg}", file=sys.stderr)
    return 2


def _unbuilt_stage_stop(stage: int) -> int:
    script = GEMMA_REWRITES[stage]
    print(f"\n{'=' * 64}")
    print(f"STOP — stage {stage} needs a Gemma-specific script that does "
          f"not exist yet:")
    print(f"    {script}")
    print(f"\nThis is expected. The Qwen3 rewrites in end-to-end/ do NOT "
          f"transfer\n(dual/partial RoPE, Per-Layer Embeddings, KV sharing, "
          f"sliding-window\nattention — see ARCHITECTURE_NOTES.md).")
    print(f"\nBuild the scripts per gemma-pipeline/scripts/README.md, then "
          f"re-run.\nStage 1 output is preserved; re-running resumes from "
          f"stage {stage}.")
    print(f"{'=' * 64}")
    return 3


def main() -> int:
    args = parse_args()

    basename = args.model_id.split("/")[-1]
    if args.model_path is None:
        args.model_path = Path("/workspace/models") / basename
    if args.vo_pin_w8 is None:
        args.vo_pin_w8 = (args.precision == "w4a16")

    if not args.model_path.exists():
        return _fatal(
            f"model_path does not exist: {args.model_path}\n"
            f"  Download first, e.g.:\n"
            f"    huggingface-cli download {args.model_id} "
            f"--local-dir {args.model_path}")

    cfg_json = args.model_path / "config.json"
    if not cfg_json.exists():
        return _fatal(f"no config.json under {args.model_path}")

    model_info = load_model_info(
        model_id=args.model_id, model_path=args.model_path,
        family_override=args.model_family, precision=args.precision,
    )
    print("\n[model-info]")
    print(summary_str(model_info))

    print(f"\n[plan]  precision={args.precision}  ctx={args.ctx}  "
          f"uint8_kv={args.uint8_kv}  vo_pin_w8={args.vo_pin_w8}")
    print(f"        workdir={args.workdir}")
    print(f"        global layers (O(ctx) KV): {model_info.num_global_layers}  |  "
          f"sliding layers (512-cap KV): {model_info.num_sliding_layers}")
    if model_info.hidden_size_per_layer_input:
        print(f"        NOTE: Per-Layer Embeddings present "
              f"(width {model_info.hidden_size_per_layer_input}) — "
              f"stage 4 must preserve the residual injection.")
    if model_info.rope_partial_rotary_factor_full != 1.0:
        print(f"        NOTE: global RoPE is partial "
              f"(factor {model_info.rope_partial_rotary_factor_full}) — "
              f"stage 4 hoist must split rotary / pass-through dims.")
    if model_info.enable_moe_block:
        print(f"        NOTE: MoE block enabled — routing surgery required "
              f"(not yet scoped).")

    if args.dry_run:
        print("\n[dry-run] resolved OK; exiting before any work.")
        return 0

    args.workdir.mkdir(parents=True, exist_ok=True)
    force_at = args.force_stage or 99
    t0 = time.time()

    # ---- Stage 1: optimum export (runs anywhere with torch + optimum) ----
    print(f"\n========== STAGE 1/9 — optimum export ==========")
    s1_dir = args.workdir / "01_optimum"
    s1_done = s1_dir / "done.json"
    if force_at <= 1 or not s1_done.exists():
        optimum = shutil.which("optimum-cli")
        if optimum is None:
            return _fatal(
                "optimum-cli not on PATH. Stage 1 needs `optimum` + `torch`.\n"
                "  pip install optimum[exporters] transformers torch\n"
                "Gemma 4 is a multimodal wrapper — the export must isolate "
                "the\ntext decoder. See ARCHITECTURE_NOTES.md stage-1 row.")
        s1_dir.mkdir(parents=True, exist_ok=True)
        cmd = [optimum, "export", "onnx",
               "--model", str(args.model_path),
               "--task", "text-generation-with-past",
               str(s1_dir)]
        print("  $ " + " ".join(cmd))
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            return _fatal(
                f"optimum export failed (rc={rc}). For Gemma 4 the likely "
                f"cause is the\nmultimodal wrapper — the text decoder may "
                f"need isolating first.\nSee ARCHITECTURE_NOTES.md "
                f"'Open questions'.")
        s1_done.write_text(json.dumps(
            {"stage": 1, "out": str(s1_dir),
             "at": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
    else:
        print(f"  [skip] done: {s1_dir}")

    if args.stop_after_stage == 1:
        print(f"\n[stopped after stage 1 in {time.time() - t0:.0f}s]")
        return 0

    # ---- Stages 2-5: Gemma-specific rewrites ----
    for stage in (2, 3, 4, 5):
        print(f"\n========== STAGE {stage}/9 — gemma rewrite ==========")
        if not GEMMA_REWRITES[stage].exists():
            return _unbuilt_stage_stop(stage)
        # When the scripts land, invoke them here. Left as an explicit
        # stop so the orchestrator never silently no-ops a missing stage.
        return _fatal(
            f"stage {stage} script exists ({GEMMA_REWRITES[stage].name}) but "
            f"its\ninvocation is not wired into this orchestrator yet — wire "
            f"it once\nthe script's CLI is finalized (mirror "
            f"end-to-end/lib/stages.py).")

    # ---- Stages 6-9: AIMET + QAIRT + bundle ----
    # These reuse end-to-end/lib (aimet.py, qairt.py, bundle.py) — they are
    # architecture-agnostic at the ONNX level. Wired once stage 5 produces
    # a pinned pathb ONNX. Stage 6 requires the CUDA AIMET venv.
    return _fatal("stages 6-9 are unreachable until stages 2-5 are built.")


if __name__ == "__main__":
    sys.exit(main())
