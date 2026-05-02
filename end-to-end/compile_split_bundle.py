"""Post-AIMET: split → per-part qairt-converter → per-part qnn-context-
binary-generator → assemble Qualcomm-genie-compatible multi-part bundle.

Use this when stage 8 of `quantize_to_npu.py` (single-bin compile) hits
the HTP serializer's 3.67 GB ceiling — typical for 4B+ models. We re-use
the AIMET-emitted ONNX + encodings from stage 6 and produce N separate
.bin files instead, packaged as a Qualcomm-genie bundle.

Usage:
    PY=/workspace/venvs/aimet-2.26-cu121-py310/bin/python
    NVLIBS=$(find /workspace/venvs/aimet-2.26-cu121-py310/lib/python3.10/site-packages/nvidia -name lib -type d | tr '\\n' ':')
    LD_LIBRARY_PATH=$NVLIBS \\
    $PY end-to-end/compile_split_bundle.py \\
        --workdir /workspace/runs/qwen3_4b_w4a16 \\
        --model-id Qwen/Qwen3-4B \\
        --precision w4a16 --ctx 512 --num-parts 4 --dsp-arch v81

Looks for:
    <workdir>/06_aimet_<precision>/<prefix>.onnx + .encodings + .data
Produces:
    <workdir>/06b_split_<precision>/part{1..N}/  (sub-onnx + sub-encodings)
    <workdir>/07b_dlc_<precision>/part{1..N}/    (per-part DLCs)
    <workdir>/08b_bin_<precision>/part{1..N}/    (per-part .bins)
    <workdir>/09b_bundle_<precision>/<bundle_name>/  + .tar
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from lib import qairt, bundle  # noqa: E402
from lib.split import split_aimet_output  # noqa: E402
from lib.model_config import load_model_info  # noqa: E402


REPO_ROOT = HERE.parent
DEFAULT_QAIRT_ROOT = Path("/workspace/sdks/qairt-2.45.40.260406")
DEFAULT_VENV = Path("/workspace/venvs/aimet-2.26-cu121-py310")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workdir", type=Path, required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--model-path", type=Path, default=None)
    p.add_argument("--model-family", type=str, default=None)
    p.add_argument("--precision", choices=("w4a16", "w8a16"), required=True)
    p.add_argument("--ctx", type=int, default=512)
    p.add_argument("--num-parts", type=int, default=4)
    p.add_argument("--dsp-arch", default="v81",
                   help="HTP arch: v75 (Snapdragon X Elite, 8 Gen 3) or v81 "
                        "(Snapdragon X2 Elite, 8 Elite Gen 5). Qualcomm's "
                        "shipping Qwen3-4B X2E bundle uses v81.")
    p.add_argument("--soc-model", type=int, default=88,
                   help="SoC model id. 88 = X2 Elite per Qualcomm's bundle.")
    p.add_argument("--qairt-root", type=Path, default=DEFAULT_QAIRT_ROOT)
    p.add_argument("--venv", type=Path, default=DEFAULT_VENV)
    p.add_argument("--bundle-name", type=str, default=None)
    p.add_argument("--qnn-config", type=Path, default=None,
                   help="Path to outer qnn config JSON; default picks v75/v81 "
                        "based on --dsp-arch.")
    p.add_argument("--force", action="store_true",
                   help="Re-run all post-AIMET stages from scratch.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.model_path is None:
        args.model_path = Path("/workspace/models") / args.model_id.split("/")[-1]
    if not args.model_path.exists():
        print(f"FATAL: model_path missing: {args.model_path}", file=sys.stderr)
        return 2
    if args.qnn_config is None:
        args.qnn_config = HERE / "configs" / f"qnn_{args.dsp_arch}_config.json"
    if not args.qnn_config.exists():
        print(f"FATAL: qnn config missing: {args.qnn_config}", file=sys.stderr)
        return 2

    model_basename = args.model_id.split("/")[-1]
    if args.bundle_name is None:
        m = model_basename.lower().replace(".", "p")
        args.bundle_name = f"{m}_{args.precision}_pathb_ctx{args.ctx}_x2e_{args.dsp_arch}"

    venv_python = args.venv / "bin" / "python"
    if not venv_python.exists():
        print(f"FATAL: venv python missing: {venv_python}", file=sys.stderr)
        return 2

    model_info = load_model_info(
        model_id=args.model_id, model_path=args.model_path,
        family_override=args.model_family, precision=args.precision,
    )

    aimet_dir = args.workdir / f"06_aimet_{args.precision}"
    export_prefix = f"{model_basename.lower().replace('.', 'p')}_pathb_{args.precision}"
    if not (aimet_dir / f"{export_prefix}.onnx").exists():
        print(f"FATAL: AIMET output not found at {aimet_dir}/{export_prefix}.onnx",
              file=sys.stderr)
        print(f"  (run quantize_to_npu.py through stage 6 first)", file=sys.stderr)
        return 2

    split_dir = args.workdir / f"06b_split_{args.precision}"
    dlc_root = args.workdir / f"07b_dlc_{args.precision}"
    bin_root = args.workdir / f"08b_bin_{args.precision}"
    bundle_root = args.workdir / f"09b_bundle_{args.precision}"

    # --------- 6b. SPLIT ---------
    split_done = split_dir / "done.json"
    if args.force or not split_done.exists():
        print(f"\n========== STAGE 6b — split AIMET output → {args.num_parts} parts ==========")
        split_dir.mkdir(parents=True, exist_ok=True)
        info = split_aimet_output(
            aimet_dir=aimet_dir, export_prefix=export_prefix,
            model_info=model_info, ctx=args.ctx,
            out_root=split_dir, num_parts=args.num_parts,
        )
        split_done.write_text(json.dumps(info, indent=2, default=str))
    else:
        print(f"[stage 6b] skip (done): {split_dir}")
        info = json.loads(split_done.read_text())

    # --------- 7b. per-part qairt-converter ---------
    print(f"\n========== STAGE 7b — qairt-converter × {args.num_parts} ==========")
    dlc_root.mkdir(parents=True, exist_ok=True)
    dlc_paths: list[Path] = []
    for pi in range(1, args.num_parts + 1):
        part_name = f"part{pi}"
        part_split = split_dir / part_name
        part_dlc_dir = dlc_root / part_name
        part_dlc_dir.mkdir(parents=True, exist_ok=True)
        dlc_path = part_dlc_dir / f"{export_prefix}_{part_name}.dlc"
        done = part_dlc_dir / "done.json"
        if args.force or not done.exists():
            t0 = time.time()
            qairt.run_qairt_converter(
                onnx_path=part_split / "model.onnx",
                encodings_path=part_split / "model.encodings",
                dlc_path=dlc_path,
                qairt_root=args.qairt_root, venv_root=args.venv,
                log_path=part_dlc_dir / "qairt.log",
            )
            done.write_text(json.dumps({
                "wall_s": time.time() - t0,
                "dlc_path": str(dlc_path),
                "dlc_size": dlc_path.stat().st_size,
            }, indent=2))
        else:
            print(f"[stage 7b/{part_name}] skip (done): {part_dlc_dir}")
        dlc_paths.append(dlc_path)

    # --------- 8b. per-part qnn-context-binary-generator ---------
    print(f"\n========== STAGE 8b — qnn-context-binary-generator × {args.num_parts} ==========")
    bin_root.mkdir(parents=True, exist_ok=True)
    bin_paths: list[Path] = []
    bin_info_paths: list[Path] = []
    for pi in range(1, args.num_parts + 1):
        part_name = f"part{pi}"
        part_bin_dir = bin_root / part_name
        part_bin_dir.mkdir(parents=True, exist_ok=True)
        bin_prefix = f"{export_prefix}_{part_name}"
        bin_path = part_bin_dir / f"{bin_prefix}.bin"
        bin_info_path = part_bin_dir / "bin_info.json"
        done = part_bin_dir / "done.json"
        if args.force or not done.exists():
            t0 = time.time()
            qairt.run_qnn_context_binary_generator(
                dlc_path=dlc_paths[pi - 1], output_dir=part_bin_dir,
                bin_prefix=bin_prefix,
                qairt_root=args.qairt_root, venv_root=args.venv,
                config_path=args.qnn_config,
                log_path=part_bin_dir / "qnn_context.log",
            )
            try:
                qairt.inspect_bin(
                    bin_path=bin_path, qairt_root=args.qairt_root,
                    venv_root=args.venv, json_out=bin_info_path,
                )
            except Exception as e:
                print(f"  [stage 8b/{part_name}] inspect_bin failed (non-fatal): {e}")
            done.write_text(json.dumps({
                "wall_s": time.time() - t0,
                "bin_path": str(bin_path),
                "bin_size": bin_path.stat().st_size,
            }, indent=2))
        else:
            print(f"[stage 8b/{part_name}] skip (done): {part_bin_dir}")
        bin_paths.append(bin_path)
        bin_info_paths.append(bin_info_path)

    # --------- 9b. assemble genie bundle ---------
    print(f"\n========== STAGE 9b — assemble multi-part genie bundle ==========")
    bundle_root.mkdir(parents=True, exist_ok=True)
    bundle_dir = bundle_root / args.bundle_name
    tar_out = bundle_root / f"{args.bundle_name}.tar"
    encodings_paths = [split_dir / f"part{pi}" / "model.encodings"
                       for pi in range(1, args.num_parts + 1)]
    info = bundle.assemble_genie_bundle(
        bin_paths=bin_paths,
        bin_info_paths=bin_info_paths,
        encodings_paths=encodings_paths,
        tokenizer_dir=args.model_path,
        bundle_dir=bundle_dir, bundle_name=args.bundle_name,
        metadata={
            "model_id": args.model_id,
            "precision": args.precision,
            "ctx": args.ctx,
            "num_parts": args.num_parts,
            "pipeline": (
                "optimum-export → pathb (rewrite_qwen3_htp + rewrite_qwen3_pathb + pin_shapes) "
                "→ aimet_onnx (PTQ + SEQ_MSE + AdaScale + V/O w8 pin) "
                "→ split into N parts at residual-stream seams "
                "→ per-part qairt-converter → per-part qnn-context-binary-generator "
                "→ multi-part Qualcomm-genie bundle"
            ),
            "dsp_arch": args.dsp_arch, "soc_model": args.soc_model,
        },
        tar_out=tar_out,
        model_info=model_info, ctx=args.ctx,
        dsp_arch=args.dsp_arch, soc_model=args.soc_model,
    )
    print(f"\n========== DONE ==========")
    print(f"  bundle dir : {bundle_dir}")
    print(f"  tar        : {tar_out}  ({tar_out.stat().st_size / 1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
