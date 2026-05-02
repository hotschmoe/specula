"""End-to-end Qwen → HTP NPU bundle.

ONE entry point. Takes a HF model id (or local path), runs the full
chain to a deployable .bin bundle on Snapdragon X2 Elite (HTP v75).

Pipeline:

    1. optimum-cli export onnx --task text-generation-with-past
    2. scripts/rewrite_qwen3_htp.py --mode stage
    3. scripts/rewrite_qwen3_htp.py --mode fold-pathbmask
    4. scripts/rewrite_qwen3_pathb.py        (rotary hoist)
    5. scripts/pin_shapes_qwen3_4b.py        (pin AR=1, ctx=N)
    6. AIMET aimet_onnx PTQ + SEQ_MSE + AdaScale (+ optional V/O pin)
    7. qairt-converter ONNX+encodings → DLC
    8. qnn-context-binary-generator DLC → HTP context .bin (v75)
    9. Bundle .bin + tokenizer + metadata, tar

Each stage is **idempotent** — drops a `done.json` marker and skips on
re-invocation. Use `--force-stage <n>` to re-run a specific stage and
the ones after it.

Defaults are MAX-QUALITY (full Qualcomm recipe):
    --num-cal-samples 128
    --use-seq-mse
    --use-ada-scale
    --ada-scale-iters 1500

That's a ~2 hr run for Qwen3-0.6B on A40 at $0.44/hr (~$0.90), and
~6-9 hr for Qwen3-4B (~$3-4). Worth the cost; we don't ship sub-0.95
cos artifacts.

Usage:

    PY=/workspace/venvs/aimet-2.26-cu121-py310/bin/python
    NVLIBS=$(find /workspace/venvs/aimet-2.26-cu121-py310/lib/python3.10/site-packages/nvidia -name lib -type d | tr '\\n' ':')
    LD_LIBRARY_PATH=$NVLIBS \\
    $PY end-to-end/quantize_to_npu.py \\
        --model-id Qwen/Qwen3-0.6B \\
        --model-path /workspace/models/Qwen3-0.6B \\
        --workdir /workspace/runs/qwen3_0p6b_w8a16 \\
        --precision w8a16 \\
        --ctx 512

For w4a16 with V/O collapse mitigation:

    --precision w4a16 --vo-pin-w8

Resume an aborted run (skips completed stages by reading done.json):

    (re-run the exact same command)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure local lib/ is importable regardless of cwd.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from lib import stages, aimet, qairt, bundle  # noqa: E402
from lib.model_config import load_model_info, summary_str  # noqa: E402


REPO_ROOT = HERE.parent
DEFAULT_QAIRT_ROOT = Path("/workspace/sdks/qairt-2.45.40.260406")
DEFAULT_VENV = Path("/workspace/venvs/aimet-2.26-cu121-py310")
QNN_CONFIG = HERE / "configs" / "qnn_v75_config.json"
QNN_INNER = HERE / "configs" / "qnn_v75_inner.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", required=True,
                   help="HF model id (e.g. Qwen/Qwen3-0.6B). Stamped into metadata.")
    p.add_argument("--model-path", type=Path, default=None,
                   help="Local model dir (HF safetensors). Defaults to /workspace/models/<basename>.")
    p.add_argument("--model-family", type=str, default=None,
                   help="Override family resolution. One of {qwen3, qwen2, qwen2_5, llama, ...}. "
                        "Default: inferred from config.json architectures + model-id.")
    p.add_argument("--workdir", type=Path, required=True,
                   help="Per-run workspace; all stage outputs land under here.")
    p.add_argument("--precision", choices=("w4a16", "w8a16"), default="w8a16")
    p.add_argument("--ctx", type=int, default=512,
                   help="Pinned attention window. The standard 0.6B/4B compile target is 512.")

    # Quality knobs (default = max-quality Qualcomm recipe).
    p.add_argument("--num-cal-samples", type=int, default=128)
    p.add_argument("--quant-scheme",
                   choices=("min_max", "post_training_tf_enhanced", "post_training_percentile"),
                   default="post_training_tf_enhanced")
    p.add_argument("--use-seq-mse", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seq-mse-candidates", type=int, default=20)
    p.add_argument("--use-ada-scale", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ada-scale-iters", type=int, default=1500)
    p.add_argument("--vo-pin-w8", action=argparse.BooleanOptionalAction, default=None,
                   help="Pin V/O proj weight bw to 8 (mitigates W4A16 V/O collapse). "
                        "Default: on for w4a16, off for w8a16.")

    # Infra knobs.
    p.add_argument("--qairt-root", type=Path, default=DEFAULT_QAIRT_ROOT)
    p.add_argument("--venv", type=Path, default=DEFAULT_VENV)
    p.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--qnn-config", type=Path, default=QNN_CONFIG)

    p.add_argument("--force-stage", type=int, default=None,
                   help="Re-run from this stage onward (1-9). E.g. --force-stage 6 "
                        "re-runs aimet + qairt + qnn + bundle.")

    p.add_argument("--bundle-name", type=str, default=None,
                   help="Override the bundle dir name. Default: <model-stem>-<prec>-pathb-ctx<ctx>-x2e.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Resolve defaults.
    model_basename = args.model_id.split("/")[-1]
    if args.model_path is None:
        args.model_path = Path("/workspace/models") / model_basename
    if not args.model_path.exists():
        print(f"FATAL: model_path does not exist: {args.model_path}", file=sys.stderr)
        return 2
    if args.vo_pin_w8 is None:
        args.vo_pin_w8 = (args.precision == "w4a16")
    if args.bundle_name is None:
        # Lower-case the model basename, rewrite "-" / "." to make it
        # filesystem-friendly.
        m = model_basename.lower().replace(".", "p")
        args.bundle_name = f"{m}-{args.precision}-pathb-ctx{args.ctx}-x2e"

    args.workdir.mkdir(parents=True, exist_ok=True)
    venv_python = args.venv / "bin" / "python"
    if not venv_python.exists():
        print(f"FATAL: venv python missing: {venv_python}", file=sys.stderr)
        return 2

    # Resolve model attributes from config.json (single source of truth
    # for layer count, head dim, rope_theta, family, etc).
    model_info = load_model_info(
        model_id=args.model_id, model_path=args.model_path,
        family_override=args.model_family, precision=args.precision,
    )
    if not model_info.family.pathb_supported:
        print(f"FATAL: family {model_info.family.name!r} is not flagged "
              f"pathb_supported. The pathb scripts (rewrite_qwen3_*) are "
              f"Qwen3-specific. Either extend pathb to this family or "
              f"pick a different model.", file=sys.stderr)
        return 2
    print("\n[model-info]")
    print(summary_str(model_info))

    # Stage paths.
    s1_dir = args.workdir / "01_optimum"
    s6_dir = args.workdir / f"06_aimet_{args.precision}"
    s7_dir = args.workdir / f"07_dlc_{args.precision}"
    s8_dir = args.workdir / f"08_bin_{args.precision}"
    s9_dir = args.workdir / f"09_bundle_{args.precision}"

    overall: dict = {
        "argv": sys.argv,
        "args": {k: str(v) for k, v in vars(args).items()},
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stages": {},
    }
    overall_path = args.workdir / "run_manifest.json"

    def _save():
        with open(overall_path, "w") as f:
            json.dump(overall, f, indent=2, default=str)

    force_at = args.force_stage or 99
    t_overall = time.time()

    # ---- 1. optimum export ----
    print(f"\n========== STAGE 1/9 — optimum export ==========")
    info = stages.stage_optimum_export(
        model_path=args.model_path, out_dir=s1_dir,
        force=(force_at <= 1), venv_python=venv_python,
    )
    overall["stages"]["1_optimum"] = info
    _save()

    # ---- 2-5. pathb chain ----
    print(f"\n========== STAGE 2-5/9 — pathb chain ==========")
    pinned_dir, pathb_info = stages.stage_pathb_chain(
        optimum_dir=s1_dir, work_root=args.workdir,
        model_stem=model_basename.lower(), ctx=args.ctx,
        force=(force_at <= 2), venv_python=venv_python,
        model_info=model_info,
    )
    overall["stages"]["2_5_pathb"] = pathb_info
    _save()

    # ---- 6. AIMET ----
    print(f"\n========== STAGE 6/9 — AIMET PTQ + SEQ_MSE + AdaScale ==========")
    aimet_done_marker = s6_dir / "done.json"
    export_prefix = f"{model_basename.lower().replace('.', 'p')}_pathb_{args.precision}"
    if force_at <= 6 or not aimet_done_marker.exists():
        s6_dir.mkdir(parents=True, exist_ok=True)
        log_path = s6_dir / "aimet.log"
        log_path.write_text("")
        info = aimet.run_aimet(
            src_dir=pinned_dir, tokenizer_path=args.model_path,
            output_dir=s6_dir, precision=args.precision, ctx=args.ctx,
            num_cal_samples=args.num_cal_samples,
            use_seq_mse=args.use_seq_mse,
            seq_mse_candidates=args.seq_mse_candidates,
            use_ada_scale=args.use_ada_scale,
            ada_scale_iters=args.ada_scale_iters,
            use_vo_pin_w8=args.vo_pin_w8,
            quant_scheme=args.quant_scheme,
            cuda=args.cuda, log_path=log_path,
            export_prefix=export_prefix,
            model_info=model_info,
        )
        with open(aimet_done_marker, "w") as f:
            json.dump(info, f, indent=2, default=str)
        overall["stages"]["6_aimet"] = info
    else:
        overall["stages"]["6_aimet"] = json.loads(aimet_done_marker.read_text())
        print(f"[stage 6] skip (done): {s6_dir}")
    _save()

    aimet_onnx = s6_dir / f"{export_prefix}.onnx"
    aimet_enc = s6_dir / f"{export_prefix}.encodings"

    # ---- 7. qairt-converter ----
    print(f"\n========== STAGE 7/9 — qairt-converter ==========")
    s7_dir.mkdir(parents=True, exist_ok=True)
    dlc_path = s7_dir / f"{export_prefix}.dlc"
    qairt_done = s7_dir / "done.json"
    if force_at <= 7 or not qairt_done.exists():
        info = qairt.run_qairt_converter(
            onnx_path=aimet_onnx, encodings_path=aimet_enc, dlc_path=dlc_path,
            qairt_root=args.qairt_root, venv_root=args.venv,
            log_path=s7_dir / "qairt_converter.log",
        )
        with open(qairt_done, "w") as f:
            json.dump(info, f, indent=2, default=str)
        overall["stages"]["7_qairt"] = info
    else:
        overall["stages"]["7_qairt"] = json.loads(qairt_done.read_text())
        print(f"[stage 7] skip (done): {s7_dir}")
    _save()

    # ---- 8. qnn-context-binary-generator ----
    print(f"\n========== STAGE 8/9 — qnn-context-binary-generator ==========")
    s8_dir.mkdir(parents=True, exist_ok=True)
    qnn_done = s8_dir / "done.json"
    bin_path = s8_dir / f"{export_prefix}.bin"
    if force_at <= 8 or not qnn_done.exists():
        info = qairt.run_qnn_context_binary_generator(
            dlc_path=dlc_path, output_dir=s8_dir,
            bin_prefix=export_prefix,
            qairt_root=args.qairt_root, venv_root=args.venv,
            config_path=args.qnn_config,
            log_path=s8_dir / "qnn_context.log",
        )
        # inspect for arch verification
        bin_info_path = s8_dir / "bin_info.json"
        bi = qairt.inspect_bin(
            bin_path=bin_path, qairt_root=args.qairt_root, venv_root=args.venv,
            json_out=bin_info_path,
        )
        # Extract dspArch for the manifest.
        try:
            arch = bi["graphs"][0]["graphBlobInfo"]["graphInfo"]["graphConfig"]["customConfig"]["dspArch"]
        except Exception:
            arch = None
            for grph in bi.get("graphs", []):
                # walk for dspArch
                import re
                txt = json.dumps(grph)
                m = re.search(r'"dspArch"\s*:\s*(\d+)', txt)
                if m:
                    arch = int(m.group(1)); break
        info["dsp_arch"] = arch
        info["bin_info_path"] = str(bin_info_path)
        with open(qnn_done, "w") as f:
            json.dump(info, f, indent=2, default=str)
        overall["stages"]["8_qnn_bin"] = info
    else:
        overall["stages"]["8_qnn_bin"] = json.loads(qnn_done.read_text())
        print(f"[stage 8] skip (done): {s8_dir}")
    _save()

    # ---- 9. bundle ----
    print(f"\n========== STAGE 9/9 — bundle assembly ==========")
    s9_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir = s9_dir / args.bundle_name
    tar_out = s9_dir / f"{args.bundle_name}.tar"
    bundle_done = s9_dir / "done.json"
    if force_at <= 9 or not bundle_done.exists():
        bin_info_path = s8_dir / "bin_info.json"
        info = bundle.assemble_bundle(
            bin_path=bin_path,
            encodings_path=aimet_enc,
            bin_info_path=bin_info_path,
            tokenizer_dir=args.model_path,
            bundle_dir=bundle_dir,
            bundle_name=args.bundle_name,
            metadata={
                "model_id": args.model_id,
                "precision": args.precision,
                "ctx": args.ctx,
                "ar": 1,
                "target_arch": "HTP v75 (Snapdragon X2 Elite NPU)",
                "pipeline": "optimum-export → pathb (rewrite_qwen3_htp + rewrite_qwen3_pathb + pin_shapes) → aimet_onnx (PTQ + SEQ_MSE + AdaScale" + (" + V/O w8 pin" if args.vo_pin_w8 else "") + ") → qairt-converter → qnn-context-binary-generator (HTP v75)",
                "recipe": {
                    "num_cal_samples": args.num_cal_samples,
                    "quant_scheme": args.quant_scheme,
                    "use_seq_mse": args.use_seq_mse,
                    "seq_mse_candidates": args.seq_mse_candidates,
                    "use_ada_scale": args.use_ada_scale,
                    "ada_scale_iters": args.ada_scale_iters,
                    "vo_pin_w8": args.vo_pin_w8,
                },
                "aimet_probe": overall["stages"]["6_aimet"].get("stages", {}).get("9_probe"),
                "started_at": overall["started_at"],
                "wall_total_s": time.time() - t_overall,
            },
            tar_out=tar_out,
        )
        with open(bundle_done, "w") as f:
            json.dump(info, f, indent=2, default=str)
        overall["stages"]["9_bundle"] = info
    else:
        overall["stages"]["9_bundle"] = json.loads(bundle_done.read_text())
        print(f"[stage 9] skip (done): {s9_dir}")
    overall["wall_total_s"] = time.time() - t_overall
    _save()

    print(f"\n========== DONE in {time.time() - t_overall:.0f}s ==========")
    print(f"  bundle dir : {bundle_dir}")
    print(f"  tar        : {tar_out}  ({tar_out.stat().st_size / 1e9:.2f} GB)")
    probe = overall["stages"]["6_aimet"].get("stages", {}).get("9_probe", {})
    if "cos_fp_q" in probe:
        print(f"  aimet probe: cos(fp,q) = {probe['cos_fp_q']:.4f}, "
              f"argmax_match = {probe.get('argmax_match')}")
    print(f"  manifest   : {overall_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
