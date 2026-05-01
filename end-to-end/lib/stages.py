"""Stage runner with idempotency. Each stage:
  - reads from the prior stage's output
  - writes to its own output dir
  - drops a `done.json` marker on success
A re-invocation skips a stage iff `done.json` exists AND the stage's
output files are non-empty (or `force=True` reruns).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _done(p: Path) -> bool:
    m = p / "done.json"
    return m.exists() and m.stat().st_size > 0


def _mark_done(p: Path, info: dict):
    with open(p / "done.json", "w") as f:
        json.dump(info, f, indent=2, default=str)


def _run(cmd: list[str], cwd: Path | None = None, env: dict | None = None,
         log_path: Path | None = None) -> int:
    print(f"[run] {' '.join(str(c) for c in cmd)}")
    if log_path:
        with open(log_path, "w") as logf:
            return subprocess.run(cmd, cwd=cwd, env=env, stdout=logf,
                                  stderr=subprocess.STDOUT).returncode
    return subprocess.run(cmd, cwd=cwd, env=env).returncode


def stage_optimum_export(
    *, model_path: Path, out_dir: Path, force: bool = False,
    venv_python: Path,
) -> dict:
    """Stage 1: HF model → optimum-cli ONNX (text-generation-with-past)."""
    if not force and _done(out_dir):
        print(f"[stage 1] skip (done): {out_dir}")
        return json.loads((out_dir / "done.json").read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "optimum_export.log"
    t0 = time.time()
    cmd = [
        str(venv_python.parent / "optimum-cli"),
        "export", "onnx",
        "--model", str(model_path),
        "--task", "text-generation-with-past",
        str(out_dir),
    ]
    rc = _run(cmd, log_path=log)
    onnx_path = out_dir / "model.onnx"
    if rc != 0 or not onnx_path.exists():
        raise RuntimeError(f"optimum export failed (rc={rc}); see {log}")
    info = {
        "stage": "1_optimum_export",
        "out_dir": str(out_dir),
        "wall_s": time.time() - t0,
        "model_onnx_size": onnx_path.stat().st_size,
        "data_size": sum(f.stat().st_size for f in out_dir.iterdir() if f.suffix in ("",".onnx_data") or "data" in f.name and f.is_file()),
    }
    _mark_done(out_dir, info)
    print(f"[stage 1] done {time.time() - t0:.1f}s")
    return info


def stage_pathb_chain(
    *, optimum_dir: Path, work_root: Path, model_stem: str,
    ctx: int, force: bool = False, venv_python: Path,
) -> tuple[Path, dict]:
    """Stages 2-5: rewrite_qwen3_htp stage + fold-pathbmask, then
    rewrite_qwen3_pathb (rotary hoist), then pin_shapes ctx=N.
    Returns (path-to-final-pinned-dir, info dict)."""
    staged_dir = work_root / "02_staged"
    pathbmask_dir = work_root / "03_pathbmask"
    pathb_dir = work_root / "04_pathb"
    pinned_dir = work_root / f"05_pathb_ctx{ctx}"

    info: dict = {"stages": {}}

    # 2. stage
    if force or not _done(staged_dir):
        staged_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        rc = _run([
            str(venv_python),
            str(SCRIPTS_DIR / "rewrite_qwen3_htp.py"),
            "--mode", "stage",
            "--optimum-dir", str(optimum_dir),
            "--staged-dir", str(staged_dir),
        ], log_path=staged_dir / "stage.log")
        if rc != 0:
            raise RuntimeError(f"rewrite_qwen3_htp stage failed (rc={rc})")
        d = {"stage": "2_rewrite_stage", "wall_s": time.time() - t0,
             "out_dir": str(staged_dir)}
        _mark_done(staged_dir, d)
        info["stages"]["2_stage"] = d
    else:
        info["stages"]["2_stage"] = json.loads((staged_dir / "done.json").read_text())
        print(f"[stage 2] skip (done): {staged_dir}")

    # 3. fold-pathbmask
    if force or not _done(pathbmask_dir):
        pathbmask_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        rc = _run([
            str(venv_python),
            str(SCRIPTS_DIR / "rewrite_qwen3_htp.py"),
            "--mode", "fold-pathbmask",
            "--optimum-dir", str(optimum_dir),
            "--staged-dir", str(staged_dir),
            "--pathbmask-dir", str(pathbmask_dir),
        ], log_path=pathbmask_dir / "stage.log")
        if rc != 0:
            raise RuntimeError(f"rewrite_qwen3_htp fold-pathbmask failed (rc={rc})")
        d = {"stage": "3_pathbmask", "wall_s": time.time() - t0,
             "out_dir": str(pathbmask_dir)}
        _mark_done(pathbmask_dir, d)
        info["stages"]["3_pathbmask"] = d
    else:
        info["stages"]["3_pathbmask"] = json.loads((pathbmask_dir / "done.json").read_text())
        print(f"[stage 3] skip (done): {pathbmask_dir}")

    # 4. rewrite_qwen3_pathb (rotary hoist)
    if force or not _done(pathb_dir):
        pathb_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        rc = _run([
            str(venv_python),
            str(SCRIPTS_DIR / "rewrite_qwen3_pathb.py"),
            "--src-dir", str(pathbmask_dir),
            "--dst-dir", str(pathb_dir),
        ], log_path=pathb_dir / "stage.log")
        if rc != 0:
            raise RuntimeError(f"rewrite_qwen3_pathb failed (rc={rc})")
        d = {"stage": "4_pathb", "wall_s": time.time() - t0,
             "out_dir": str(pathb_dir)}
        _mark_done(pathb_dir, d)
        info["stages"]["4_pathb"] = d
    else:
        info["stages"]["4_pathb"] = json.loads((pathb_dir / "done.json").read_text())
        print(f"[stage 4] skip (done): {pathb_dir}")

    # 5. pin shapes
    if force or not _done(pinned_dir):
        pinned_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        rc = _run([
            str(venv_python),
            str(SCRIPTS_DIR / "pin_shapes_qwen3_4b.py"),
            "--src-dir", str(pathb_dir),
            "--dst-dir", str(pinned_dir),
            "--ctx", str(ctx),
            "--seq-q", "1",
        ], log_path=pinned_dir / "stage.log")
        if rc != 0:
            raise RuntimeError(f"pin_shapes_qwen3_4b failed (rc={rc})")
        d = {"stage": "5_pinned", "wall_s": time.time() - t0,
             "out_dir": str(pinned_dir), "ctx": ctx}
        _mark_done(pinned_dir, d)
        info["stages"]["5_pinned"] = d
    else:
        info["stages"]["5_pinned"] = json.loads((pinned_dir / "done.json").read_text())
        print(f"[stage 5] skip (done): {pinned_dir}")

    return pinned_dir, info
