"""Compare a specula-produced multi-part bundle against Qualcomm's
shipping `qwen3_4b-genie-w4a16-qualcomm_snapdragon_x2_elite/` for the
M2 reproduction goal.

What we can compare WITHOUT the X2E hardware (this is purely a
file-system + ONNX-level audit):

  - Part count + per-part .bin sizes
  - genie_config.json structural fields (ctx-size, kv-dim, pos-id-dim,
    rope-theta, vocab, sampler defaults)
  - htp_backend_ext_config.json (dsp_arch, soc_model, weight_sharing)
  - tokenizer.json sha256 (must match exactly)
  - Per-part bin_info.json structure (graph IO shapes & dtypes — KV
    dtype, cos/sin dtype, lm_head dtype)
  - AIMET encodings: per-part param/activation entry counts, KV-tensor
    bitwidth coverage

What we CAN'T compare without on-device execution:
  - First-decode logit cosine (needs HTP runtime, has to happen on X2E)
  - Argmax agreement on the 46-token oracle prompt

Output is a markdown report dropped to stdout (and optionally a file).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Optional


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _bin_files(d: Path) -> list[Path]:
    return sorted(d.glob("*part_*_of_*.bin"))


def _section(title: str) -> str:
    return f"\n## {title}\n"


def compare(ours_dir: Path, theirs_dir: Path) -> str:
    lines = ["# specula vs Qualcomm Qwen3-4B w4a16 bundle comparison\n"]
    lines.append(f"  ours   : `{ours_dir}`")
    lines.append(f"  theirs : `{theirs_dir}`")

    # --- 1. Part count + per-part .bin sizes ---
    lines.append(_section("Per-part .bin sizes"))
    ours_bins = _bin_files(ours_dir)
    theirs_bins = _bin_files(theirs_dir)
    if len(ours_bins) != len(theirs_bins):
        lines.append(f"⚠ part count mismatch: ours={len(ours_bins)}  theirs={len(theirs_bins)}")
    lines.append("")
    lines.append("| part | ours (MB) | theirs (MB) | Δ MB | Δ % |")
    lines.append("|---|---:|---:|---:|---:|")
    n = min(len(ours_bins), len(theirs_bins))
    for i in range(n):
        ous = ours_bins[i].stat().st_size / 1e6
        ths = theirs_bins[i].stat().st_size / 1e6
        delta = ous - ths
        pct = (delta / ths) * 100 if ths > 0 else 0
        lines.append(f"| {i+1} | {ous:.0f} | {ths:.0f} | {delta:+.0f} | {pct:+.1f}% |")
    o_tot = sum(p.stat().st_size for p in ours_bins) / 1e6
    t_tot = sum(p.stat().st_size for p in theirs_bins) / 1e6
    lines.append(f"| **total** | **{o_tot:.0f}** | **{t_tot:.0f}** | "
                 f"{o_tot - t_tot:+.0f} | {(o_tot - t_tot) / t_tot * 100:+.1f}% |")

    # --- 2. genie_config.json ---
    lines.append(_section("genie_config.json"))
    fields = [
        ("dialog.context.size", "ctx"),
        ("dialog.context.n-vocab", "vocab"),
        ("dialog.context.bos-token", "bos"),
        ("dialog.context.eos-token", "eos"),
        ("dialog.engine.backend.QnnHtp.kv-dim", "kv-dim"),
        ("dialog.engine.backend.QnnHtp.pos-id-dim", "pos-id-dim"),
        ("dialog.engine.backend.QnnHtp.rope-theta", "rope-theta"),
        ("dialog.engine.backend.QnnHtp.cpu-mask", "cpu-mask"),
        ("dialog.engine.backend.QnnHtp.use-mmap", "use-mmap"),
        ("dialog.engine.backend.QnnHtp.poll", "poll"),
        ("dialog.engine.n-threads", "n-threads"),
    ]
    def _get(d, dotted):
        for k in dotted.split("."):
            if not isinstance(d, dict) or k not in d:
                return None
            d = d[k]
        return d

    o_gc = json.loads((ours_dir / "genie_config.json").read_text()) if (ours_dir / "genie_config.json").exists() else {}
    t_gc = json.loads((theirs_dir / "genie_config.json").read_text()) if (theirs_dir / "genie_config.json").exists() else {}
    lines.append("| field | ours | theirs | match |")
    lines.append("|---|---|---|:---:|")
    for path, label in fields:
        ov = _get(o_gc, path); tv = _get(t_gc, path)
        match = "✓" if ov == tv else "✗"
        lines.append(f"| {label} | `{ov}` | `{tv}` | {match} |")

    # --- 3. htp_backend_ext_config.json ---
    lines.append(_section("htp_backend_ext_config.json"))
    o_hp = json.loads((ours_dir / "htp_backend_ext_config.json").read_text()) if (ours_dir / "htp_backend_ext_config.json").exists() else {}
    t_hp = json.loads((theirs_dir / "htp_backend_ext_config.json").read_text()) if (theirs_dir / "htp_backend_ext_config.json").exists() else {}
    fields_hp = [
        ("devices.0.soc_model", "soc_model"),
        ("devices.0.dsp_arch", "dsp_arch"),
        ("devices.0.cores.0.perf_profile", "perf_profile"),
        ("memory.mem_type", "mem_type"),
        ("context.weight_sharing_enabled", "weight_sharing_enabled"),
    ]
    def _get_a(d, dotted):
        # Same as _get but supports list-index by integer keys.
        for k in dotted.split("."):
            if isinstance(d, list):
                try: d = d[int(k)]
                except Exception: return None
            elif isinstance(d, dict) and k in d:
                d = d[k]
            else:
                return None
        return d
    lines.append("| field | ours | theirs | match |")
    lines.append("|---|---|---|:---:|")
    for path, label in fields_hp:
        ov = _get_a(o_hp, path); tv = _get_a(t_hp, path)
        match = "✓" if ov == tv else "✗"
        lines.append(f"| {label} | `{ov}` | `{tv}` | {match} |")

    # --- 4. tokenizer.json sha256 ---
    lines.append(_section("tokenizer.json sha256"))
    ot = ours_dir / "tokenizer.json"; tt = theirs_dir / "tokenizer.json"
    if ot.exists() and tt.exists():
        os_ = _sha256(ot); ts = _sha256(tt)
        lines.append(f"  ours   = `{os_}`")
        lines.append(f"  theirs = `{ts}`")
        lines.append(f"  match  = {'✓' if os_ == ts else '✗ — different tokenizer revisions in play'}")
    else:
        lines.append(f"  one or both missing (ours_exists={ot.exists()}, theirs_exists={tt.exists()})")

    # --- 5. AIMET encodings — per-part counts ---
    lines.append(_section("AIMET encodings (per-part counts)"))
    enc_dir = ours_dir / "encodings"
    if not enc_dir.exists():
        lines.append("  no encodings/ subdir under our bundle (skipping)")
    else:
        lines.append("| part | act_entries | param_entries |")
        lines.append("|---|---:|---:|")
        for i, enc_path in enumerate(sorted(enc_dir.glob("part_*_of_*.encodings")), start=1):
            try:
                e = json.loads(enc_path.read_text())
            except Exception as ex:
                lines.append(f"| {i} | (failed to parse: {ex}) | |")
                continue
            n_act = len(e.get("activation_encodings", e.get("activation_encoding", {})))
            n_par = len(e.get("param_encodings", e.get("param_encoding", {})))
            lines.append(f"| {i} | {n_act} | {n_par} |")

    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ours-dir", type=Path, required=True,
                   help="The bundle dir we produced "
                        "(e.g. .../09b_bundle_w4a16/qwen3_4b_w4a16_pathb_ctx512_x2e_v81/)")
    p.add_argument("--theirs-dir", type=Path,
                   default=Path("/workspace/models/qualcomm-qwen3-4b-ref/"
                                "qwen3_4b-genie-w4a16-qualcomm_snapdragon_x2_elite"),
                   help="Qualcomm reference bundle dir.")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    report = compare(args.ours_dir, args.theirs_dir)
    print(report)
    if args.out:
        args.out.write_text(report)
        print(f"\n[wrote {args.out}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
