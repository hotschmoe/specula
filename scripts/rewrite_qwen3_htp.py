"""Phase 5 step 6 - rewrite Qwen3-0.6B ONNX into HTP-compilable shape.

Replaces the session 7-9 `simplify_qwen3_no_mask.py` pipeline, which
used onnxsim and produced a computationally corrupted graph (see
docs/phase5_export_on_x86.md "DEPRECATED" section). The corruption
mode was silent: the graph looked structurally clean (0 Cast-to-BOOL,
0 Range) but the onnxsim constant-folder folded a position-dependent
subgraph using a pinned dummy `position_ids` value, which is why the
decoded text came out as anti-correlated Arabic glyphs.

This script uses protobuf-level edits only. No onnxsim on the
interior graph -- ever.

Three modes, chained:

    --mode stage
        Input : models/qwen3-0.6b-optimum/
        Output: models/qwen3-0.6b-staged/
        Does  : (2a) promote attention_mask INT64 input -> initializer
                     of shape [1, 512] all-ones.
                (2b) elide Where(IsNaN(x), const, x) -> x on the 28
                     per-layer softmax NaN-safety guards.
        Safe  : each transform individually verified cos = +1.0000 vs
                source on CPU-ORT in session 10 (doc's table).

    --mode fold-patha
        Input : models/qwen3-0.6b-staged/
        Output: models/qwen3-0.6b-patha/
        Does  : replace Gather_5 with
                ConstantOfShape(Shape(indices), value=True, dtype=BOOL),
                drop the attention_mask Cast(->BOOL), trim orphans.
        Tests : "HTP rejects Cast-to-BOOL, not BOOL tensors themselves."
                If HTP actually rejects any BOOL tensor, this artifact
                fails at AI Hub compile and we fall through to pathbmask.

    --mode fold-pathbmask
        Input : models/qwen3-0.6b-staged/
        Output: models/qwen3-0.6b-pathbmask/
        Does  : replace Where(bool_mask, scores, -inf) with
                Add(scores, attention_bias), where attention_bias is a
                new FP16 graph input of shape [1, 1, 1, 512]. Deletes
                the entire BOOL mask-construction subgraph. Runtime
                supplies a pre-computed additive causal mask per step
                (all zeros for a fully-populated decode window).
        Tests : "Adopting Qualcomm's production mask pattern is
                enough for HTP compile." Matches the pattern in
                models/qualcomm-qwen3-4b-ref/ (per phase5 doc Path B).

Run (full pipeline, from x86 box):

    .venv-x86-export/Scripts/python.exe scripts/rewrite_qwen3_htp.py --mode stage
    .venv-x86-export/Scripts/python.exe scripts/rewrite_qwen3_htp.py --mode fold-patha
    .venv-x86-export/Scripts/python.exe scripts/rewrite_qwen3_htp.py --mode fold-pathbmask

Then validate with scripts/probe_cos_vs_source.py before transfer.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper


REPO_ROOT = Path(__file__).resolve().parent.parent
OPTIMUM_DIR = REPO_ROOT / "models" / "qwen3-0.6b-optimum"
STAGED_DIR = REPO_ROOT / "models" / "qwen3-0.6b-staged"
PATHA_DIR = REPO_ROOT / "models" / "qwen3-0.6b-patha"
PATHBMASK_DIR = REPO_ROOT / "models" / "qwen3-0.6b-pathbmask"

CTX_LEN = 512  # hard decode-window size we compile for


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_model(src_dir: Path) -> onnx.ModelProto:
    """Load an ONNX model with external data resolved into memory.

    Resolving up-front matters because later we rewrite the graph
    (promoting inputs to initializers, deleting nodes), and some of
    those initializers are external. The clean way to not worry about
    it is to pull everything into memory now and re-save consolidated
    later.
    """
    onnx_path = src_dir / "model.onnx"
    if not onnx_path.exists():
        print(f"ERROR: model.onnx missing at {onnx_path}", file=sys.stderr)
        sys.exit(2)
    t0 = time.perf_counter()
    model = onnx.load(str(onnx_path), load_external_data=True)
    print(f"loaded {onnx_path.name} in {time.perf_counter() - t0:.2f} s "
          f"({len(model.graph.node)} nodes, {len(model.graph.initializer)} initializers)")
    return model


def save_model(model: onnx.ModelProto, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = dst_dir / "model.onnx"
    data_name = "model.onnx_data"
    print(f"saving to {onnx_path} (+ {data_name})")
    t0 = time.perf_counter()
    onnx.save(
        model,
        str(onnx_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
        size_threshold=1024,
    )
    elapsed = time.perf_counter() - t0
    onnx_mb = onnx_path.stat().st_size / (1024 * 1024)
    data_mb = (dst_dir / data_name).stat().st_size / (1024 * 1024)
    print(f"  saved in {elapsed:.2f} s; graph={onnx_mb:.1f} MB, data={data_mb:.1f} MB")


def copy_sidecars(src_dir: Path, dst_dir: Path) -> None:
    """Bring over tokenizer + config + chat_template from the optimum dir."""
    sidecar_names = (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "merges.txt",
        "vocab.json",
        "chat_template.jinja",
        "added_tokens.json",
    )
    copied = 0
    for name in sidecar_names:
        src = src_dir / name
        if not src.exists():
            continue
        dst = dst_dir / name
        shutil.copy2(src, dst)
        copied += 1
    print(f"  copied {copied} sidecar files")


def summarize_graph(model: onnx.ModelProto, label: str) -> None:
    from collections import Counter
    op_counts: Counter = Counter()
    for n in model.graph.node:
        op_counts[(n.domain or "", n.op_type)] += 1
    n_isnan = sum(c for (d, o), c in op_counts.items() if o == "IsNaN")
    n_range = sum(c for (d, o), c in op_counts.items() if o == "Range")
    n_ms = sum(c for (d, _o), c in op_counts.items() if d == "com.microsoft")
    # BOOL casts
    n_cast_to_bool = 0
    for n in model.graph.node:
        if n.op_type == "Cast":
            for a in n.attribute:
                if a.name == "to" and a.i == TensorProto.BOOL:
                    n_cast_to_bool += 1
    print(
        f"  [{label}] nodes={len(model.graph.node)} "
        f"inputs={len(model.graph.input)} outputs={len(model.graph.output)} "
        f"initializers={len(model.graph.initializer)} | "
        f"IsNaN={n_isnan} Range={n_range} Cast->BOOL={n_cast_to_bool} "
        f"com.microsoft={n_ms}"
    )


# ---------------------------------------------------------------------------
# Stage (2a + 2b)
# ---------------------------------------------------------------------------

def promote_attention_mask(model: onnx.ModelProto) -> None:
    """Step 2a: attention_mask INT64 input -> initializer [1, CTX_LEN] ones.

    In the decode regime we compile for, the full 512-position window
    is always valid (no padding), so attention_mask is always [1]*512.
    Making it a graph-level constant unblocks everything downstream that
    needs a known-value mask for HTP compatibility.
    """
    names_in = [i.name for i in model.graph.input]
    if "attention_mask" not in names_in:
        print("  WARNING: attention_mask not in graph inputs; skipping 2a")
        return
    inputs_to_keep = [i for i in model.graph.input if i.name != "attention_mask"]
    del model.graph.input[:]
    model.graph.input.extend(inputs_to_keep)

    arr = np.ones((1, CTX_LEN), dtype=np.int64)
    init = numpy_helper.from_array(arr, name="attention_mask")
    model.graph.initializer.append(init)
    print(f"  2a: attention_mask promoted to initializer [1, {CTX_LEN}] INT64 all-ones")


def elide_isnan_where_guards(model: onnx.ModelProto) -> int:
    """Step 2b: rewrite Where(IsNaN(x), const, x) -> x.

    optimum emits 28 of these, one per attention layer, wrapping the
    softmax output to defend against NaN. HTP rejects BOOL everywhere,
    and IsNaN outputs BOOL. In the decode-only + full-KV regime,
    softmax never produces NaN (denominator always >= 1 non-zero
    contribution because the causal mask always permits >= 1 token),
    so Where(IsNaN(x), 0, x) == x identically. Returns count elided.
    """
    nodes = list(model.graph.node)
    producer = {o: n for n in nodes for o in n.output}
    renames: dict[str, str] = {}
    drop: set[int] = set()

    for w in nodes:
        if w.op_type != "Where":
            continue
        cond, _, false_val = w.input[:3]
        isnan_node = producer.get(cond)
        if isnan_node is None or isnan_node.op_type != "IsNaN":
            continue
        # Only elide if Where's false-branch == IsNaN's operand (the
        # "return x unless x is NaN" shape). Reject other Where uses.
        if isnan_node.input[0] != false_val:
            continue
        renames[w.output[0]] = false_val
        drop.add(id(w))
        # Drop the IsNaN only if nothing else consumes it (it usually
        # feeds only the paired Where).
        isnan_consumers = [n for n in nodes if isnan_node.output[0] in n.input]
        if len(isnan_consumers) == 1:
            drop.add(id(isnan_node))

    # Rewrite downstream consumers of the now-dead Where outputs.
    for n in nodes:
        for i, inp in enumerate(list(n.input)):
            if inp in renames:
                n.input[i] = renames[inp]
    for out in model.graph.output:
        if out.name in renames:
            out.name = renames[out.name]

    kept = [n for n in nodes if id(n) not in drop]
    del model.graph.node[:]
    model.graph.node.extend(kept)

    # value_info cleanup: drop entries whose tensors no longer exist.
    live = {vi.name for vi in model.graph.input}
    live |= {vi.name for vi in model.graph.output}
    live |= {o for n in kept for o in n.output}
    live |= {i.name for i in model.graph.initializer}
    surviving_vi = [vi for vi in model.graph.value_info if vi.name in live]
    del model.graph.value_info[:]
    model.graph.value_info.extend(surviving_vi)

    return len(renames)


def run_stage() -> int:
    model = load_model(OPTIMUM_DIR)
    summarize_graph(model, "input")

    print("2a: promoting attention_mask to initializer ...")
    promote_attention_mask(model)

    print("2b: eliding Where(IsNaN(x), c, x) -> x ...")
    elided = elide_isnan_where_guards(model)
    print(f"  2b: elided {elided} guards (expected 28)")

    # Don't re-run shape inference here. Shape inference on the full
    # 7k-node model with external-data resolved is slow and not needed
    # at this point -- the rewrites preserve dtypes and shapes on every
    # remaining edge.

    summarize_graph(model, "staged")
    save_model(model, STAGED_DIR)
    copy_sidecars(OPTIMUM_DIR, STAGED_DIR)
    return 0


# ---------------------------------------------------------------------------
# Fold A - placeholder; written after recon
# ---------------------------------------------------------------------------

def run_fold_patha() -> int:
    print("fold-patha: not yet implemented; run recon first "
          "(scripts/survey_bool_region.py)")
    return 3


# ---------------------------------------------------------------------------
# Fold B-mask - placeholder; written after recon
# ---------------------------------------------------------------------------

def run_fold_pathbmask() -> int:
    print("fold-pathbmask: not yet implemented; run recon first "
          "(scripts/survey_bool_region.py)")
    return 3


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("stage", "fold-patha", "fold-pathbmask"),
    )
    args = parser.parse_args()

    dispatch = {
        "stage": run_stage,
        "fold-patha": run_fold_patha,
        "fold-pathbmask": run_fold_pathbmask,
    }
    return dispatch[args.mode]()


if __name__ == "__main__":
    sys.exit(main())
