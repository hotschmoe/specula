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
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


REPO_ROOT = Path(__file__).resolve().parent.parent

# Defaults for the original Qwen3-0.6B pipeline. Override via --model-stem
# (e.g. `--model-stem qwen3-4b-optimum-arm` -> model dirs all share that
# stem prefix, so `<stem>-staged`, `<stem>-patha`, `<stem>-pathbmask`).
DEFAULT_STEM = "qwen3-0.6b"

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


def rewire_inputs(nodes: list, old_name: str, new_name: str) -> int:
    """Replace every occurrence of old_name in any node's input with new_name.
    Returns count of edits made."""
    edits = 0
    for n in nodes:
        for i, inp in enumerate(n.input):
            if inp == old_name:
                n.input[i] = new_name
                edits += 1
    return edits


def rewire_graph_outputs(model: onnx.ModelProto, old_name: str, new_name: str) -> int:
    edits = 0
    for o in model.graph.output:
        if o.name == old_name:
            o.name = new_name
            edits += 1
    return edits


def drop_nodes_by_name(model: onnx.ModelProto, names: set[str]) -> int:
    """Drop graph nodes whose .name is in the set. Returns drop count."""
    kept = [n for n in model.graph.node if n.name not in names]
    dropped = len(model.graph.node) - len(kept)
    del model.graph.node[:]
    model.graph.node.extend(kept)
    return dropped


def prune_dead_nodes(model: onnx.ModelProto) -> int:
    """Remove nodes whose outputs nothing consumes. Iterate until fixed point.

    A node is dead iff:
      - none of its outputs are graph outputs, AND
      - none of its outputs are inputs to any surviving node.

    This is the standard reachability DCE for ONNX graphs. Runs after the
    fold rewrites to drop the orphaned BOOL-chain nodes.
    """
    graph_out_names = {o.name for o in model.graph.output}
    total_dropped = 0
    while True:
        nodes = list(model.graph.node)
        consumed: set[str] = set(graph_out_names)
        for n in nodes:
            for i in n.input:
                consumed.add(i)
        kept = []
        for n in nodes:
            if any(o in consumed for o in n.output):
                kept.append(n)
        dropped = len(nodes) - len(kept)
        if dropped == 0:
            break
        del model.graph.node[:]
        model.graph.node.extend(kept)
        total_dropped += dropped
    return total_dropped


def prune_unused_initializers(model: onnx.ModelProto) -> int:
    """Drop initializers not referenced by any node's input."""
    refs: set[str] = set()
    for n in model.graph.node:
        for i in n.input:
            refs.add(i)
    for o in model.graph.output:
        refs.add(o.name)
    kept = [init for init in model.graph.initializer if init.name in refs]
    dropped = len(model.graph.initializer) - len(kept)
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    return dropped


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


def run_stage(optimum_dir: Path, staged_dir: Path) -> int:
    model = load_model(optimum_dir)
    summarize_graph(model, "input")

    print("2a: promoting attention_mask to initializer ...")
    promote_attention_mask(model)

    print("2b: eliding Where(IsNaN(x), c, x) -> x ...")
    elided = elide_isnan_where_guards(model)
    # The expected count equals num_hidden_layers; printed here only for
    # operator sanity, not enforced (works for any layer count).
    print(f"  2b: elided {elided} guards (one per attention layer)")

    # Don't re-run shape inference here. Shape inference on the full
    # 7k-node model with external-data resolved is slow and not needed
    # at this point -- the rewrites preserve dtypes and shapes on every
    # remaining edge.

    summarize_graph(model, "staged")
    save_model(model, staged_dir)
    copy_sidecars(optimum_dir, staged_dir)
    return 0


# ---------------------------------------------------------------------------
# Fold A - placeholder; written after recon
# ---------------------------------------------------------------------------

def run_fold_patha(optimum_dir: Path, staged_dir: Path, patha_dir: Path) -> int:
    """Path A: replace Gather_5 with ConstantOfShape(Shape(idx), True, BOOL)
    + delete the 2 BOOL->BOOL identity casts (Cast_5, Cast_6) + fold away
    Cast_2 (INT64 attention_mask -> BOOL -> obsolete after Gather_5 swap).
    End state: 0 Cast->BOOL in the graph.

    This is the Qualcomm-HTP Path A experiment: does HTP reject Cast-to-BOOL
    specifically, or any BOOL tensor? If HTP accepts BOOL tensors as long
    as the casts are gone, this artifact compiles and validates. If HTP
    rejects any BOOL tensor, this artifact still fails at AI Hub and we
    use Path B-mask instead. Either way, the compile result is research
    signal we didn't have before.
    """
    model = load_model(staged_dir)
    summarize_graph(model, "input")

    # Names from recon. Validate they exist before editing anything.
    gather5_name = "/model/Gather_5"
    flatten_name = "/model/Flatten"
    cast2_name = "/model/Cast_2"
    cast5_name = "/model/Cast_5"
    cast6_name = "/model/Cast_6"

    node_by_name = {n.name: n for n in model.graph.node if n.name}
    for req in (gather5_name, flatten_name, cast2_name, cast5_name, cast6_name):
        if req not in node_by_name:
            print(f"ERROR: expected node {req} not found in staged graph", file=sys.stderr)
            return 2

    gather5 = node_by_name[gather5_name]
    cast2 = node_by_name[cast2_name]
    cast5 = node_by_name[cast5_name]
    cast6 = node_by_name[cast6_name]
    gather5_indices = gather5.input[1]  # /model/Add_2_output_0
    gather5_output = gather5.output[0]  # /model/Gather_5_output_0

    nodes = list(model.graph.node)

    # ---- Step A.1: build Shape + ConstantOfShape to replace Gather_5 ----
    # Shape(gather5_indices) -> int64 1-D shape tensor
    shape_tensor_name = "/model/PathA/GatherShape_output_0"
    shape_node = helper.make_node(
        "Shape",
        inputs=[gather5_indices],
        outputs=[shape_tensor_name],
        name="/model/PathA/GatherShape",
    )
    # ConstantOfShape with value = True (BOOL scalar)
    true_scalar = numpy_helper.from_array(
        np.ones((1,), dtype=bool),
        name="/model/PathA/ConstTrue_value",
    )
    constofshape_node = helper.make_node(
        "ConstantOfShape",
        inputs=[shape_tensor_name],
        outputs=[gather5_output],  # reuse Gather_5's output name -> no rewire needed for downstream
        name="/model/PathA/ConstantOfShape",
        value=true_scalar,
    )

    # Insert the two new nodes BEFORE Gather_5 in node order. Their outputs
    # feed the same tensor name Gather_5 used, so downstream is unchanged.
    idx_gather5 = nodes.index(gather5)
    new_nodes = nodes[:idx_gather5] + [shape_node, constofshape_node] + nodes[idx_gather5:]
    # Drop Gather_5 and Flatten by name (Gather_5 replaced; Flatten's only
    # consumer was Gather_5).
    drop_names = {gather5_name, flatten_name}

    # ---- Step A.2: eliminate the BOOL->BOOL identity casts ----
    # Cast_5: rewire consumers of Cast_5_output_0 -> LessOrEqual_output_0
    cast5_edits = rewire_inputs(new_nodes, cast5.output[0], cast5.input[0])
    cast5_edits += rewire_graph_outputs(model, cast5.output[0], cast5.input[0])
    drop_names.add(cast5_name)
    print(f"  A.2a: Cast_5 identity rewired ({cast5_edits} edge(s))")

    # Cast_6: rewire consumers of Cast_6_output_0 -> Reshape_1_output_0
    cast6_edits = rewire_inputs(new_nodes, cast6.output[0], cast6.input[0])
    cast6_edits += rewire_graph_outputs(model, cast6.output[0], cast6.input[0])
    drop_names.add(cast6_name)
    print(f"  A.2b: Cast_6 identity rewired ({cast6_edits} edge(s))")

    # ---- Step A.3: fold Cast_2 away ----
    # Cast_2 feeds two consumers: Flatten (already dropped via step A.1) and
    # Shape_4 (which reads shape only -- dtype doesn't matter). Rewire Shape_4
    # to read attention_mask directly, then drop Cast_2.
    cast2_out = cast2.output[0]
    cast2_edits = rewire_inputs(new_nodes, cast2_out, cast2.input[0])
    # graph_outputs never contain Cast_2_output_0 but be safe:
    cast2_edits += rewire_graph_outputs(model, cast2_out, cast2.input[0])
    drop_names.add(cast2_name)
    print(f"  A.3: Cast_2 folded (rewired {cast2_edits} edge(s) to attention_mask)")

    # Commit the node list.
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    dropped = drop_nodes_by_name(model, drop_names)
    print(f"  dropped {dropped} nodes: {sorted(drop_names)}")

    # Clean up any orphaned nodes from the rewire + drop.
    extra = prune_dead_nodes(model)
    if extra:
        print(f"  DCE removed {extra} orphaned upstream nodes")
    unused_init = prune_unused_initializers(model)
    if unused_init:
        print(f"  pruned {unused_init} unused initializers")

    summarize_graph(model, "patha")
    save_model(model, patha_dir)
    copy_sidecars(optimum_dir, patha_dir)
    return 0


# ---------------------------------------------------------------------------
# Fold B-mask - direct attention_bias splice
# ---------------------------------------------------------------------------

def run_fold_pathbmask(optimum_dir: Path, staged_dir: Path, pathbmask_dir: Path) -> int:
    """Path B-mask: bypass the BOOL mask subgraph entirely by feeding the
    attention additive bias in as a new FP32 graph input. Rewire each of
    the 28 Add_2 nodes to read from the new input instead of Where_2's
    output, then let DCE clean up the BOOL chain.

    Shape: the bias is `[1, 1, seq_q, seq_k]` FP32, fully dynamic on
    seq_q/seq_k so the graph still accepts any decode position. For the
    compile-target decode regime (past=511, seq_q=1, seq_k=512), runtime
    feeds all zeros (all positions valid, no future to mask).

    Pattern originates from the Qualcomm Qwen3-4B X2E bundle
    (models/qualcomm-qwen3-4b-ref/ per docs/phase5_export_on_x86.md
    Path B). This artifact tests whether adopting just the mask half of
    their production pattern is enough to clear HTP compile; the RoPE
    half is deferred to a future cycle.
    """
    model = load_model(staged_dir)
    summarize_graph(model, "input")

    # ---- Step 1: add attention_bias graph input ----
    bias_name = "attention_bias"
    bias_type = helper.make_tensor_type_proto(
        elem_type=TensorProto.FLOAT,
        shape=["batch_size", 1, "seq_q", "seq_k"],
    )
    bias_value_info = helper.make_value_info(bias_name, bias_type)
    model.graph.input.append(bias_value_info)
    print(f"  1: added {bias_name} graph input "
          f"(FLOAT shape=[batch,1,seq_q,seq_k])")

    # ---- Step 2: rewire the 28 per-layer Add_2 nodes ----
    # Each layer has an Add_2 that takes (MatMul_output_0, Where_2_output_0).
    # Swap the second input to point at attention_bias.
    splices = 0
    for n in model.graph.node:
        if n.op_type != "Add":
            continue
        if not n.name or not n.name.endswith("/self_attn/Add_2"):
            continue
        if len(n.input) < 2:
            continue
        # Identify which input is Where_2's output and rewire it.
        # Convention in optimum export: input[1] is Where_2_output_0.
        for i, inp in enumerate(n.input):
            if inp.endswith("/self_attn/Where_2_output_0"):
                n.input[i] = bias_name
                splices += 1
                break
    print(f"  2: spliced attention_bias into {splices} Add_2 nodes "
          "(one per attention layer)")
    if splices == 0:
        print(f"WARNING: zero splices — node naming convention may have changed")

    # ---- Step 3: let DCE drop the BOOL chain ----
    # The 28 Where_2 outputs are now orphaned; DCE reclaims them and
    # walks back through Expand, Slice_4, And, And_1, Cast_5/6, LessOrEqual,
    # Range_2, etc., until it hits the attention_mask initializer + Constant
    # sources that have no other consumer.
    dropped = prune_dead_nodes(model)
    print(f"  3: DCE removed {dropped} now-orphaned nodes "
          "(Where_2/Slice_4/Expand/And/LessOrEqual/... chain)")
    unused_init = prune_unused_initializers(model)
    print(f"     + pruned {unused_init} unused initializers "
          "(attention_mask typically drops out here)")

    summarize_graph(model, "pathbmask")
    save_model(model, pathbmask_dir)
    copy_sidecars(optimum_dir, pathbmask_dir)
    return 0


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
    parser.add_argument(
        "--model-stem",
        default=DEFAULT_STEM,
        help=f"Model directory stem (default: {DEFAULT_STEM!r}). "
             f"Used to derive <stem>-optimum, <stem>-staged, <stem>-patha, "
             f"<stem>-pathbmask under models/.",
    )
    parser.add_argument(
        "--optimum-dir", type=Path, default=None,
        help="Override the optimum source directory (default: models/<stem>-optimum)",
    )
    parser.add_argument(
        "--staged-dir", type=Path, default=None,
        help="Override the staged output/source directory (default: models/<stem>-staged)",
    )
    parser.add_argument(
        "--patha-dir", type=Path, default=None,
        help="Override the path-A output directory (default: models/<stem>-patha)",
    )
    parser.add_argument(
        "--pathbmask-dir", type=Path, default=None,
        help="Override the path-B-mask output directory (default: models/<stem>-pathbmask)",
    )
    args = parser.parse_args()

    models_root = REPO_ROOT / "models"
    optimum_dir = args.optimum_dir or (models_root / f"{args.model_stem}-optimum")
    staged_dir = args.staged_dir or (models_root / f"{args.model_stem}-staged")
    patha_dir = args.patha_dir or (models_root / f"{args.model_stem}-patha")
    pathbmask_dir = args.pathbmask_dir or (models_root / f"{args.model_stem}-pathbmask")

    if args.mode == "stage":
        return run_stage(optimum_dir, staged_dir)
    if args.mode == "fold-patha":
        return run_fold_patha(optimum_dir, staged_dir, patha_dir)
    if args.mode == "fold-pathbmask":
        return run_fold_pathbmask(optimum_dir, staged_dir, pathbmask_dir)
    return 2


if __name__ == "__main__":
    sys.exit(main())
