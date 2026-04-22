"""Phase 5 step 4a - stage a Qwen3 ONNX for qai-hub upload.

qai-hub requires model directories to contain only .onnx / .data /
.encodings / .bin files. The optimum export names its external-data
file `model.onnx_data`, which qai-hub rejects. Fix is a two-step:

    (1) patch every TensorProto.external_data entry that points at
        `model.onnx_data` to point at `model.data` instead.
    (2) copy the original weights file under the new name. (We used
        to hardlink for zero-duplication, but ORT 1.24+ refuses to
        read external-data files with >1 hard links.)

A third transform pins every graph input's declared shape to the
static dims AI Hub expects. Session 11 failed both Path A and Path
B-mask because the optimum export keeps batch + sequence dims as
symbolic `-1`; AI Hub's compiler pipeline has a pass
(`_op_identity.py:30`) that calls `np.broadcast_shapes` on dims and
chokes on SymbolicDim. See `docs/phase5_step6_compile_retro.md`.
The pin is a pure protobuf edit on `ValueInfoProto` and preserves
graph numerics bit-for-bit (cos=1.0 per session 10 bisection).

Multiple source variants are supported via --path; each comes from
`docs/phase5_step6_export_report.md` (x86 handoff):

    patha      BOOL casts folded out (ConstantOfShape replaces Gather_5),
               BOOL tensors remain downstream. Drops attention_mask input.
    pathbmask  entire BOOL mask subgraph deleted + new FP32 attention_bias
               input spliced into 28 Add_2 nodes. Zero BOOL tensors.
    pathb      pathbmask + rotary_emb hoisted out of the graph. Adds
               position_ids_cos / position_ids_sin as the trailing two
               graph inputs. Required for w4a16 PTQ (inline rotary fails
               QNN op-validation — see `docs/qwen3_perf_levers_investigation.md`
               Lever C). Source handoff: `status_x86.md` session 2.

The session 7-9 `nomask` variant is quarantined (onnxsim corruption) and
not selectable here.

Run:
    .venv\\Scripts\\python.exe scripts\\prep_onnx_for_ai_hub.py --path patha
    .venv\\Scripts\\python.exe scripts\\prep_onnx_for_ai_hub.py --path pathbmask
    .venv\\Scripts\\python.exe scripts\\prep_onnx_for_ai_hub.py --path pathb --ctx 256
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import onnx
import onnxruntime as ort
from onnx import shape_inference


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compile_qwen3_ai_hub import (  # noqa: E402
    DEFAULT_CONTEXT_MAX,
    build_input_specs,
    build_paths,
)

OLD_LOCATION = "model.onnx_data"
NEW_LOCATION = "model.data"


def _patch_tensor(tensor: onnx.TensorProto) -> bool:
    for entry in tensor.external_data:
        if entry.key == "location" and entry.value == OLD_LOCATION:
            entry.value = NEW_LOCATION
            return True
    return False


def patch_external_data_refs(model: onnx.ModelProto) -> tuple[int, int]:
    """Rewrite every `location` external_data entry in the model.

    Covers two places external tensors can live:
    - graph.initializer (the common case)
    - node.attribute.t for Constant / Constant-like ops with tensor
      attributes (optimum's --no-post-process export keeps a lot of
      constants here instead of promoting them to initializers).

    Returns (initializer_patches, node_attribute_patches).
    """
    init_patched = 0
    for tensor in model.graph.initializer:
        if _patch_tensor(tensor):
            init_patched += 1

    node_patched = 0
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR and attr.t.external_data:
                if _patch_tensor(attr.t):
                    node_patched += 1
            if attr.type == onnx.AttributeProto.TENSORS:
                for t in attr.tensors:
                    if t.external_data and _patch_tensor(t):
                        node_patched += 1
    return init_patched, node_patched


def materialize(src: Path, dst: Path) -> str:
    if dst.exists():
        dst.unlink()
    shutil.copy2(src, dst)
    return "copy"


def pin_input_shapes(model: onnx.ModelProto, specs: dict) -> tuple[int, int]:
    """Rewrite each matching graph input's TensorShapeProto to static dims.

    For each input listed in `specs`, overwrite its `dim` entries so every
    `dim_value` is the concrete integer from the spec. Inputs not in
    `specs` are left alone (spec-vs-ONNX parity is enforced at compile
    `--check` time).

    Returns (pinned_count, already_static_count).
    """
    pinned = 0
    already_static = 0
    for inp in model.graph.input:
        spec = specs.get(inp.name)
        if spec is None:
            continue
        target_shape, _dtype = spec
        t_shape = inp.type.tensor_type.shape
        current = tuple(
            (d.dim_value if d.HasField("dim_value") else d.dim_param)
            for d in t_shape.dim
        )
        if current == tuple(int(x) for x in target_shape):
            already_static += 1
            continue
        del t_shape.dim[:]
        for dim_value in target_shape:
            t_shape.dim.add().dim_value = int(dim_value)
        pinned += 1
    return pinned, already_static


# Maps the three symbolic dim names optimum-onnx emits for a
# causal-decoder-with-past export to their concrete values in our
# compile regime (CONTEXT_MAX = 512, decode-only). If any *other*
# dim_param name shows up in a future export we want to fail loudly
# rather than ship a graph that AI Hub's rewriter will choke on later
# with a SymbolicDim TypeError (see docs/phase5_step6_compile_retro.md).
def build_optimum_dim_params(ctx: int) -> dict:
    """The three symbolic dim names optimum-onnx emits for a causal
    decoder with past-kv export, resolved to the concrete values for a
    given CONTEXT_MAX tier."""
    return {
        "batch_size": 1,
        "sequence_length": 1,
        "past_sequence_length + sequence_length": ctx,
    }


def resolve_dim_params(model: onnx.ModelProto, dim_map: dict) -> tuple[int, set]:
    """Replace every matching `dim_param` on graph.input / .output / .value_info
    with a concrete `dim_value`.

    Session 11 found that pinning graph.input alone is insufficient: graph
    outputs (logits, present.N.*) retained dim_params like
    `'past_sequence_length + sequence_length'` which propagate into AI Hub's
    onnx_ir representation as SymbolicDim and crash the next rewrite pass.

    Returns (substitutions_made, unresolved_names) where unresolved_names
    is the set of dim_param strings that appeared but weren't in dim_map.
    Caller should fail if unresolved_names is non-empty so we never upload
    a graph with surviving symbolic dims.
    """
    substituted = 0
    unresolved: set[str] = set()
    for collection in (model.graph.input, model.graph.output, model.graph.value_info):
        for vi in collection:
            for d in vi.type.tensor_type.shape.dim:
                if not d.HasField("dim_param"):
                    continue
                name = d.dim_param
                if name in dim_map:
                    # Setting dim_value on a oneof auto-clears dim_param.
                    d.dim_value = int(dim_map[name])
                    substituted += 1
                else:
                    unresolved.add(name)
    return substituted, unresolved


def stage(path_key: str, ctx: int = DEFAULT_CONTEXT_MAX) -> int:
    path_cfg = build_paths(path_key, ctx)
    source_dir = path_cfg["source_dir"]
    staging = path_cfg["staging_dir"]
    source_onnx = source_dir / "model.onnx"
    # Two possible source-data names. The optimum export always emits
    # `model.onnx_data`; the pathb rewrite emits `model.data` directly
    # (see scripts/rewrite_qwen3_pathb.py: save_as_external_data with
    # location="model.data"). Either is fine — staging always renames
    # to `model.data` and `patch_external_data_refs` below is a no-op
    # when the rename is already done upstream.
    source_data_optimum = source_dir / "model.onnx_data"
    source_data_prerenamed = source_dir / "model.data"
    if source_data_optimum.exists():
        source_data = source_data_optimum
    elif source_data_prerenamed.exists():
        source_data = source_data_prerenamed
    else:
        print(f"ERROR: external-data file missing; looked for "
              f"{source_data_optimum.name} and {source_data_prerenamed.name} "
              f"in {source_dir}")
        return 2
    staged_onnx = staging / "model.onnx"
    staged_data = staging / "model.data"

    if not source_onnx.exists():
        print(f"ERROR: source ONNX missing at {source_onnx}")
        return 2

    staging.mkdir(parents=True, exist_ok=True)
    print(f"path key            : {path_key}")
    print(f"ctx                 : {ctx}")
    print(f"source dir          : {source_dir}")
    print(f"staging dir         : {staging}")

    print(f"loading graph-only (no weights into memory) from {source_onnx.name} ...")
    t0 = time.perf_counter()
    model = onnx.load(str(source_onnx), load_external_data=False)
    print(f"  loaded in {time.perf_counter() - t0:.2f} s, {len(model.graph.initializer)} initializers")

    init_patched, node_patched = patch_external_data_refs(model)
    total = init_patched + node_patched
    print(f"patched {total} external_data references: '{OLD_LOCATION}' -> '{NEW_LOCATION}'")
    print(f"  initializer tensors        : {init_patched}")
    print(f"  node attribute tensors     : {node_patched}")
    if total == 0:
        print("WARNING: no external_data references found to patch.")

    with (source_dir / "config.json").open() as f:
        cfg = json.load(f)
    compile_extra = path_cfg["extra_input_specs"]
    specs = build_input_specs(cfg, compile_extra, ctx=ctx)
    pinned, already_static = pin_input_shapes(model, specs)
    print(f"pinned {pinned} graph input shapes to static dims "
          f"({already_static} already static, "
          f"{len(specs) - pinned - already_static} not declared on graph)")

    substituted, unresolved = resolve_dim_params(model, build_optimum_dim_params(ctx))
    print(f"resolved {substituted} dim_param references (inputs + outputs + value_info)")
    if unresolved:
        print(f"ERROR: unresolved dim_param names would crash AI Hub rewriter: {sorted(unresolved)}")
        print(f"  add to OPTIMUM_DIM_PARAMS in {Path(__file__).name} with the correct integer")
        return 2

    # Save the pre-ORT intermediate so ORT can load it. ORT needs the weights
    # sidecar too to resolve external_data refs during optimization.
    t0 = time.perf_counter()
    onnx.save(model, str(staged_onnx), save_as_external_data=False)
    print(f"  saved pre-ORT graph in {time.perf_counter() - t0:.2f} s, {staged_onnx.stat().st_size / (1024*1024):.1f} MB")

    print(f"copying weights {source_data.name} -> {staged_data.name} ...")
    t0 = time.perf_counter()
    mode = materialize(source_data, staged_data)
    elapsed = time.perf_counter() - t0
    data_mb = staged_data.stat().st_size / (1024 * 1024)
    print(f"  {mode} complete in {elapsed:.2f} s, {data_mb:.1f} MB")

    # Run ORT's ORT_ENABLE_BASIC graph optimization pass in-place on the
    # staged ONNX. Session 11 v4 found that input-shape-pinning alone is
    # insufficient: even with every input dim concrete and value_info
    # populated, AI Hub's compiler still produces ~735 SymbolicDim-tagged
    # internal tensors and crashes in the downstream rewriter
    # (`_op_identity.py:30: np.broadcast_shapes on SymbolicDim`). ORT-BASIC
    # folds constants + eliminates redundant nodes, taking the Path A graph
    # from 7580 -> 2061 nodes and leaving only 6 tensors (all in
    # /model/rotary_emb/) with residual symbolic dims. ENABLE_BASIC is
    # strictly safe numerically: no operator fusions that would introduce
    # com.microsoft ops (the ones we spent sessions 7-9 removing), just
    # constant folding + redundant-node elimination. Known equivalent to
    # source per session 10's `optimum-ortopt` bisection (cos=1.0).
    ort_out = staged_onnx.with_suffix(".ort_basic.onnx")
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    opts.optimized_model_filepath = str(ort_out)
    opts.log_severity_level = 3
    print(f"running ORT ENABLE_BASIC graph optimization (in-place) ...")
    t0 = time.perf_counter()
    ort.InferenceSession(str(staged_onnx), sess_options=opts, providers=["CPUExecutionProvider"])
    print(f"  ORT optimization complete in {time.perf_counter() - t0:.1f} s")

    # Re-load the optimized graph (external data refs still point at model.data;
    # ORT preserves those for the big weights and inlines smaller constants).
    model = onnx.load(str(ort_out), load_external_data=False)
    print(f"  post-ORT node count: {len(model.graph.node)} "
          f"(started at {len(onnx.load(str(staged_onnx), load_external_data=False).graph.node)})")

    # Second dim_param pass: ORT introduces its own `unk__N` placeholders for
    # dims it couldn't resolve. In our decode regime (batch=1, seq_len=1),
    # every such dim is 1 — hardcode that. If a future export produces
    # `unk__N` on a tensor whose concrete value isn't 1, the compile-side
    # `--check` will catch the mismatch.
    ort_subs = 0
    ort_unks: set[str] = set()
    for coll in (model.graph.input, model.graph.output, model.graph.value_info):
        for vi in coll:
            for d in vi.type.tensor_type.shape.dim:
                if d.HasField("dim_param") and d.dim_param.startswith("unk__"):
                    ort_unks.add(d.dim_param)
                    d.dim_value = 1
                    ort_subs += 1
    print(f"resolved {ort_subs} ORT-generated `unk__*` dim_params -> 1 "
          f"({len(ort_unks)} distinct names: {sorted(ort_unks)})")

    # Final shape inference pass — with ORT's folded graph + all dim_params
    # substituted, every internal tensor should now have fully concrete dims.
    t0 = time.perf_counter()
    model = shape_inference.infer_shapes(
        model, check_type=False, strict_mode=False, data_prop=True
    )
    unresolved_tensors = sum(
        1
        for vi in model.graph.value_info
        if any(not d.HasField("dim_value") for d in vi.type.tensor_type.shape.dim)
    )
    print(f"  final shape_inference: {len(model.graph.value_info)} value_info entries, "
          f"{unresolved_tensors} tensors still with symbolic dims, "
          f"{time.perf_counter() - t0:.1f} s")
    if unresolved_tensors:
        print(f"WARNING: {unresolved_tensors} tensors have unresolved shapes; "
              "AI Hub compile may still hit the SymbolicDim rewriter crash.")

    print(f"saving final graph to {staged_onnx.name} ...")
    t0 = time.perf_counter()
    onnx.save(model, str(staged_onnx), save_as_external_data=False)
    ort_out.unlink()  # cleanup intermediate
    print(f"  saved in {time.perf_counter() - t0:.2f} s, {staged_onnx.stat().st_size / (1024*1024):.1f} MB")

    print("\nstaging directory contents:")
    for p in sorted(staging.iterdir()):
        size_mb = p.stat().st_size / (1024 * 1024) if p.is_file() else 0
        print(f"  {p.name:20s} {size_mb:8.1f} MB")

    print(f"\n=== STATUS: ok; AI Hub upload can target {staging} ===")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", choices=("patha", "pathbmask", "pathb"), required=True)
    parser.add_argument("--ctx", type=int, default=DEFAULT_CONTEXT_MAX,
                        help=f"compiled context window size (default: {DEFAULT_CONTEXT_MAX})")
    args = parser.parse_args()
    return stage(args.path, ctx=args.ctx)


if __name__ == "__main__":
    sys.exit(main())
