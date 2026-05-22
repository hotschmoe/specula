"""Submit a Qualcomm AI Hub compile job for the Gemma 4 decoder.

Route 1, step 2: take the onnx-community fp16 decoder, pin its 5
symbolic dims to a concrete (ctx, AR) shape via `input_specs`, and
submit a `submit_compile_job` targeting `Snapdragon X2 Elite CRD`
(HTP v75 — the project's silicon, confirmed available on AI Hub).

We pin shapes through `input_specs` rather than rewriting the 4.5 GB
ONNX: the symbolic dims (batch_size, sequence_length,
past_sequence_length, total_sequence_length, num_logits_to_keep) are
substituted with concrete values; every already-fixed dim (256/512
per-layer KV head_dim, 1536 hidden, 35-layer PLE, 262144 vocab) is
read straight from the graph and passed through unchanged.

First run intentionally uses a SMALL ctx (512, AR=1 decode) — it is the
cheapest "does AI Hub digest this graph at all" probe. Gemma's in-graph
dual rotary + dynamic mask construction are exactly the ops the Qwen3
project had to rewrite for the bare QAIRT toolchain; whether AI Hub's
compiler handles them is the open question this job answers. Bump
`--ctx 32768` once the small compile succeeds.

Usage:
    python scripts/submit_compile_gemma4.py \
        --onnx ../models/gemma-4-E2B-it-ONNX/onnx/decoder_model_merged_fp16.onnx \
        --ctx 512 --dry-run        # print input_specs, submit nothing
    python scripts/submit_compile_gemma4.py --onnx ... --ctx 512
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnx
from onnx import TensorProto

_DT = {
    TensorProto.FLOAT: "float32",
    TensorProto.FLOAT16: "float16",
    TensorProto.INT64: "int64",
    TensorProto.INT32: "int32",
    TensorProto.BOOL: "int8",
}

DEVICE = "Snapdragon X2 Elite CRD"


def build_input_specs(onnx_path: Path, ctx: int, ar: int) -> dict:
    """Read the decoder graph, return {name: (shape, dtype)} with every
    symbolic dim pinned for an AR-token step at context length `ctx`."""
    model = onnx.load(str(onnx_path), load_external_data=False)

    # Symbolic-dim -> concrete value. For an AR=`ar` step at ctx `ctx`:
    #   sequence_length    = ar           (tokens fed this step)
    #   past_sequence_length = ctx - ar   (KV already cached)
    #   total_sequence_length = ctx       (past + current)
    subst = {
        "batch_size": 1,
        "sequence_length": ar,
        "past_sequence_length": ctx - ar,
        "total_sequence_length": ctx,
        "num_logits_to_keep": ar,
    }

    specs: dict = {}
    for t in model.graph.input:
        dims = []
        for d in t.type.tensor_type.shape.dim:
            if d.HasField("dim_value"):
                dims.append(int(d.dim_value))
            elif d.HasField("dim_param"):
                if d.dim_param not in subst:
                    raise ValueError(
                        f"input {t.name!r}: unknown symbolic dim "
                        f"{d.dim_param!r} — extend the subst map.")
                dims.append(subst[d.dim_param])
            else:
                raise ValueError(f"input {t.name!r}: dim with neither "
                                 f"value nor param")
        dtype = _DT.get(t.type.tensor_type.elem_type)
        if dtype is None:
            raise ValueError(f"input {t.name!r}: unmapped elem_type "
                             f"{t.type.tensor_type.elem_type}")
        specs[t.name] = (tuple(dims), dtype)
    return specs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", type=Path, required=True,
                    help="The .onnx file, OR a qai-hub model directory "
                         "(one .onnx + one .data — see repack_onnx.py). "
                         "A directory is required for the >2 GB decoder so "
                         "qai-hub uploads the external weights.")
    ap.add_argument("--ctx", type=int, default=512,
                    help="Context length to pin. Start at 512; goal 32768.")
    ap.add_argument("--ar", type=int, default=1,
                    help="Tokens per step. 1 = decode graph; >1 = prefill.")
    ap.add_argument("--device", default=DEVICE)
    ap.add_argument("--name", default=None, help="AI Hub job name.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print input_specs and exit — submit nothing.")
    args = ap.parse_args()

    if not args.onnx.exists():
        print(f"FATAL: not found: {args.onnx}", file=sys.stderr)
        return 2

    # Accept either a .onnx file or a qai-hub model directory. The graph
    # (for input_specs) is read from the .onnx; the upload target is the
    # directory when given one, so external weights ride along.
    if args.onnx.is_dir():
        onnx_files = sorted(args.onnx.glob("*.onnx"))
        if len(onnx_files) != 1:
            print(f"FATAL: model dir must hold exactly one .onnx, found "
                  f"{len(onnx_files)}", file=sys.stderr)
            return 2
        graph_path = onnx_files[0]
        upload_target = args.onnx
    else:
        graph_path = args.onnx
        upload_target = args.onnx

    specs = build_input_specs(graph_path, args.ctx, args.ar)
    print(f"[input_specs]  ctx={args.ctx}  ar={args.ar}  "
          f"({len(specs)} inputs)")
    for name, (shape, dtype) in specs.items():
        tag = ""
        if "past" in name:
            tag = " <- KV (head_dim 512=global / 256=sliding)"
        elif name == "per_layer_inputs":
            tag = " <- PLE"
        print(f"  {name:<32} {dtype:<8} {list(shape)}{tag}")

    if args.dry_run:
        print("\n[dry-run] nothing submitted.")
        return 0

    name = args.name or f"gemma4-e2b-decoder-fp16-ctx{args.ctx}-ar{args.ar}"
    print(f"\n[submit] uploading {upload_target} ...")
    print(f"         device={args.device!r}  job={name!r}")

    import qai_hub as hub
    job = hub.submit_compile_job(
        model=str(upload_target),
        device=hub.Device(args.device),
        name=name,
        input_specs=specs,
        options="--target_runtime qnn_context_binary",
    )
    print(f"\n[ok] compile job submitted: {job.job_id}")
    print(f"     {job.url}")
    print("     monitor with: qai-hub list-jobs   (or the URL above)")
    print("     on success -> submit_quantize_job (w8a16) + submit_inference_job")
    return 0


if __name__ == "__main__":
    sys.exit(main())
