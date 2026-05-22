"""Model-family-aware introspection for the Gemma e2e pipeline.

Parallel to `end-to-end/lib/model_config.py` (the Qwen3 pipeline), but
Gemma 4 keeps its decoder attributes one level down in a `text_config`
block because the top-level architecture is the multimodal wrapper
`Gemma4ForConditionalGeneration`. This module flattens that nesting and
surfaces the Gemma-specific fields the rest of the pipeline needs:

  - dual RoPE (separate theta + rotary factor for sliding vs full layers)
  - per-layer attention type map (`layer_types`)
  - Per-Layer Embeddings (PLE): a second embedding table
  - shared-KV layers (`num_kv_shared_layers`)
  - final logit soft-capping

None of these exist in Qwen3, so the Qwen3 `ModelInfo` cannot be reused
as-is. See `gemma-pipeline/ARCHITECTURE_NOTES.md` for why each field
matters to ONNX export / pathb rewrite / AIMET / QAIRT.

This module is pure stdlib + dataclasses — it runs anywhere (x86, ARM,
cloud) with no torch / onnx / CUDA dependency. It is the one piece of
the Gemma pipeline that is fully testable on the current x86 dev box.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class FamilyConfig:
    """Static, per-family overrides — what does not come from config.json.

    `aimet_adascale_model_type` is the string aimet_onnx 2.26 accepts in
    `AdaScaleModelConfig(model_type=...)`. As of 2.26 the known set is
    {qwen2, qwen3, llama, mistral, phi3} — there is NO native `gemma`
    adapter. `gemma4` therefore falls back to the closest structural
    match and AdaScale block detection must be verified with the
    find_blocks debug script before spending GPU time (see
    ARCHITECTURE_NOTES.md §AIMET).
    """
    name: str
    aimet_adascale_model_type: str
    pathb_supported: bool
    rope_scaling_supported: bool = False
    hf_config_arch_class: str = ""
    # Gemma-specific structural flags (all absent in Qwen3).
    has_per_layer_embeddings: bool = False
    has_sliding_window: bool = False
    has_shared_kv: bool = False
    has_logit_softcap: bool = False
    is_multimodal_wrapper: bool = False


FAMILY_CONFIGS: dict[str, FamilyConfig] = {
    # Gemma 4 — the target of this pipeline. `pathb_supported=False`
    # until the Gemma-specific rewrites land (scripts/README.md): the
    # Qwen3 pathb scripts do NOT transfer (different rotary, dual RoPE,
    # PLE, shared KV). The flag flips to True when those exist.
    "gemma4": FamilyConfig(
        name="gemma4",
        aimet_adascale_model_type="llama",  # closest available adapter; VERIFY
        pathb_supported=False,
        rope_scaling_supported=True,        # dual-RoPE / proportional global
        hf_config_arch_class="Gemma4ForConditionalGeneration",
        has_per_layer_embeddings=True,
        has_sliding_window=True,
        has_shared_kv=True,
        has_logit_softcap=True,
        is_multimodal_wrapper=True,
    ),
    # Gemma 3 — kept as a reference point; not a pipeline target.
    "gemma3": FamilyConfig(
        name="gemma3",
        aimet_adascale_model_type="llama",
        pathb_supported=False,
        rope_scaling_supported=True,
        hf_config_arch_class="Gemma3ForConditionalGeneration",
        has_per_layer_embeddings=False,
        has_sliding_window=True,
        has_shared_kv=False,
        has_logit_softcap=False,
        is_multimodal_wrapper=True,
    ),
}


@dataclass
class ModelInfo:
    """Normalized Gemma model attributes consumed by the pipeline."""
    # From the user.
    model_id: str
    model_path: Path
    family: FamilyConfig

    # Derived from config.json text_config.
    architecture: str
    text_model_type: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int
    rms_norm_eps: float
    torch_dtype: str = "bfloat16"

    # Gemma-specific (no Qwen3 equivalent).
    global_head_dim: int = 0                 # full-attention layers may differ
    sliding_window: int = 0
    layer_types: list[str] = field(default_factory=list)  # per-layer attn type
    num_kv_shared_layers: int = 0
    hidden_size_per_layer_input: int = 0     # PLE residual width; 0 = no PLE
    final_logit_softcapping: Optional[float] = None
    rope_theta_full: float = 1_000_000.0
    rope_theta_sliding: float = 10_000.0
    rope_partial_rotary_factor_full: float = 1.0   # 0.25 on Gemma4 global
    enable_moe_block: bool = False

    # Set externally from the --precision arg.
    precision: str = ""

    @property
    def head_count_ratio(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def model_basename(self) -> str:
        return self.model_id.split("/")[-1]

    @property
    def bundle_stem(self) -> str:
        m = self.model_basename.lower().replace(".", "p")
        return f"{m}-{self.precision}-pathb"

    @property
    def num_global_layers(self) -> int:
        return sum(1 for t in self.layer_types if t == "full_attention")

    @property
    def num_sliding_layers(self) -> int:
        return sum(1 for t in self.layer_types if t == "sliding_attention")

    @property
    def num_kv_owning_layers(self) -> int:
        """Layers that actually carry their own KV-cache I/O.

        With KV sharing, the last `num_kv_shared_layers` reuse an earlier
        layer's K/V — they do NOT add KV graph inputs/outputs. This count
        drives the KV-cache size estimate and the partition seam map.
        """
        return self.num_hidden_layers - self.num_kv_shared_layers


def _text_cfg(cfg: dict) -> dict:
    """Return the decoder config, whether nested under text_config or flat."""
    return cfg.get("text_config", cfg)


def resolve_family(model_id: str, model_path: Path,
                   family_override: Optional[str] = None) -> FamilyConfig:
    if family_override:
        if family_override not in FAMILY_CONFIGS:
            raise ValueError(
                f"unknown model family: {family_override!r}. "
                f"known: {sorted(FAMILY_CONFIGS)}"
            )
        return FAMILY_CONFIGS[family_override]

    cfg_path = model_path / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        archs = cfg.get("architectures") or []
        for fam in FAMILY_CONFIGS.values():
            if fam.hf_config_arch_class and fam.hf_config_arch_class in archs:
                return fam
        mt = cfg.get("model_type", "")
        for key, fam in FAMILY_CONFIGS.items():
            if key == mt or mt.startswith(key):
                return fam

    needle = model_id.lower().replace("-", "").replace("_", "")
    for key, fam in FAMILY_CONFIGS.items():
        if key in needle:
            return fam

    raise ValueError(
        f"could not infer model family for model_id={model_id!r}. "
        f"Pass --model-family explicitly. Known: {sorted(FAMILY_CONFIGS)}"
    )


def load_model_info(model_id: str, model_path: Path,
                    family_override: Optional[str] = None,
                    precision: str = "") -> ModelInfo:
    """Load + normalize a Gemma HF config.json into a ModelInfo struct."""
    cfg_path = model_path / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"missing config.json at {cfg_path}")
    cfg = json.loads(cfg_path.read_text())
    tc = _text_cfg(cfg)

    family = resolve_family(model_id, model_path, family_override)
    architecture = (cfg.get("architectures") or [""])[0]

    hidden_size = int(tc["hidden_size"])
    num_attention_heads = int(tc["num_attention_heads"])
    num_key_value_heads = int(tc.get("num_key_value_heads", num_attention_heads))
    head_dim = int(tc.get("head_dim", hidden_size // num_attention_heads))

    # Gemma4 dual-RoPE lives under `rope_parameters` with `full_attention`
    # and `sliding_attention` sub-blocks; older configs are flatter.
    rope = tc.get("rope_parameters", {}) or {}
    full = rope.get("full_attention", {}) if isinstance(rope, dict) else {}
    slide = rope.get("sliding_attention", {}) if isinstance(rope, dict) else {}
    rope_theta_full = float(full.get("rope_theta", tc.get("rope_theta", 1_000_000.0)))
    rope_theta_sliding = float(slide.get("rope_theta", tc.get("rope_local_base_freq", 10_000.0)))
    partial_full = float(full.get("partial_rotary_factor", 1.0))

    layer_types = list(tc.get("layer_types") or [])

    info = ModelInfo(
        model_id=model_id,
        model_path=model_path,
        family=family,
        architecture=architecture,
        text_model_type=str(tc.get("model_type", "")),
        hidden_size=hidden_size,
        num_hidden_layers=int(tc["num_hidden_layers"]),
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        intermediate_size=int(tc.get("intermediate_size", 0)),
        vocab_size=int(tc.get("vocab_size", 0)),
        max_position_embeddings=int(tc.get("max_position_embeddings", 0)),
        rms_norm_eps=float(tc.get("rms_norm_eps", 1e-6)),
        torch_dtype=str(cfg.get("dtype", cfg.get("torch_dtype", "bfloat16"))),
        global_head_dim=int(tc.get("global_head_dim", head_dim)),
        sliding_window=int(tc.get("sliding_window", 0)),
        layer_types=layer_types,
        num_kv_shared_layers=int(tc.get("num_kv_shared_layers", 0)),
        hidden_size_per_layer_input=int(tc.get("hidden_size_per_layer_input", 0)),
        final_logit_softcapping=(
            float(tc["final_logit_softcapping"])
            if tc.get("final_logit_softcapping") is not None else None
        ),
        rope_theta_full=rope_theta_full,
        rope_theta_sliding=rope_theta_sliding,
        rope_partial_rotary_factor_full=partial_full,
        enable_moe_block=bool(tc.get("enable_moe_block", False)),
        precision=precision,
    )
    return info


def summary_str(info: ModelInfo) -> str:
    return "\n".join([
        f"  model_id            : {info.model_id}",
        f"  model_path          : {info.model_path}",
        f"  family              : {info.family.name}",
        f"  architecture        : {info.architecture}  (text: {info.text_model_type})",
        f"  precision           : {info.precision}",
        f"  hidden_size         : {info.hidden_size}",
        f"  num_hidden_layers   : {info.num_hidden_layers}",
        f"  num_attn_heads      : {info.num_attention_heads}",
        f"  num_kv_heads        : {info.num_key_value_heads} (GQA fan-out {info.head_count_ratio})",
        f"  head_dim            : {info.head_dim}  (global {info.global_head_dim})",
        f"  intermediate_size   : {info.intermediate_size}",
        f"  vocab_size          : {info.vocab_size}",
        f"  max_position        : {info.max_position_embeddings}",
        f"  sliding_window      : {info.sliding_window}",
        f"  layer_types         : {info.num_sliding_layers} sliding / {info.num_global_layers} global",
        f"  kv_shared_layers    : {info.num_kv_shared_layers}  -> {info.num_kv_owning_layers} KV-owning layers",
        f"  per_layer_emb width : {info.hidden_size_per_layer_input}  ({'PLE present' if info.hidden_size_per_layer_input else 'no PLE'})",
        f"  rope_theta          : full {info.rope_theta_full:g} (partial {info.rope_partial_rotary_factor_full}) / sliding {info.rope_theta_sliding:g}",
        f"  logit_softcap       : {info.final_logit_softcapping}",
        f"  enable_moe_block    : {info.enable_moe_block}",
        f"  aimet_adascale_mt   : {info.family.aimet_adascale_model_type}  (VERIFY — no native gemma adapter)",
        f"  pathb_supported     : {info.family.pathb_supported}",
    ])


if __name__ == "__main__":
    # Smoke test: point at a downloaded Gemma config dir.
    import sys
    if len(sys.argv) < 2:
        print("usage: python -m lib.model_config <model_dir> [model_id]")
        sys.exit(1)
    p = Path(sys.argv[1])
    mid = sys.argv[2] if len(sys.argv) > 2 else f"local/{p.name}"
    print(summary_str(load_model_info(mid, p, precision="w4a16")))
