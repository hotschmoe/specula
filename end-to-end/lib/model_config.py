"""Model-family-aware introspection for the e2e pipeline.

Read HF `config.json` once, normalize into a `ModelInfo` struct that
the rest of the pipeline consumes. Per-family overrides live in the
`FAMILY_CONFIGS` map below — extending to a new model family is
adding one entry, not branching every consumer.

A "family" here is what aimet_onnx's AdaScale calls a model_type:
{qwen3, qwen2, llama, mistral, phi3} as of aimet_onnx 2.26. Plus we
add `qwen3_5` / `qwen3_6` slots for the future Qwen generations
(known to need rope_scaling support; see TODO below).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class FamilyConfig:
    """Static, per-family overrides — what doesn't come from config.json.

    `aimet_adascale_model_type` is the string aimet_onnx 2.26 accepts
    in `AdaScaleModelConfig(model_type=...)`. Currently
    {qwen2, qwen3, llama, mistral, phi3}. Newer Qwen generations
    don't have a native adapter; for them `aimet_adascale_model_type`
    falls back to "qwen3" (closest match) and we accept that AdaScale
    block detection may be off — verify with the find_blocks debug
    script before committing GPU time.
    """
    name: str
    aimet_adascale_model_type: str
    pathb_supported: bool
    rope_scaling_supported: bool = False  # Qwen3.5+ have rope_scaling
    hf_config_arch_class: str = ""  # e.g. "Qwen3ForCausalLM"


FAMILY_CONFIGS: dict[str, FamilyConfig] = {
    "qwen3": FamilyConfig(
        name="qwen3",
        aimet_adascale_model_type="qwen3",
        pathb_supported=True,
        rope_scaling_supported=False,
        hf_config_arch_class="Qwen3ForCausalLM",
    ),
    "qwen2": FamilyConfig(
        name="qwen2",
        aimet_adascale_model_type="qwen2",
        pathb_supported=False,  # pathb scripts are Qwen3-specific
        rope_scaling_supported=False,
        hf_config_arch_class="Qwen2ForCausalLM",
    ),
    "qwen2_5": FamilyConfig(
        name="qwen2_5",
        aimet_adascale_model_type="qwen2",  # closest aimet adapter
        pathb_supported=False,
        rope_scaling_supported=False,
        hf_config_arch_class="Qwen2ForCausalLM",
    ),
    "llama": FamilyConfig(
        name="llama",
        aimet_adascale_model_type="llama",
        pathb_supported=False,
        rope_scaling_supported=True,  # Llama3+ have rope_scaling
        hf_config_arch_class="LlamaForCausalLM",
    ),
    # TODO: qwen3_5/qwen3_6 once we test rope_scaling pathb support.
    # The pathb rotary hoist asserts Constant_7/8 == 1.0 (identity
    # attention_scaling); Qwen3.5 may have non-identity scaling that
    # we'd need to fold into the externally-computed cos/sin.
}


@dataclass
class ModelInfo:
    """Normalized model attributes consumed by the e2e pipeline."""
    # From the user — required.
    model_id: str           # HF id, e.g. "Qwen/Qwen3-4B"
    model_path: Path        # local HF dir
    family: FamilyConfig    # resolved per-family config

    # Derived from config.json.
    architecture: str       # config.architectures[0]
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rope_theta: float
    rope_scaling: Optional[dict] = None
    vocab_size: int = 0
    max_position_embeddings: int = 0
    torch_dtype: str = "bfloat16"

    # Derived from precision arg (set externally).
    precision: str = ""             # "w8a16" | "w4a16"

    @property
    def head_count_ratio(self) -> int:
        """num_attention_heads / num_key_value_heads — the GQA fan-out."""
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def model_basename(self) -> str:
        return self.model_id.split("/")[-1]

    @property
    def bundle_stem(self) -> str:
        m = self.model_basename.lower().replace(".", "p")
        return f"{m}-{self.precision}-pathb"


def resolve_family(model_id: str, model_path: Path,
                   family_override: Optional[str] = None) -> FamilyConfig:
    """Pick a FamilyConfig for the model.

    Order of resolution:
      1. explicit `family_override` (passed via --model-family)
      2. inferred from HF config.json `architectures[0]`
      3. inferred from model_id casing (case-insensitive substring)
    """
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

    # Fall back to model id substring match.
    needle = model_id.lower()
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
    """Load + normalize HF config.json into a ModelInfo struct."""
    cfg_path = model_path / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"missing config.json at {cfg_path}")
    cfg = json.loads(cfg_path.read_text())

    family = resolve_family(model_id, model_path, family_override)

    architecture = (cfg.get("architectures") or [""])[0]
    hidden_size = int(cfg["hidden_size"])
    num_hidden_layers = int(cfg["num_hidden_layers"])
    num_attention_heads = int(cfg["num_attention_heads"])
    num_key_value_heads = int(cfg.get("num_key_value_heads", num_attention_heads))
    head_dim = int(cfg.get("head_dim", hidden_size // num_attention_heads))
    rope_theta = float(cfg.get("rope_theta", 10000.0))
    rope_scaling = cfg.get("rope_scaling") or None

    if rope_scaling is not None and not family.rope_scaling_supported:
        raise NotImplementedError(
            f"model has rope_scaling={rope_scaling} but family {family.name!r} "
            "is not yet flagged as supporting rope_scaling. The pathb rotary "
            "hoist (rewrite_qwen3_pathb.py) asserts identity attention_scaling; "
            "to support rope_scaling we'd need to fold the scaling factor into "
            "the externally-computed cos/sin cache (see lib/rope.py). Mark "
            "family.rope_scaling_supported=True after that work lands."
        )

    info = ModelInfo(
        model_id=model_id,
        model_path=model_path,
        family=family,
        architecture=architecture,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        rope_theta=rope_theta,
        rope_scaling=rope_scaling,
        vocab_size=int(cfg.get("vocab_size", 0)),
        max_position_embeddings=int(cfg.get("max_position_embeddings", 0)),
        torch_dtype=str(cfg.get("torch_dtype", "bfloat16")),
        precision=precision,
    )
    return info


def summary_str(info: ModelInfo) -> str:
    """Multi-line description for run logs."""
    return "\n".join([
        f"  model_id          : {info.model_id}",
        f"  model_path        : {info.model_path}",
        f"  family            : {info.family.name}",
        f"  architecture      : {info.architecture}",
        f"  precision         : {info.precision}",
        f"  hidden_size       : {info.hidden_size}",
        f"  num_hidden_layers : {info.num_hidden_layers}",
        f"  num_attn_heads    : {info.num_attention_heads}",
        f"  num_kv_heads      : {info.num_key_value_heads} (GQA fan-out {info.head_count_ratio})",
        f"  head_dim          : {info.head_dim}",
        f"  rope_theta        : {info.rope_theta}",
        f"  rope_scaling      : {info.rope_scaling}",
        f"  vocab_size        : {info.vocab_size}",
        f"  max_position      : {info.max_position_embeddings}",
        f"  aimet_adascale_mt : {info.family.aimet_adascale_model_type}",
        f"  pathb_supported   : {info.family.pathb_supported}",
    ])
