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

    # Hybrid (SSM/Mamba2 + attention) families — the `qwen35` arch
    # (Qwen3.6). Dense families leave these at the defaults. The SSM dims
    # are GGUF-confirmed architectural constants (gguf_dump of
    # Qwen3.6-27B-MTP, see docs/qwen3_6_27b_npu_kickoff.md) used as
    # fallbacks when a downstream config.json lacks the keys.
    is_hybrid: bool = False
    full_attention_interval: int = 0   # 1-in-N attention blocks; 0 = dense
    # "linear_attention" (gated-delta-net / Mamba2) dims — real HF qwen3_5
    # config keys (under text_config). 0 for dense families. Fallbacks only;
    # load_model_info reads the live config.json values when present.
    linear_conv_kernel_dim: int = 0
    linear_key_head_dim: int = 0
    linear_value_head_dim: int = 0
    linear_num_key_heads: int = 0
    linear_num_value_heads: int = 0


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
    # Qwen3.6-27B branding ships the `qwen3_5` arch (a vision-language
    # model: arch class Qwen3_5ForConditionalGeneration, LLM dims nested
    # under config.json["text_config"]). Values below verified against the
    # real HF config.json (Qwen/Qwen3.6-27B, transformers 4.57.1).
    "qwen3_6": FamilyConfig(
        name="qwen3_6",
        # No native qwen3_5 adapter in aimet_onnx 2.26; "qwen3" is the
        # closest dense-attention match. AdaScale block detection will only
        # resolve the 16 full_attention blocks — verify with find_blocks
        # before spending GPU time (and note the gated-delta-net "linear"
        # layers are invisible to it). Likely an AI Hub Workbench candidate.
        aimet_adascale_model_type="qwen3",
        pathb_supported=True,          # hybrid pathb IS the work; flag on so the pipeline runs
        rope_scaling_supported=False,  # natively trained to 256k; uses mrope, not rope_scaling
        hf_config_arch_class="Qwen3_5ForConditionalGeneration",
        is_hybrid=True,
        full_attention_interval=4,     # config: 1-in-4 → full_attention at blocks 3,7,…,63
        linear_conv_kernel_dim=4,      # config text_config.linear_conv_kernel_dim
        linear_key_head_dim=128,       # config text_config.linear_key_head_dim
        linear_value_head_dim=128,     # config text_config.linear_value_head_dim
        linear_num_key_heads=16,       # config text_config.linear_num_key_heads
        linear_num_value_heads=48,     # config text_config.linear_num_value_heads
    ),
    # TODO: qwen3_5 (text-only sibling) if/when we target it; same hybrid
    # shape, no vision_config.
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

    # Hybrid (SSM + attention) — resolved from config.json text_config,
    # else the family fallbacks. All 0 / empty for dense models.
    full_attention_interval: int = 0    # 1-in-N attention blocks; 0 = dense
    layer_types: list[str] = field(default_factory=list)  # authoritative per-block list, if present
    nextn_predict_layers: int = 0       # MTP head count (mtp_num_hidden_layers); 0 = none
    partial_rotary_factor: float = 1.0  # fraction of head_dim that gets RoPE (qwen3_5: 0.25)
    rope_parameters: Optional[dict] = None  # mrope_interleaved / mrope_section / rope_type
    # "linear_attention" (gated-delta-net) dims — real qwen3_5 keys.
    linear_conv_kernel_dim: int = 0
    linear_key_head_dim: int = 0
    linear_value_head_dim: int = 0
    linear_num_key_heads: int = 0
    linear_num_value_heads: int = 0

    @property
    def head_count_ratio(self) -> int:
        """num_attention_heads / num_key_value_heads — the GQA fan-out."""
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def is_hybrid(self) -> bool:
        """True for SSM+attention hybrids (qwen35/Qwen3.6)."""
        return self.family.is_hybrid and self.full_attention_interval > 0

    @property
    def block_types(self) -> list[str]:
        """Per-decoder-block mixer type, length == num_hidden_layers.

        Prefer the authoritative `layer_types` list from config.json
        (qwen3_5 enumerates every block as "full_attention" /
        "linear_attention"). Fall back to the 1-in-N interval rule, then
        all-"attention" for dense. The MTP head is NOT included here.
        """
        if self.layer_types:
            return ["attention" if t == "full_attention" else "ssm"
                    for t in self.layer_types]
        if not self.is_hybrid:
            return ["attention"] * self.num_hidden_layers
        n = self.full_attention_interval
        return ["attention" if i % n == n - 1 else "ssm"
                for i in range(self.num_hidden_layers)]

    @property
    def ssm_inner_size(self) -> int:
        """gated-delta-net inner width = num_value_heads x value_head_dim
        (6144 for Qwen3.6-27B). 0 for dense."""
        return self.linear_num_value_heads * self.linear_value_head_dim

    @property
    def rotary_ndims(self) -> int:
        """head_dim slots that actually get RoPE (partial rotary).
        Qwen3.6-27B: 0.25 x 256 = 64. Drives the snag-2 cos/sin hoist."""
        return int(round(self.head_dim * self.partial_rotary_factor))

    @property
    def attention_layer_indices(self) -> list[int]:
        """Indices of the KV-carrying attention layers (drives split.py KV
        I/O and the runtime KV manager — snags 3/5)."""
        return [i for i, t in enumerate(self.block_types) if t == "attention"]

    @property
    def num_attention_layers(self) -> int:
        """Count of KV-carrying layers. Equals num_hidden_layers for dense;
        ~num_hidden_layers / full_attention_interval for hybrids."""
        return len(self.attention_layer_indices)

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
    # qwen3_5 VLMs nest the language-model dims under "text_config";
    # dense text models keep them top-level. Use text_config when present.
    tcfg = cfg.get("text_config", cfg)
    hidden_size = int(tcfg["hidden_size"])
    num_hidden_layers = int(tcfg["num_hidden_layers"])
    num_attention_heads = int(tcfg["num_attention_heads"])
    num_key_value_heads = int(tcfg.get("num_key_value_heads", num_attention_heads))
    head_dim = int(tcfg.get("head_dim", hidden_size // num_attention_heads))
    rope_parameters = tcfg.get("rope_parameters") or None
    rope_theta = float((rope_parameters or {}).get(
        "rope_theta", tcfg.get("rope_theta", 10000.0)))
    rope_scaling = tcfg.get("rope_scaling") or None
    partial_rotary_factor = float(tcfg.get("partial_rotary_factor", 1.0))

    if rope_scaling is not None and not family.rope_scaling_supported:
        raise NotImplementedError(
            f"model has rope_scaling={rope_scaling} but family {family.name!r} "
            "is not yet flagged as supporting rope_scaling. The pathb rotary "
            "hoist (rewrite_qwen3_pathb.py) asserts identity attention_scaling; "
            "to support rope_scaling we'd need to fold the scaling factor into "
            "the externally-computed cos/sin cache (see lib/rope.py). Mark "
            "family.rope_scaling_supported=True after that work lands."
        )

    # Hybrid ("linear_attention" / gated-delta-net) dims — read from
    # text_config; family fallbacks for any missing key. layer_types is the
    # authoritative per-block map when present. Verified against the real
    # Qwen/Qwen3.6-27B config.json (transformers 4.57.1).
    layer_types = list(tcfg.get("layer_types") or [])
    full_attention_interval = int(
        tcfg.get("full_attention_interval", family.full_attention_interval))
    nextn_predict_layers = int(
        tcfg.get("mtp_num_hidden_layers",
                 tcfg.get("num_nextn_predict_layers", 0)))
    linear_conv_kernel_dim = int(tcfg.get("linear_conv_kernel_dim", family.linear_conv_kernel_dim))
    linear_key_head_dim = int(tcfg.get("linear_key_head_dim", family.linear_key_head_dim))
    linear_value_head_dim = int(tcfg.get("linear_value_head_dim", family.linear_value_head_dim))
    linear_num_key_heads = int(tcfg.get("linear_num_key_heads", family.linear_num_key_heads))
    linear_num_value_heads = int(tcfg.get("linear_num_value_heads", family.linear_num_value_heads))

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
        vocab_size=int(tcfg.get("vocab_size", 0)),
        max_position_embeddings=int(tcfg.get("max_position_embeddings", 0)),
        torch_dtype=str(tcfg.get("torch_dtype", tcfg.get("dtype", "bfloat16"))),
        precision=precision,
        full_attention_interval=full_attention_interval,
        layer_types=layer_types,
        nextn_predict_layers=nextn_predict_layers,
        partial_rotary_factor=partial_rotary_factor,
        rope_parameters=rope_parameters,
        linear_conv_kernel_dim=linear_conv_kernel_dim,
        linear_key_head_dim=linear_key_head_dim,
        linear_value_head_dim=linear_value_head_dim,
        linear_num_key_heads=linear_num_key_heads,
        linear_num_value_heads=linear_num_value_heads,
    )
    return info


def summary_str(info: ModelInfo) -> str:
    """Multi-line description for run logs."""
    lines = [
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
    ]
    if info.is_hybrid:
        ssm_count = info.num_hidden_layers - info.num_attention_layers
        mrope = (info.rope_parameters or {}).get("mrope_section")
        lines += [
            f"  hybrid            : yes (linear_attention + full_attention, qwen3_5)",
            f"  full_attn_interval: {info.full_attention_interval} (1-in-N attention)",
            f"  attn layers       : {info.num_attention_layers} (KV-carrying) {info.attention_layer_indices[:4]}...",
            f"  ssm layers        : {ssm_count} (gated-delta-net, O(1) state, no KV)",
            f"  mtp head layers   : {info.nextn_predict_layers}",
            f"  partial_rotary    : {info.partial_rotary_factor} -> rotary_ndims={info.rotary_ndims} of {info.head_dim}",
            f"  mrope_section     : {mrope}",
            f"  linear dims       : conv{info.linear_conv_kernel_dim} "
            f"kheads{info.linear_num_key_heads}x{info.linear_key_head_dim} "
            f"vheads{info.linear_num_value_heads}x{info.linear_value_head_dim} "
            f"inner{info.ssm_inner_size}",
        ]
    return "\n".join(lines)
