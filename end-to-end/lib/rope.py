"""Qwen3-family RoPE cos/sin cache.

Standard recipe (no rope_scaling). For Qwen3.5/3.6 with rope_scaling
this needs an extension that folds the scaling factor in.
"""
from __future__ import annotations

import numpy as np


def build_rope_cache(rope_theta: float, head_dim: int, max_pos: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (cos, sin) of shape [max_pos, head_dim], FP32."""
    half = head_dim // 2
    inv_freq = 1.0 / (rope_theta ** (np.arange(0, half, dtype=np.float64) / half))
    pos = np.arange(max_pos, dtype=np.float64)
    freqs = np.outer(pos, inv_freq)            # [max_pos, half]
    emb = np.concatenate([freqs, freqs], -1)   # [max_pos, head_dim]
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)
