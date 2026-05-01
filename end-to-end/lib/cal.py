"""Calibration data generator for the pathb-pinned ONNX.

Walks calibration prompts through a FP onnxruntime session, yielding
real (input_ids, position_ids, attention_bias, KV) tuples that match
the pinned graph's AR=1 ctx=N shapes.

Pathb cache layout: current token lands at slot (ctx-1); real past
KVs grow backward from slot (ctx-2). attention_bias masks the unused
zero-init slots at the front of the window.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import onnxruntime as ort


# Inline cal prompt bank — mix of code, prose, dialogue, structured.
# Kept in repo so we never depend on HF datasets at run time.
DEFAULT_PROMPTS = [
    "The quick brown fox jumps over the lazy dog. The dog sleeps in the sun.",
    "def fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "Q: What is the capital of France?\nA: The capital of France is Paris.",
    "import numpy as np\narr = np.zeros((3, 4), dtype=np.float32)\nprint(arr.shape)",
    "In machine learning, gradient descent is an iterative optimization algorithm.",
    "The user said: hello, can you help me debug this Python script please?",
    "<html>\n  <body>\n    <h1>Welcome</h1>\n    <p>Hello world</p>\n  </body>\n</html>",
    "{\"name\": \"alice\", \"age\": 30, \"items\": [\"apple\", \"banana\", \"cherry\"]}",
    "She walked through the silent forest, listening to the wind in the leaves.",
    "Theorem: Every prime number greater than 2 is odd. Proof: suppose p is prime and even.",
    "git commit -m 'fix: handle empty input gracefully'\ngit push origin main",
    "The matrix product AB is defined when the columns of A equal the rows of B.",
    "User: How do I sort a list?\nAssistant: Use the sorted() built-in or list.sort() in place.",
    "After the rain stopped, a rainbow appeared above the distant mountains.",
    "TODO: refactor authentication to use JWT instead of session cookies for the API",
    "She opened the letter and read the first line three times before continuing.",
    "class Trie:\n    def __init__(self):\n        self.children = {}\n        self.end = False",
    "The reaction proceeds via an SN2 mechanism with inversion of stereochemistry.",
    "On a bright cold day in April, the clocks were striking thirteen.",
    "SELECT user_id, COUNT(*) FROM events WHERE created_at > NOW() - INTERVAL '1 day' GROUP BY user_id",
    "The protein folds into its native conformation through a series of intermediate states.",
    "She looked at the painting for a long time without saying anything.",
    "fn main() {\n    let v: Vec<i32> = (1..=10).collect();\n    println!(\"{:?}\", v);\n}",
    "The supply chain disruption affected delivery times across multiple sectors of the economy.",
    "He poured the tea slowly, watching the steam rise into the cool morning air.",
    "interface User {\n  id: string;\n  name: string;\n  email?: string;\n}",
    "The asymptotic complexity of mergesort is O(n log n) in both time and space.",
    "When the wind blew through the valley, the trees swayed and the leaves whispered.",
    "ALTER TABLE users ADD COLUMN last_login TIMESTAMP DEFAULT CURRENT_TIMESTAMP;",
    "If a function f is continuous on [a,b], then by the IVT it attains every value between f(a) and f(b).",
    "She stepped onto the train just as the doors closed behind her, holding her breath.",
    "User: Why is the sky blue?\nAssistant: Rayleigh scattering — shorter wavelengths scatter more.",
]


def _kv_input_names(sess: ort.InferenceSession) -> list[str]:
    return [i.name for i in sess.get_inputs() if i.name.startswith("past_key_values.")]


def _present_output_names(sess: ort.InferenceSession) -> list[str]:
    return [o.name for o in sess.get_outputs() if o.name.startswith("present.")]


def _kv_input_shapes(sess: ort.InferenceSession) -> dict[str, list[int]]:
    return {
        i.name: [d if isinstance(d, int) else 1 for d in i.shape]
        for i in sess.get_inputs()
        if i.name.startswith("past_key_values.")
    }


def cal_iter(
    sess: ort.InferenceSession,
    tokenizer,
    rope_cos: np.ndarray,
    rope_sin: np.ndarray,
    ctx: int,
    max_samples: int,
    prompts: list[str] | None = None,
) -> Iterator[dict[str, np.ndarray]]:
    """Yield AR=1 decode-step input dicts. Each dict feeds the pinned graph
    end-to-end (input_ids, position_ids, attention_bias, RoPE cos/sin,
    all past_key_values.* tensors) at consistent shapes.

    Mask convention matches the pathb cache layout: the CURRENT token
    sits at slot (ctx-1) of present, real past KVs grow backward, and
    everything before slot (ctx-1-pos) is masked with -65504 (FP16 -inf).
    """
    prompts = prompts or DEFAULT_PROMPTS
    past_names = _kv_input_names(sess)
    present_names = _present_output_names(sess)
    assert len(past_names) == len(present_names), (
        f"input/output KV mismatch: {len(past_names)} vs {len(present_names)}"
    )
    kv_shape = _kv_input_shapes(sess)
    yielded = 0

    for prompt in prompts:
        if yielded >= max_samples:
            return
        ids = tokenizer(prompt, return_tensors="np").input_ids[0].tolist()
        if len(ids) > ctx - 1:
            ids = ids[: ctx - 1]
        past = {n: np.zeros(kv_shape[n], dtype=np.float32) for n in past_names}
        for pos, tok in enumerate(ids):
            if pos >= ctx - 1 or yielded >= max_samples:
                break
            input_ids = np.array([[tok]], dtype=np.int64)
            position_ids = np.array([[pos]], dtype=np.int64)
            # Mask everything before the live window.
            attn_bias = np.full((1, 1, 1, ctx), -65504.0, dtype=np.float32)
            attn_bias[..., ctx - 1 - pos:] = 0.0
            cos_step = rope_cos[pos: pos + 1][None, ...].astype(np.float32)
            sin_step = rope_sin[pos: pos + 1][None, ...].astype(np.float32)
            feeds: dict[str, np.ndarray] = {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "attention_bias": attn_bias,
                "position_ids_cos": cos_step,
                "position_ids_sin": sin_step,
            }
            feeds.update(past)
            yield feeds
            yielded += 1
            outs = sess.run(present_names, feeds)
            past_len = ctx - 1
            past = {
                p_n: outs[i][..., -past_len:, :].astype(np.float32, copy=False)
                for i, p_n in enumerate(past_names)
            }
