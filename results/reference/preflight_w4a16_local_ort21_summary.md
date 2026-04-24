# ORT-QNN 2.1.0 isolated probe against x86 w4a16-local binary — verdict

Ran 2026-04-22 while x86 is in-flight on QAIRT 2.42 SDK install. Goal
was to bypass the 2.42 rebuild if the matched-version 2.45 runtime
could load the binary.

## TL;DR — negative

All three plausible load paths fail on this hardware / driver:

| attempt | approach | outcome |
|---|---|---|
| 1 | legacy `providers=[("QNNExecutionProvider", opts)]` | silent fallback to CPU — 2.x plugin-EP doesn't bind via legacy API (already documented). |
| 2 | plugin-EP + `disable_file_mapped_weights=1` | option not honoured (EP logs "User specified disable_file_mapped_weights: 0" despite us passing "1"); process crashes silent `exit 127` at file-mapping warning, same point as no-flag baseline. |
| 3 | plugin-EP + `embed_mode=1` (binary inlined into wrapper ONNX) | farthest yet — EP logs "Node qnn_ctx is using an embedded cache. Disabling file mapping for this node." then crashes silent `exit 127` during `LoadCachedQnnContextFromBuffer` (one log line past the historical 2.1.0 bug pattern). |

Matches `docs/npu_ort_qnn_version_match.md` §"ORT-QNN 2.1.0 loader
bugs on this Hexagon driver" — the bugs reproduce across two
different 2.45 binaries (the earlier AI-Hub-compiled one and this
x86-locally-compiled one), and across three workaround attempts.

## Isolated venv setup (preserved for future retries)

`.venv-ort21/` at the repo root:

```
onnxruntime              1.25.0
onnxruntime-qnn          2.1.0    (bundles QAIRT 2.45.40.260406130939)
onnx                     1.21.0
numpy                    2.4.4
```

QAIRT DLL path (plugin-EP layout, different from 1.24.4):
`.venv-ort21/Lib/site-packages/onnxruntime_qnn/QnnHtp.dll`

Probe script: `scripts/probe_ort21_w4a16_local.py`. Re-run with
`--skip-legacy --skip-filemap-disable` to skip straight to embed_mode=1
for future version retries (2.1.1+ / 2.2.x when they ship).

## Logs on disk

- `results/preflight_w4a16_local_load.log` — ORT 1.24.4 attempt, error 5000.
- `results/preflight_w4a16_local_ort21.log` — ORT 2.1.0 attempts 1+2
  (interleaved UTF-8 python stdout + UTF-16-LE native stderr).
- `results/preflight_w4a16_local_ort21_filemap{,2}.{stdout,stderr}` —
  disable_file_mapped_weights isolated run.
- `results/preflight_w4a16_local_ort21_embed.{stdout,stderr}` — embed_mode=1
  attempt, the farthest any path reached.

## Decision

Wait for x86's QAIRT 2.42 SDK install and rebuild. ORT-QNN 2.1.0
cannot load this binary on this hardware without an ORT-side fix.
Revisit when a 2.1.1 / 2.2.x ORT-QNN wheel ships on PyPI.
