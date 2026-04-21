# ORT-QNN ↔ AI Hub QAIRT version matching

Added 2026-04-21 (session 9) when step 5 tripped on a QAIRT version
mismatch despite step 4 compiling cleanly. Keep this note live — every
AI Hub compile pipeline depends on the runtime matching the compiler.

## TL;DR

Qualcomm AI Hub compiles Hexagon context binaries against a specific
QAIRT SDK version. `onnxruntime-qnn` from PyPI bundles *its own* copy
of QAIRT. **Those two versions must match exactly, or the loader throws
`QnnBackendManager::LoadCachedQnnContextFromBuffer Failed to create
context from binary. Error code: 5000`.** The binary compiles fine, the
wrapper ONNX parses fine, the device is recognized, and it still dies
inside the QNN context loader because the serialized context format is
version-gated.

Current matched stack (2026-04-21):

| component | version | bundles / emits |
|---|---|---|
| AI Hub compile | (server-side) | QAIRT **2.45.40.260406** |
| `onnxruntime-qnn` | **2.1.0** (from PyPI) | QAIRT 2.45.40.260406 |
| QAIRT local install | 2.45.40.260406 | — (not on the load path) |

`onnxruntime-qnn` **1.24.4** (the version `docs/npu_scoping.md` §4
originally pinned) bundles QAIRT **2.42.0** and will reject every
binary produced by the current AI Hub pipeline. Do not downgrade below
2.1.0 without also pinning the AI Hub compile to `--qairt_version 2.42`.

## How we diagnosed it

1. `models/qwen3_0_6b_draft_v81_ctx512.bin` compiled cleanly (job
   `jgzx6xlz5`, SUCCESS at t=400s).
2. `scripts/npu_load_qwen3_bin.py` built an `EPContext` wrapper ONNX
   and handed it to `InferenceSession`; load failed with
   `QNN error code 5000` from `LoadCachedQnnContextFromBuffer`.
3. Pulled the AI Hub job log via
   `hub.get_job('jgzx6xlz5').download_job_logs('results/ai_hub_jgzx6xlz5/')`.
   First INFO line: `/qairt_sdk/default/2.45.0/bin/...` — compile was
   2.45.0.
4. Scanned the bytes of `.venv/Lib/site-packages/onnxruntime/capi/QnnHtp.dll`
   for version strings:
   ```
   QnnHtp.dll (1.24.4): ['2.42.0', '2.42.0.251225135753', '3.3.3']
   ```
   — ORT-QNN 1.24.4 was bundling 2.42.
5. Peeked inside the `onnxruntime-qnn 2.1.0` wheel on PyPI:
   ```
   QnnHtp.dll (2.1.0): ['2.45.40', '2.45.40.260406130939']
   ```
   Matched AI Hub exactly. Also ships `Genie.dll` (not present in 1.24.4).

## How to check this in a fresh session

```powershell
# 1. What did AI Hub compile against?
.venv\Scripts\python.exe -c "import qai_hub as hub; \
  hub.get_job('<JOB_ID>').download_job_logs('results/ai_hub_<JOB_ID>/')"
Select-String "/qairt_sdk/default/" "results\ai_hub_<JOB_ID>\<JOB_ID>.log" | Select -First 1

# 2. What does ORT-QNN's bundled runtime report?
.venv\Scripts\python.exe -c @"
from pathlib import Path
import re, onnxruntime as ort
p = Path(ort.__file__).parent / 'capi' / 'QnnHtp.dll'
blob = p.read_bytes()
vers = sorted({m.decode() for m in re.findall(rb'[0-9]+\\.[0-9]+\\.[0-9]+(?:\\.[0-9]+)?', blob) if m.startswith(b'2.')})
print('ORT-QNN bundles QAIRT:', vers[:5])
"@
```

If the two values disagree, a 5000 error on binary load is almost
guaranteed. Fix by upgrading/downgrading `onnxruntime-qnn` until the
bundled QAIRT matches the `/qairt_sdk/default/<version>/` in the log.

## Which onnxruntime-qnn bundles which QAIRT

Verified from PyPI wheels (cp312 win_arm64):

| onnxruntime-qnn | bundled QAIRT | ships Genie.dll? |
|-----------------|---------------|---|
| 1.22.2 | not checked | — |
| 1.23.x | not checked | — |
| 1.24.x | **2.42.0.251225135753** | no |
| 2.0.0  | not checked | not checked |
| **2.1.0** (latest as of 2026-04-21) | **2.45.40.260406130939** | **yes** |

When the next QAIRT ticks (e.g. AI Hub moves to 2.46), a new ORT-QNN
release that bundles it is usually available within weeks on PyPI.
Re-run the probe above to confirm before compiling.

## If pinning ORT-QNN isn't an option

Alternate route: pass `--qairt_version 2.42` (or whatever version your
ORT-QNN bundles) to `qai_hub.submit_compile_job(options=...)`. AI Hub
honours older QAIRT versions for backwards compatibility. This keeps
Python side pinned at 1.24.x while still producing a loadable binary.
Downside: older QAIRT may reject ops the 2.45 converter accepts — the
nomask ONNX happens to be simple enough to compile on 2.42, but no
promise.

## ORT-QNN 2.1.0 loader bugs on this Hexagon driver (session 9 finding)

**Heads up: 2.1.0 matches the QAIRT version but the loader itself is
broken** for context-binary models on the X2E94100 driver shipped with
this machine. Verified independently:

- The compiled `.bin` is healthy. QAIRT 2.45's `qnn-context-binary-utility.exe`
  loads it cleanly and emits the inspection JSON.
- The plugin-EP machinery in 2.1.0 works for ordinary ONNX models — the
  tiny matmul smoke test runs end-to-end on the HTP via the new
  `add_provider_for_devices` API (legacy `providers=[("QNN..."]`
  silently falls back to CPU because 2.x is plugin-style, not built-in).
- Wrapping the .bin in an `EPContext` ONNX (`embed_mode=0`,
  `ep_cache_context=<bin filename>`) reproducibly hits:

      qnn_backend_manager.cc:1600 LoadCachedQnnContextFromBuffer:
      Failed to create context with file mapping enabled.
      Error: QNN_COMMON_ERROR_NOT_SUPPORTED, Code 1000.
      Retrying with feature disabled.

  ORT then crashes the Python interpreter (exit 127, no traceback,
  no further stderr). The retry-without-mapping codepath is the bug.
  Setting `enable_htp_shared_memory_allocator=1` doesn't avoid it.
- Switching to `embed_mode=1` (binary serialized inline as a 1.43 GB
  raw-bytes string attribute) builds a 1.43 GB wrapper ONNX, but
  `InferenceSession()` crashes silently before any ORT log line. A
  separate failure mode in 2.1.0's load path.

So while 2.1.0 is the only PyPI release that bundles QAIRT 2.45 today,
it's not actually usable for loading AI Hub binaries on this hardware.

**Working configuration to fall back to:**

- `onnxruntime-qnn==1.24.4` (built-in EP, legacy `providers=[(name, opts)]`).
- AI Hub compile with `options="... --qairt_version 2.42"` so the binary
  matches 1.24.4's bundled QAIRT 2.42.

This keeps us on the proven session-2 toolchain (the tiny matmul smoke
test ran fine on 1.24.4) and just costs one extra `--reuse-upload`
recompile per .bin.

When ORT-QNN 2.1.1 / 2.2.x ships and the loader bugs are fixed, retry
the 2.45 path — the Python wins (Genie.dll bundled, plugin-EP API
generalises to GPU/CPU) are real once the loader is stable.

## Why this matters beyond step 5

Every subsequent recompile (step 6 if logits are wrong and we need to
drop to W8A16, step 10 sweeping k values with different compile
options) must keep this pair in sync. When in doubt, run the version
probe first — one 60-second check beats one 20-minute compile that
produces an unloadable binary.
