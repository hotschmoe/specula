# Probe 3: does OpenCL -ngl 0 honor cache_prompt for prefix-cache reuse?
# Start server, send long prompt, send same long prompt + delta, compare PPs.

$ErrorActionPreference = "Continue"

$bin   = "C:\Users\hotschmoe\Documents\GitHub\specula\llama.cpp\build-opencl\bin\llama-server.exe"
$model = "C:\Users\hotschmoe\Documents\GitHub\specula\models\Qwen3.6-35B-A3B-MXFP4_MOE.gguf"
$out   = "C:\Users\hotschmoe\Documents\GitHub\specula\results\csv\probe3_delta_prefill_2026-05-13.md"
$port  = 8191

# Build a ~8000-token base prompt by repeating a programming-context block.
$block = @'
Below is the contents of a Python file in a coding repository. Read it carefully
and be prepared to answer questions about its structure, control flow, and any
bugs or edge cases. The file implements a quantization pipeline that takes a
PyTorch model and exports it to a quantized GGUF format suitable for inference
on edge devices. The pipeline applies SEQ_MSE calibration, runs an importance-
matrix sweep over a small calibration dataset, and finally writes the quantized
tensors out to the GGUF container. Key functions include load_calibration_data,
compute_importance_matrix, apply_seq_mse, repack_weights_for_kleidiai, and
write_gguf_v3. There are several edge cases around mixed-precision quants and
expert-routing tensors in MoE models that the code attempts to handle.

import torch
import numpy as np
from pathlib import Path
from typing import Iterator, Tuple

def load_calibration_data(path, batch_size=4):
    files = sorted(path.glob("*.npz"))
    for f in files:
        arr = np.load(f)["data"]
        for i in range(0, arr.shape[0], batch_size):
            yield torch.from_numpy(arr[i:i+batch_size]).float()

def compute_importance_matrix(model, calib_iter, n_samples=64):
    importance = {}
    hooks = []
    for name, mod in model.named_modules():
        if hasattr(mod, "weight") and mod.weight.dim() == 2:
            def make_hook(n):
                def hook(_, inputs, _output):
                    x = inputs[0].detach()
                    importance.setdefault(n, []).append((x * x).sum(dim=0).cpu().numpy())
                return hook
            hooks.append(mod.register_forward_hook(make_hook(name)))
    with torch.no_grad():
        for i, batch in enumerate(calib_iter):
            if i >= n_samples: break
            model(batch)
    for h in hooks:
        h.remove()
    return {k: np.stack(v).mean(axis=0) for k, v in importance.items()}
'@
# Use a unique signature in each request to invalidate prompt-cache if we want to
# (we don't here, but if we did, we'd append it).
$basePrompt = ($block * 12)  # ~8k tokens of repeated context
# Sanity: count tokens roughly. Each block has ~700 tokens => 12x ~= 8400
$delta = "`n`nNow tell me, in one sentence, what does compute_importance_matrix return?"
$turn1Prompt = "$basePrompt`n`nQuestion: list every function defined above."
$turn2Prompt = "$basePrompt$delta"  # SAME base prefix, different tail

"# Probe 3: delta-prefill behavior via cache_prompt on OpenCL -ngl 0" | Out-File $out -Encoding utf8
"Build: $bin" | Out-File $out -Append -Encoding utf8
"Model: 35B-A3B MXFP4_MOE, -ngl 0 -t 18, -c 16384" | Out-File $out -Append -Encoding utf8

# Start server
$srvOut = "C:\Users\hotschmoe\AppData\Local\Temp\probe3_srv_stdout.log"
$srvErr = "C:\Users\hotschmoe\AppData\Local\Temp\probe3_srv_stderr.log"
$p = Start-Process -FilePath $bin -ArgumentList @(
  "-m",$model,
  "-c","16384","-ngl","0","-t","18",
  "--port","$port","--host","127.0.0.1","--no-warmup"
) -RedirectStandardOutput $srvOut -RedirectStandardError $srvErr -NoNewWindow -PassThru

Write-Output "Started server PID $($p.Id), waiting for ready..."
$ready = $false
for ($i=0; $i -lt 120; $i++) {
  Start-Sleep -Seconds 3
  try {
    $h = Invoke-WebRequest -Uri "http://127.0.0.1:$port/health" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
    if ($h.StatusCode -eq 200) { $ready = $true; Write-Output "Ready after $i polls"; break }
  } catch {}
}
if (-not $ready) {
  Write-Output "Server not ready"
  Get-Content $srvErr -Tail 30
  Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
  exit 1
}

function Run-Request($promptText, $label) {
  $body = @{
    prompt = $promptText
    n_predict = 64
    temperature = 0
    cache_prompt = $true
    stream = $false
  } | ConvertTo-Json
  $t0 = Get-Date
  $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$($script:port)/completion" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 900
  $wall = (Get-Date) - $t0
  $row = "| $label | $($resp.timings.prompt_n) | $([math]::Round($resp.timings.prompt_ms,0)) | $([math]::Round($resp.timings.prompt_per_second,1)) | $([math]::Round($resp.timings.predicted_ms,0)) | $([math]::Round($resp.timings.predicted_per_second,2)) | $([math]::Round($wall.TotalSeconds,1)) | $($resp.timings.cache_n) |"
  $row | Out-File $script:out -Append -Encoding utf8
  Write-Output $row
}

"" | Out-File $out -Append -Encoding utf8
"| turn | prompt_n | prompt_ms | PP t/s | predicted_ms | TG t/s | wall s | cache_n |" | Out-File $out -Append -Encoding utf8
"|---|---:|---:|---:|---:|---:|---:|---:|" | Out-File $out -Append -Encoding utf8

Write-Output "--- Turn 1: full base prompt, no cache ---"
Run-Request $turn1Prompt "turn1_cold"

Write-Output "--- Turn 2: same base prefix + delta tail (should hit cache) ---"
Run-Request $turn2Prompt "turn2_cached"

Write-Output "--- Turn 3: rerun turn 2 verbatim (should be max cache hit) ---"
Run-Request $turn2Prompt "turn3_rerun"

Write-Output "--- Killing server ---"
Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
"---DONE---"