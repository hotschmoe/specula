<#
.SYNOPSIS
    Start a llama-server for Qwen3-14B-Q4_K_M as the SQ1 heterogeneous-
    demo target.

.DESCRIPTION
    Forked from serve_daily_driver.ps1 — the SQ1 demo runs Qwen3-4B
    on the NPU as draft and Qwen3-14B-Q4_K_M on CPU/Vulkan as target.
    This script stands up the target endpoint that the SQ1 driver
    (last_side_quest/sq1_heterogeneous/demo_path_*.py) talks to.

    Default ctx is 32k (Qwen3 native max is 32k without rope-scaling
    overrides). The NPU draft caps at 4k regardless; once the
    target's KV exceeds 4k, the draft falls behind and accept rate
    degrades — that's a real measurement in the demo, not a bug.

    Default backend is CPU (most predictable; matches the 14B
    baseline numbers in docs/qwen3_4b_baseline_all_backends.md if
    they're extended to 14B). Vulkan available as opt-in.

    Default port is 8081 — avoids the 8080 the daily-driver uses.

.PARAMETER Backend
    'cpu' (default) or 'vulkan'.

.PARAMETER Port
    Listen port. Default 8081.

.PARAMETER Ctx
    Context size. Default 32768. The 4B NPU draft caps at cl=4096
    so extending target ctx past 4k means draft is "blind" to the
    older context (sliding-window-style draft).

.PARAMETER Threads
    CPU thread count for cpu backend. Default 8 (Phase 5 sweet spot
    on the 18-core X2 Elite Extreme; leaves 4 P-cores for the SQ1
    driver + NPU sidecar).

.EXAMPLE
    .\scripts\serve_target_14b.ps1
    # → CPU 14B target on port 8081, ctx=32k.

.EXAMPLE
    .\scripts\serve_target_14b.ps1 -Backend vulkan -Ctx 16384
#>
[CmdletBinding()]
param(
    [ValidateSet('cpu', 'vulkan')][string]$Backend = 'cpu',
    [int]$Port = 8081,
    [int]$Ctx  = 32768,
    [int]$Threads = 8
)

$ErrorActionPreference = 'Continue'

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$model    = Join-Path $repoRoot 'models\Qwen3-14B-Q4_K_M.gguf'

if ($Backend -eq 'vulkan') {
    $exe   = Join-Path $repoRoot 'llama.cpp\build-vulkan\bin\llama-server.exe'
    $extra = @(
        '-ngl', '99',
        '--flash-attn', 'off',
        '--no-warmup'
    )
    # Same Adreno-driver knobs that worked for Qwen3.6-35B-A3B
    # (per daily_driver/recipe.md).
    $env:GGML_VK_DISABLE_F16        = '1'
    $env:GGML_VK_PREFER_HOST_MEMORY = '1'
} else {
    $exe   = Join-Path $repoRoot 'llama.cpp\build-cpu\bin\llama-server.exe'
    $extra = @('-t', $Threads)
}

if (-not (Test-Path $exe))   { throw "Server binary not found: $exe" }
if (-not (Test-Path $model)) {
    throw "Model not found: $model `n  Run: curl.exe -L -C - --retry 5 --fail -o `"$model`" https://huggingface.co/Qwen/Qwen3-14B-GGUF/resolve/main/Qwen3-14B-Q4_K_M.gguf"
}

$serverArgs = @(
    '-m', $model,
    '-c', $Ctx,
    '--alias', 'target-14b',
    '--host', '127.0.0.1',
    '--port', $Port
) + $extra

Write-Host ""
Write-Host "SQ1 target-14b llama-server"            -ForegroundColor Cyan
Write-Host "  backend : $Backend"
Write-Host "  model   : $(Split-Path $model -Leaf)"
Write-Host "  ctx     : $Ctx"
Write-Host "  endpoint: http://127.0.0.1:$Port  (OpenAI-compatible at /v1)"
Write-Host ""
Write-Host "SQ1 driver (in another shell):"
Write-Host "  .venv\Scripts\python.exe last_side_quest\sq1_heterogeneous\demo_path_a.py --target-port $Port" -ForegroundColor Green
Write-Host ""
Write-Host "Ctrl+C to stop."
Write-Host ""

& $exe @serverArgs
