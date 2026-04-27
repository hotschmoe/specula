<#
.SYNOPSIS
    Start the daily-driver llama-server for opencode / agent-harness use.

.DESCRIPTION
    Wraps the canonical Qwen3.6-35B-A3B serve config from
    `daily_driver/recipe.md` (post-Phase-10c). Vulkan is the default;
    CPU is the fallback. Prints the OpenAI-compatible endpoint URL on
    startup so opencode can be pointed at it.

    Phase 10c numbers (real-prompt, llama-server, d=16k+ multi-turn):
      Vulkan  cold session 5.7 min, per-turn 13.2 s, TG 20.0 t/s
      CPU     cold session 12.5 min, per-turn 21.8 s, TG 15.5 t/s

    The Vulkan build uses --flash-attn off (FA livelocks at this
    llama.cpp commit) and --no-warmup (defers shader compilation to
    the first real request). The first request after startup will be
    slow; subsequent are normal.

.PARAMETER Backend
    'vulkan' (default, recommended) or 'cpu' (fallback).

.PARAMETER Port
    Listen port. Default 8080.

.PARAMETER Ctx
    Context size. Default 131072 (128k). Set lower if you want to
    cap memory headroom.

.PARAMETER Threads
    CPU thread count for `cpu` backend. Default 8 (Phase 5 sweet
    spot, leaves 4 P-cores for the agent harness).

.EXAMPLE
    # Default: Vulkan canonical
    .\scripts\serve_daily_driver.ps1

.EXAMPLE
    # CPU fallback when Vulkan is unavailable
    .\scripts\serve_daily_driver.ps1 -Backend cpu

.EXAMPLE
    # Custom port + smaller ctx
    .\scripts\serve_daily_driver.ps1 -Port 9090 -Ctx 32768
#>
[CmdletBinding()]
param(
    [ValidateSet('vulkan', 'cpu')][string]$Backend = 'vulkan',
    [int]$Port = 8080,
    [int]$Ctx  = 131072,
    [int]$Threads = 8
)

$ErrorActionPreference = 'Continue'

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..')

if ($Backend -eq 'vulkan') {
    $exe   = Join-Path $repoRoot 'llama.cpp\build-vulkan\bin\llama-server.exe'
    $model = Join-Path $repoRoot 'models\Qwen3.6-35B-A3B-MXFP4_MOE.gguf'
    $extra = @(
        '-ngl', '99',
        '--flash-attn', 'off',
        '--no-warmup'
    )
    $env:GGML_VK_DISABLE_F16        = '1'
    $env:GGML_VK_PREFER_HOST_MEMORY = '1'
} else {
    $exe   = Join-Path $repoRoot 'llama.cpp\build-cpu\bin\llama-server.exe'
    $model = Join-Path $repoRoot 'models\Qwen3.6-35B-A3B-Q4_K_M.gguf'
    $extra = @('-t', $Threads)
}

if (-not (Test-Path $exe))   { throw "Server binary not found: $exe" }
if (-not (Test-Path $model)) { throw "Model not found: $model -- fetch via daily_driver/recipe.md step 1" }

$serverArgs = @(
    '-m', $model,
    '-c', $Ctx,
    '--alias', 'daily-driver',
    '--host', '127.0.0.1',
    '--port', $Port
) + $extra

Write-Host ""
Write-Host "daily-driver llama-server"            -ForegroundColor Cyan
Write-Host "  backend : $Backend"
Write-Host "  model   : $(Split-Path $model -Leaf)"
Write-Host "  ctx     : $Ctx"
Write-Host "  endpoint: http://127.0.0.1:$Port  (OpenAI-compatible at /v1)"
if ($Backend -eq 'vulkan') {
    Write-Host "  note    : --no-warmup means the FIRST request will be slow" -ForegroundColor Yellow
    Write-Host "            (Vulkan shader JIT + 16k cold prefill ~ 5-6 min)"
}
Write-Host ""
Write-Host "Point opencode (or any OpenAI-compat client) at:"
Write-Host "  base URL: http://127.0.0.1:$Port/v1"   -ForegroundColor Green
Write-Host "  api key : (any non-empty string)"
Write-Host ""
Write-Host "Ctrl+C to stop."
Write-Host ""

& $exe @serverArgs
