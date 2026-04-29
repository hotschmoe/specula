<#
.SYNOPSIS
    Start a llama-server endpoint for a small Qwen comparator.

.DESCRIPTION
    Sister script to `scripts/serve_daily_driver.ps1`. That one targets
    Qwen3.6-35B-A3B at port 8080. This one parameterizes for the smaller
    SQ6 comparators:
      * Qwen3-4B-Q4_K_M     (port 8082 by default)
      * Qwen2.5-7B-Q4_0     (port 8083 by default)

    Vulkan is the recommended backend per
    `docs/qwen3_4b_baseline_all_backends.md` — pure Q4_0 + DISABLE_F16
    + PREFER_HOST gives the best TG single-stream + concurrency.

    The NPU server (npu_engine/http_server.py) handles Qwen3-4B-NPU on
    port 8081. With this script you can stand up Vulkan/CPU comparators
    on 8082 and 8083 simultaneously without colliding.

.PARAMETER Model
    'qwen3-4b' (Q4_K_M) or 'qwen2_5-7b' (Q4_0).

.PARAMETER Backend
    'vulkan' (default) or 'cpu'.

.PARAMETER Port
    Listen port. Defaults: 8082 for qwen3-4b, 8083 for qwen2_5-7b.

.PARAMETER Ctx
    Context size. Default 16384 (well above realistic SQ6 chat).

.PARAMETER Threads
    CPU thread count for `cpu` backend. Default 8.

.EXAMPLE
    .\scripts\serve_small.ps1 -Model qwen3-4b
    .\scripts\serve_small.ps1 -Model qwen2_5-7b -Backend cpu -Port 9000
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('qwen3-4b', 'qwen2_5-7b')] [string]$Model,
    [ValidateSet('vulkan', 'cpu')] [string]$Backend = 'vulkan',
    [int]$Port = 0,
    [int]$Ctx  = 16384,
    [int]$Threads = 8
)

$ErrorActionPreference = 'Continue'
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..')

# Resolve model GGUF + alias by Model name + Backend (Vulkan prefers Q4_0
# per all-backends doc; CPU happily takes Q4_K_M).
switch ($Model) {
    'qwen3-4b' {
        $alias = 'qwen3-4b'
        $defaultPort = 8082
        $modelFile = if ($Backend -eq 'vulkan') {
            'Qwen3-4B-Q4_0.gguf'   # may not exist; falls through to Q4_K_M
        } else { 'Qwen3-4B-Q4_K_M.gguf' }
    }
    'qwen2_5-7b' {
        $alias = 'qwen2_5-7b'
        $defaultPort = 8083
        $modelFile = if ($Backend -eq 'vulkan') {
            'Qwen2.5-7B-Instruct-Q4_0.gguf'
        } else { 'Qwen2.5-7B-Instruct-Q4_K_M.gguf' }
    }
}

# Fallback: if the Vulkan-preferred Q4_0 isn't on disk, drop back to Q4_K_M.
$modelPath = Join-Path $repoRoot "models\$modelFile"
if (-not (Test-Path $modelPath)) {
    $fallback = $modelFile -replace 'Q4_0', 'Q4_K_M'
    $fallbackPath = Join-Path $repoRoot "models\$fallback"
    if (Test-Path $fallbackPath) {
        Write-Host "  Q4_0 not on disk; falling back to Q4_K_M" -ForegroundColor Yellow
        $modelPath = $fallbackPath
        $modelFile = $fallback
    } else {
        throw "Model not found: $modelPath (also checked $fallbackPath)"
    }
}

if ($Port -eq 0) { $Port = $defaultPort }

if ($Backend -eq 'vulkan') {
    $exe = Join-Path $repoRoot 'llama.cpp\build-vulkan\bin\llama-server.exe'
    $extra = @(
        '-ngl', '99',
        '--flash-attn', 'off',
        '--no-warmup'
    )
    # Knobs from the all-backends doc — Adreno Vulkan ICD's FP16 codepath
    # falls into a slow scalar fallback for Q4 matmul; FP32 runs cleanly.
    $env:GGML_VK_DISABLE_F16        = '1'
    $env:GGML_VK_PREFER_HOST_MEMORY = '1'
} else {
    $exe = Join-Path $repoRoot 'llama.cpp\build-cpu\bin\llama-server.exe'
    $extra = @('-t', $Threads)
}

if (-not (Test-Path $exe)) { throw "Server binary not found: $exe" }

# Bind to 0.0.0.0 so WSL can hit us via host.docker.internal. (Daily-driver
# binds 127.0.0.1; harder to drive from WSL. SQ6 wants WSL clients.)
$serverArgs = @(
    '-m', $modelPath,
    '-c', $Ctx,
    '--alias', $alias,
    '--host', '0.0.0.0',
    '--port', $Port
) + $extra

Write-Host ""
Write-Host "small-model llama-server" -ForegroundColor Cyan
Write-Host "  alias  : $alias"
Write-Host "  backend: $Backend"
Write-Host "  model  : $modelFile"
Write-Host "  ctx    : $Ctx"
Write-Host "  bind   : 0.0.0.0:$Port  (reachable from WSL via host.docker.internal:$Port)"
Write-Host ""
Write-Host "Point pi (or any OpenAI-compat client) at:"
Write-Host "  base URL: http://127.0.0.1:$Port/v1   (from Windows)"   -ForegroundColor Green
Write-Host "  base URL: http://host.docker.internal:$Port/v1   (from WSL)" -ForegroundColor Green
Write-Host "  api key : (any non-empty string)"
Write-Host ""
Write-Host "Ctrl+C to stop."
Write-Host ""

& $exe @serverArgs
