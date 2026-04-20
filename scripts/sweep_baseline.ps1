<#
.SYNOPSIS
    Phase 1 baseline: sweep autoregressive token-generation tok/s for a given
    model across backends, threads, and context sizes. Writes a CSV row per run.

.DESCRIPTION
    Uses llama-bench (which supports comma-separated lists for most flags,
    producing the cartesian product in one invocation).

.PARAMETER Model
    Filename inside ..\models\ (e.g. "Qwen3-8B-Q4_K_M.gguf")

.PARAMETER Backend
    Which device to test: cpu, vulkan, opencl. (hexagon requires a separate
    build -- see docs/backend/snapdragon/README.md).

.EXAMPLE
    .\sweep_baseline.ps1 -Model Qwen3-8B-Q4_K_M.gguf -Backend cpu
    .\sweep_baseline.ps1 -Model Qwen3-8B-Q4_K_M.gguf -Backend vulkan
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Model,
    [ValidateSet('cpu', 'vulkan', 'opencl')][string]$Backend = 'cpu',
    [string]$BuildDir = (Join-Path $PSScriptRoot '..\llama.cpp\build-vulkan-opencl'),
    [string]$ModelsDir = (Join-Path $PSScriptRoot '..\models'),
    [string]$ResultsDir = (Join-Path $PSScriptRoot '..\results')
)

$ErrorActionPreference = 'Stop'

$modelPath = Join-Path $ModelsDir $Model
if (-not (Test-Path $modelPath)) { throw "Model not found: $modelPath" }

$bench = Get-ChildItem "$BuildDir\bin\Release\llama-bench.exe","$BuildDir\bin\llama-bench.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $bench) { throw "llama-bench.exe not found under $BuildDir -- run build_llama_cpp.ps1 first" }

New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null
$ts = Get-Date -Format "yyyyMMdd-HHmmss"
$csv = Join-Path $ResultsDir "baseline-$($Backend)-$($Model -replace '\.gguf$','')-$ts.csv"

# Backend-specific device selection for llama-bench.
# llama-bench uses -d for device and -ngl for gpu layer count.
switch ($Backend) {
    'cpu'    { $deviceArgs = @('-ngl', '0') }
    'vulkan' { $deviceArgs = @('-ngl', '99') }
    'opencl' { $deviceArgs = @('-ngl', '99', '-d', 'GPUOpenCL') }
}

Write-Host "Sweep: $Model on $Backend" -ForegroundColor Green
Write-Host "Output: $csv"
Write-Host ""

& $bench.FullName `
    -m $modelPath `
    @deviceArgs `
    -t 4,8,12 `
    -p 512 `
    -n 128 `
    -r 3 `
    -o csv | Tee-Object -FilePath $csv

Write-Host ""
Write-Host "Wrote $csv" -ForegroundColor Green
