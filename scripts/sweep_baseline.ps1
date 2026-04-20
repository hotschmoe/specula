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
    [ValidateSet('cpu', 'cpu-kleidiai', 'vulkan', 'opencl')][string]$Backend = 'cpu',
    # BuildDir is resolved from $Backend if not supplied. Each backend
    # has its own build-<preset>/ directory (see build_llama_cpp.ps1).
    [string]$BuildDir = '',
    [string]$ModelsDir = (Join-Path $PSScriptRoot '..\models'),
    [string]$ResultsDir = (Join-Path $PSScriptRoot '..\results')
)

# NB: intentionally NOT 'Stop' — PS 5.1 wraps every stderr line from a
# native exe in a NativeCommandError, and llama-bench routes status
# output (device enumeration, timings) through stderr. With 'Stop'
# the first such line kills this script mid-run. Explicit
# $LASTEXITCODE checks below are the correct signal.
$ErrorActionPreference = 'Continue'

if (-not $BuildDir) {
    $BuildDir = Join-Path $PSScriptRoot "..\llama.cpp\build-$Backend"
}

$modelPath = Join-Path $ModelsDir $Model
if (-not (Test-Path $modelPath)) { throw "Model not found: $modelPath" }

$bench = Get-ChildItem "$BuildDir\bin\Release\llama-bench.exe","$BuildDir\bin\llama-bench.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $bench) { throw "llama-bench.exe not found under $BuildDir -- build the '$Backend' preset first (scripts\build_llama_cpp.ps1 -Preset $Backend)" }

New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null
$ts = Get-Date -Format "yyyyMMdd-HHmmss"
$csv = Join-Path $ResultsDir "baseline-$($Backend)-$($Model -replace '\.gguf$','')-$ts.csv"

# Backend-specific args:
# - CPU: sweep threads (8/12/18) to capture scaling on the 18-core X2
#   Elite. -ngl 0 guarantees no GPU offload even if the binary has a
#   GPU backend compiled in (CPU build doesn't, but stays correct).
# - OpenCL / Vulkan: fix threads at a single value. With -ngl 99 the
#   GPU runs the tensors, CPU handles orchestration only; the -t
#   sweep is wasted time.
# No -d flag: each backend has its own build-<preset>\ so the only
# backend available at runtime is the one we asked for. llama-bench
# auto-selects it.
switch ($Backend) {
    'cpu'          { $deviceArgs = @('-ngl', '0',  '-t', '8,12,18') }
    'cpu-kleidiai' { $deviceArgs = @('-ngl', '0',  '-t', '8,12,18') }
    'vulkan'       { $deviceArgs = @('-ngl', '99', '-t', '8') }
    'opencl'       { $deviceArgs = @('-ngl', '99', '-t', '8') }
}

Write-Host "Sweep: $Model on $Backend" -ForegroundColor Green
Write-Host "Bench: $($bench.FullName)"
Write-Host "Output: $csv"
Write-Host ""

# Bench matrix: PP at two prompt sizes (short + longer to catch
# compute vs cache-thrash behavior) × TG at two gen lengths × 3 reps.
# Numbers chosen to keep total runtime bounded while still producing
# enough signal for a Phase 1 baseline.
& $bench.FullName `
    -m $modelPath `
    @deviceArgs `
    -p 128,512 `
    -n 64,128 `
    -r 3 `
    -o csv | Tee-Object -FilePath $csv
if ($LASTEXITCODE -ne 0) { throw "llama-bench failed (exit $LASTEXITCODE)" }

Write-Host ""
Write-Host "Wrote $csv" -ForegroundColor Green
