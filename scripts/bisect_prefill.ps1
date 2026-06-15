<#
.SYNOPSIS
    `git bisect run` payload for the dense-prefill regression in the
    llama.cpp window 856c3adac..e37abd6b5 (A/B-confirmed -47% pp512 on
    Qwen3-4B Q4_0; see results/csv/backend_refresh_2026-06-12.md).

.DESCRIPTION
    For the commit currently checked out in llama.cpp/, this script does a
    clean CPU-preset build of llama-bench only, runs a pp512 measurement on
    Qwen3-4B Q4_0, logs (commit, verdict, pp512) to a CSV, and exits with a
    git-bisect status code:

        exit 0   GOOD  pp512 >= -GoodThresh  (predates the regression)
        exit 1   BAD   pp512 <  -GoodThresh  (carries the regression)
        exit 125 SKIP  build or bench failed (bisect skips this commit)

    The threshold is NOT hardcoded blindly: run this script manually on each
    endpoint first (-Calibrate) to record the true pp512 at the known-good
    and known-bad commits, then set -GoodThresh to their midpoint before
    handing it to `git bisect run`.

    A full clean rebuild is forced every step because ninja incremental
    builds are unreliable across a 489-commit span.

.PARAMETER GoodThresh
    pp512 t/s at/above which a commit is classified GOOD. Default 282 is the
    midpoint of the documented OpenCL-build endpoints (369 old, 194 new);
    re-set it from -Calibrate output for the CPU preset actually used here.

.PARAMETER Calibrate
    Build + bench + log + print pp512, then always exit 0. Use to measure an
    endpoint without applying bisect verdict semantics.

.EXAMPLE
    # 1. calibrate endpoints
    git -C ..\llama.cpp checkout 856c3adac; .\bisect_prefill.ps1 -Calibrate
    git -C ..\llama.cpp checkout e37abd6b5; .\bisect_prefill.ps1 -Calibrate
    # 2. set threshold to the midpoint, then:
    git -C ..\llama.cpp bisect start e37abd6b5 856c3adac
    git -C ..\llama.cpp bisect run powershell -NoProfile -File <abs path>\bisect_prefill.ps1 -GoodThresh <mid>
#>
[CmdletBinding()]
param(
    [double]$GoodThresh = 282.0,
    [string]$Model = 'C:\Users\hotschmoe\Documents\GitHub\specula\models\Qwen3-4B-Q4_0.gguf',
    [int]$Jobs = 12,
    [int]$Reps = 5,
    [int]$Threads = 16,
    [switch]$Calibrate
)

$ErrorActionPreference = 'Continue'

$repoRoot    = 'C:\Users\hotschmoe\Documents\GitHub\specula'
$llamaDir    = Join-Path $repoRoot 'llama.cpp'
$buildScript = Join-Path $repoRoot 'scripts\build_llama_cpp.ps1'
$buildDir    = Join-Path $llamaDir 'build-cpu-bisect'
$logCsv      = Join-Path $repoRoot 'results\csv\bisect_prefill_log.csv'

if (-not (Test-Path $logCsv)) {
    'commit,verdict,pp512_ts' | Out-File -FilePath $logCsv -Encoding utf8
}

$commit = (git -C $llamaDir rev-parse --short HEAD).Trim()
Write-Host "=== BISECT step: $commit (thresh=$GoodThresh) ===" -ForegroundColor Cyan

# --- clean build of llama-bench only -----------------------------------------
if (Test-Path $buildDir) { Remove-Item -Recurse -Force $buildDir }

$buildOk = $true
try {
    & $buildScript -Preset cpu -NoGit -BuildDirSuffix '-bisect' -Jobs $Jobs -Targets 'llama-bench'
    if ($LASTEXITCODE -ne 0) { $buildOk = $false }
} catch {
    Write-Host "BUILD THREW: $_" -ForegroundColor Red
    $buildOk = $false
}

$bench = Join-Path $buildDir 'bin\llama-bench.exe'
if ((-not $buildOk) -or (-not (Test-Path $bench))) {
    Write-Host "SKIP $commit (build failed)" -ForegroundColor Yellow
    Add-Content -Path $logCsv -Value "$commit,BUILD_FAIL,"
    exit 125
}

# --- bench pp512 (pure CPU matmul: -ngl 0) -----------------------------------
$raw = & $bench -m $Model -p 512 -n 0 -r $Reps -ngl 0 -t $Threads -o json
$pp = $null
try {
    $obj = $raw | ConvertFrom-Json
    $row = $obj | Where-Object { $_.n_prompt -eq 512 -and $_.n_gen -eq 0 } | Select-Object -First 1
    if ($row) { $pp = [double]$row.avg_ts }
} catch { }

if ($null -eq $pp) {
    Write-Host "SKIP $commit (bench/parse failed)" -ForegroundColor Yellow
    Add-Content -Path $logCsv -Value "$commit,BENCH_FAIL,"
    exit 125
}

$ppR = [math]::Round($pp, 1)
if ($Calibrate) {
    Add-Content -Path $logCsv -Value "$commit,CALIBRATE,$ppR"
    Write-Host "CALIBRATE $commit pp512=$ppR" -ForegroundColor Magenta
    exit 0
}

$verdict = if ($pp -ge $GoodThresh) { 'GOOD' } else { 'BAD' }
Add-Content -Path $logCsv -Value "$commit,$verdict,$ppR"
Write-Host "$commit pp512=$ppR -> $verdict" -ForegroundColor Green
if ($pp -ge $GoodThresh) { exit 0 } else { exit 1 }
