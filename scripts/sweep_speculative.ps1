<#
.SYNOPSIS
    Phase 2 speculative-decoding sweep: runs llama-speculative across a matrix
    of draft-max values, parses acceptance rate + tok/s from stderr, writes CSV.

.DESCRIPTION
    llama-bench doesn't currently cover the speculative flow fully, so this
    wraps llama-speculative directly. The acceptance rate line looks like:
        draft acceptance rate = 0.57576 ( 171 accepted / 297 generated)
    We parse this and emit a structured CSV row.

.PARAMETER TargetModel
    Filename inside ..\models\ for the target (larger) model.

.PARAMETER DraftModel
    Filename inside ..\models\ for the draft (smaller) model.

.PARAMETER DraftMaxValues
    Comma-separated list of draft-max values to sweep.

.PARAMETER Prompts
    JSONL file of prompts to run. Each line should be {"prompt": "..."}.

.EXAMPLE
    .\sweep_speculative.ps1 -TargetModel Qwen3-8B-Q4_K_M.gguf -DraftModel Qwen3-0.6B-Q8_0.gguf
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$TargetModel,
    [Parameter(Mandatory=$true)][string]$DraftModel,
    [int[]]$DraftMaxValues = @(4, 8, 16, 32),
    [string]$Prompts = (Join-Path $PSScriptRoot '..\prompts\humaneval_subset.jsonl'),
    [int]$NPredict = 256,
    [string]$Backend = 'vulkan',
    [string]$BuildDir = (Join-Path $PSScriptRoot '..\llama.cpp\build-vulkan-opencl'),
    [string]$ModelsDir = (Join-Path $PSScriptRoot '..\models'),
    [string]$ResultsDir = (Join-Path $PSScriptRoot '..\results')
)

$ErrorActionPreference = 'Stop'

$spec = Get-ChildItem "$BuildDir\bin\Release\llama-speculative.exe","$BuildDir\bin\llama-speculative.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $spec) { throw "llama-speculative.exe not found under $BuildDir" }

$targetPath = Join-Path $ModelsDir $TargetModel
$draftPath  = Join-Path $ModelsDir $DraftModel
foreach ($p in @($targetPath, $draftPath)) {
    if (-not (Test-Path $p)) { throw "Missing: $p" }
}

# Backend → -ngl
$ngl = if ($Backend -eq 'cpu') { 0 } else { 99 }

New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null
$ts = Get-Date -Format "yyyyMMdd-HHmmss"
$csvPath = Join-Path $ResultsDir "spec-$($Backend)-$($TargetModel -replace '\.gguf$','')-$ts.csv"

# Header
"timestamp,target,draft,backend,draft_max,prompt_idx,accepted,generated,acceptance_rate,tok_per_sec,wallclock_s" | Out-File -FilePath $csvPath -Encoding utf8

# Load prompts
if (-not (Test-Path $Prompts)) {
    Write-Warning "Prompt file $Prompts not found — falling back to a single built-in prompt."
    $promptLines = @('{"prompt":"def fibonacci(n):\n    # Return the nth Fibonacci number\n    "}')
} else {
    $promptLines = Get-Content $Prompts
}

$regexAccept = 'draft acceptance rate = ([\d.]+)\s*\(\s*(\d+) accepted / (\d+) generated\)'
$regexTokSec = '(\d+\.\d+)\s*tokens per second'

foreach ($dMax in $DraftMaxValues) {
    for ($i = 0; $i -lt $promptLines.Count; $i++) {
        try {
            $prompt = ($promptLines[$i] | ConvertFrom-Json).prompt
        } catch {
            Write-Warning "Bad JSON on prompt line $i — skipping"
            continue
        }

        Write-Host "── draft-max=$dMax   prompt $i" -ForegroundColor Cyan

        $start = Get-Date
        $output = & $spec.FullName `
            -m $targetPath `
            -md $draftPath `
            --draft-max $dMax `
            -ngl $ngl `
            -ngld $ngl `
            -n $NPredict `
            --temp 0 `
            -p $prompt 2>&1 | Out-String
        $elapsed = ((Get-Date) - $start).TotalSeconds

        # Parse acceptance rate
        $acceptance = ''
        $accepted = ''
        $generated = ''
        if ($output -match $regexAccept) {
            $acceptance = $matches[1]
            $accepted   = $matches[2]
            $generated  = $matches[3]
        }

        # Parse tok/s (llama-speculative prints several; take the last one, which is overall)
        $tokSec = ''
        $tokMatches = [regex]::Matches($output, $regexTokSec)
        if ($tokMatches.Count -gt 0) {
            $tokSec = $tokMatches[$tokMatches.Count - 1].Groups[1].Value
        }

        $row = @(
            (Get-Date -Format o),
            $TargetModel,
            $DraftModel,
            $Backend,
            $dMax,
            $i,
            $accepted,
            $generated,
            $acceptance,
            $tokSec,
            [math]::Round($elapsed, 2)
        ) -join ','
        $row | Out-File -FilePath $csvPath -Append -Encoding utf8

        Write-Host "   acceptance=$acceptance  tok/s=$tokSec  wall=${elapsed}s"
    }
}

Write-Host ""
Write-Host "Wrote $csvPath" -ForegroundColor Green
