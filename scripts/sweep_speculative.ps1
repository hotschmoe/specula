<#
.SYNOPSIS
    Phase 2 speculative-decoding sweep: runs llama-speculative across a matrix
    of draft-max values, parses the stderr summary, writes a CSV row per run.

.DESCRIPTION
    llama-bench does not cover the speculative flow, so we wrap llama-speculative
    directly. The summary block at end-of-run looks like:

        encoded   13 tokens in    0.150 seconds, speed:   86.468 t/s
        decoded   67 tokens in    2.602 seconds, speed:   25.753 t/s

        n_draft   = 16
        n_predict = 67
        n_drafted = 144
        n_accept  = 57
        accept    = 39.583%

    We parse the decoded-speed and accept fields as primary metrics, plus
    n_drafted / n_accept for raw counts.

    NOTE: the Phase-1 matrix established that CPU wins TG on this hardware
    (8B Q4_K_M CPU 26 t/s vs OpenCL PP-strong, TG-weaker), so the default
    Backend is 'cpu'. Override with -Backend opencl once we want to verify
    the cross-device behaviour.

.PARAMETER TargetModel
    Filename inside ..\models\ for the target (larger) model.

.PARAMETER DraftModel
    Filename inside ..\models\ for the draft (smaller) model.

.PARAMETER DraftMaxValues
    Comma-separated list of --draft-max values to sweep.

.PARAMETER Prompts
    JSONL file of prompts. Each line: {"prompt": "..."}.

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
    [int]$Threads = 18,
    [string]$Backend = 'cpu',
    [string]$BuildDir,
    [int]$TargetNgl = -1,
    [int]$DraftNgl  = -1,
    [string]$PlacementTag,
    [string]$ModelsDir = (Join-Path $PSScriptRoot '..\models'),
    [string]$ResultsDir = (Join-Path $PSScriptRoot '..\results'),
    [string]$LogDir
)

$ErrorActionPreference = 'Stop'

# Default BuildDir follows Backend unless explicitly overridden.
if (-not $BuildDir) {
    $BuildDir = Join-Path $PSScriptRoot "..\llama.cpp\build-$Backend"
}

$spec = Get-ChildItem "$BuildDir\bin\Release\llama-speculative.exe","$BuildDir\bin\llama-speculative.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $spec) { throw "llama-speculative.exe not found under $BuildDir" }

$targetPath = Join-Path $ModelsDir $TargetModel
$draftPath  = Join-Path $ModelsDir $DraftModel
foreach ($p in @($targetPath, $draftPath)) {
    if (-not (Test-Path $p)) { throw "Missing: $p" }
}

# Layer-offload resolution:
#   - Explicit -TargetNgl / -DraftNgl override everything (use -1 for "omit",
#     0 for "all CPU", 99 for "all GPU"). Allows asymmetric placement like
#     target=OpenCL + draft=CPU on the build-opencl binary.
#   - Otherwise fall back to Backend-driven default: cpu omits flags, any
#     other backend passes -ngl 99 -ngld 99.
$gpuLayerArgs = @()
if ($TargetNgl -ge 0 -or $DraftNgl -ge 0) {
    if ($TargetNgl -ge 0) { $gpuLayerArgs += @('-ngl',  $TargetNgl) }
    if ($DraftNgl  -ge 0) { $gpuLayerArgs += @('-ngld', $DraftNgl)  }
} elseif ($Backend -ne 'cpu') {
    $gpuLayerArgs = @('-ngl', '99', '-ngld', '99')
}

New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null
$ts = Get-Date -Format "yyyyMMdd-HHmmss"
$placement = if ($PlacementTag) { "-$PlacementTag" } else { "" }
$tag = "$Backend$placement-$($TargetModel -replace '\.gguf$','')-vs-$($DraftModel -replace '\.gguf$','')"
$csvPath = Join-Path $ResultsDir "spec-$tag-$ts.csv"

if (-not $LogDir) { $LogDir = Join-Path $ResultsDir "spec-$tag-$ts" }
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# Header
"timestamp,target,draft,backend,threads,draft_max,n_predict_req,prompt_idx,n_drafted,n_accept,accept_pct,decoded_tokens,decoded_seconds,decoded_tok_per_sec,encoded_tok_per_sec,wallclock_s" |
    Out-File -FilePath $csvPath -Encoding utf8

# Load prompts
if (-not (Test-Path $Prompts)) {
    throw "Prompt file not found: $Prompts"
}
$promptLines = Get-Content $Prompts

# Regexes for the llama-speculative summary block.
$regexEncoded = 'encoded\s+(\d+)\s+tokens in\s+([\d.]+)\s+seconds, speed:\s+([\d.]+)\s+t/s'
$regexDecoded = 'decoded\s+(\d+)\s+tokens in\s+([\d.]+)\s+seconds, speed:\s+([\d.]+)\s+t/s'
$regexAcceptP = 'accept\s*=\s*([\d.]+)\s*%'
$regexNDraft  = 'n_drafted\s*=\s*(\d+)'
$regexNAccept = 'n_accept\s*=\s*(\d+)'

$total = $DraftMaxValues.Count * $promptLines.Count
$done  = 0

foreach ($dMax in $DraftMaxValues) {
    for ($i = 0; $i -lt $promptLines.Count; $i++) {
        $done++
        try {
            $prompt = ($promptLines[$i] | ConvertFrom-Json).prompt
        } catch {
            Write-Warning "Bad JSON on prompt line $i -- skipping"
            continue
        }

        Write-Host "[$done/$total] draft-max=$dMax prompt=$i" -ForegroundColor Cyan

        # PS 5.1 wraps native-exe stderr into ErrorRecords when you use 2>&1,
        # which trips $ErrorActionPreference='Stop'. Use Start-Process with
        # file-based redirection instead — it also keeps stdout/stderr clean.
        $logPath = Join-Path $LogDir "dmax$dMax-p$i.log"
        $stderrPath = "$logPath.err"
        $stdoutPath = "$logPath.out"

        # Write the prompt to a temp file and pass via -f to avoid
        # Start-Process's argument-array quoting splitting on spaces/newlines.
        # Use -e (escape processing) so literal "\n" in the JSONL becomes a
        # real newline, matching the original -p behaviour.
        $promptFile = Join-Path $LogDir "dmax$dMax-p$i.prompt.txt"
        Set-Content -Path $promptFile -Value $prompt -Encoding utf8 -NoNewline

        $exeArgs = @(
            '-m', $targetPath,
            '-md', $draftPath,
            '-t', $Threads, '-td', $Threads,
            '--draft-max', $dMax, '--draft-min', 0,
            '-n', $NPredict,
            '--temp', 0,
            '-e',
            '-f', $promptFile
        ) + $gpuLayerArgs

        $start = Get-Date
        $proc = Start-Process -FilePath $spec.FullName `
            -ArgumentList $exeArgs `
            -NoNewWindow -Wait -PassThru `
            -RedirectStandardOutput $stdoutPath `
            -RedirectStandardError  $stderrPath
        $elapsed = ((Get-Date) - $start).TotalSeconds

        # The summary block (encoded/decoded/n_accept/accept) goes to stderr
        # (LOG_INF). Stdout has the generated text. Combine for the log file.
        $stderrText = if (Test-Path $stderrPath) { Get-Content $stderrPath -Raw } else { '' }
        $stdoutText = if (Test-Path $stdoutPath) { Get-Content $stdoutPath -Raw } else { '' }
        $output = $stderrText + "`n===STDOUT===`n" + $stdoutText
        $output | Out-File -FilePath $logPath -Encoding utf8
        Remove-Item $stderrPath, $stdoutPath -ErrorAction SilentlyContinue

        # Parse.
        $encTs   = ''
        $decTs   = ''; $decTok = ''; $decSec = ''
        $acceptP = ''
        $nDraft  = ''; $nAccept = ''

        if ($output -match $regexEncoded) { $encTs = $matches[3] }
        if ($output -match $regexDecoded) {
            $decTok = $matches[1]; $decSec = $matches[2]; $decTs = $matches[3]
        }
        if ($output -match $regexAcceptP) { $acceptP = $matches[1] }
        if ($output -match $regexNDraft)  { $nDraft  = $matches[1] }
        if ($output -match $regexNAccept) { $nAccept = $matches[1] }

        $row = @(
            (Get-Date -Format o),
            $TargetModel,
            $DraftModel,
            $Backend,
            $Threads,
            $dMax,
            $NPredict,
            $i,
            $nDraft,
            $nAccept,
            $acceptP,
            $decTok,
            $decSec,
            $decTs,
            $encTs,
            [math]::Round($elapsed, 2)
        ) -join ','
        $row | Out-File -FilePath $csvPath -Append -Encoding utf8

        Write-Host "   accept=${acceptP}%  decoded=${decTs} t/s  wall=${elapsed}s"
    }
}

Write-Host ""
Write-Host "Wrote $csvPath" -ForegroundColor Green
Write-Host "Raw logs in $LogDir"
