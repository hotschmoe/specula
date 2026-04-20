<#
.SYNOPSIS
    Download initial GGUF model set for specula speculative-decoding experiments.

.DESCRIPTION
    Uses curl.exe with resume (-C -) so partially downloaded files continue
    from where they left off if interrupted. Safe to re-run.

.PARAMETER ModelsDir
    Destination directory. Defaults to ..\models relative to this script.

.PARAMETER Tier
    Which model tier(s) to fetch:
        core      – draft + Qwen3-8B target          (~7.5 GB total) [default]
        prod      – adds Qwen3-14B target            (+9.0 GB)
        stretch   – adds Qwen3-32B target            (+19 GB)
        all       – everything including MoE         (~50 GB)

.EXAMPLE
    .\download_models.ps1
    .\download_models.ps1 -Tier prod
    .\download_models.ps1 -Tier all -ModelsDir D:\models
#>
[CmdletBinding()]
param(
    [string]$ModelsDir = (Join-Path $PSScriptRoot '..\models'),
    [ValidateSet('core', 'prod', 'stretch', 'all')]
    [string]$Tier = 'core'
)

$ErrorActionPreference = 'Stop'

# Model registry: {repo, file, role, tier, approx_size_gb}
$allModels = @(
    @{ Repo='Qwen/Qwen3-0.6B-GGUF';    File='Qwen3-0.6B-Q8_0.gguf';    Role='draft (primary)';     Tier='core';    SizeGB=0.64 },
    @{ Repo='Qwen/Qwen3-1.7B-GGUF';    File='Qwen3-1.7B-Q8_0.gguf';    Role='draft (alternate)';   Tier='core';    SizeGB=1.83 },
    @{ Repo='Qwen/Qwen3-8B-GGUF';      File='Qwen3-8B-Q4_K_M.gguf';    Role='target (iteration)';  Tier='core';    SizeGB=5.03 },
    @{ Repo='Qwen/Qwen3-14B-GGUF';     File='Qwen3-14B-Q4_K_M.gguf';   Role='target (prod)';       Tier='prod';    SizeGB=9.00 },
    @{ Repo='Qwen/Qwen3-32B-GGUF';     File='Qwen3-32B-Q4_K_M.gguf';   Role='target (stretch)';    Tier='stretch'; SizeGB=19.0 },
    @{ Repo='Qwen/Qwen3-30B-A3B-GGUF'; File='Qwen3-30B-A3B-Q4_K_M.gguf'; Role='target (MoE)';      Tier='all';     SizeGB=18.0 }
)

# Filter by tier: each tier includes all lower tiers
$tierOrder = @{ 'core'=0; 'prod'=1; 'stretch'=2; 'all'=3 }
$wantLevel = $tierOrder[$Tier]
$models = $allModels | Where-Object { $tierOrder[$_.Tier] -le $wantLevel }

# Ensure models dir exists
New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null
$ModelsDir = (Resolve-Path $ModelsDir).Path

$totalGB = ($models | Measure-Object -Property SizeGB -Sum).Sum
Write-Host ""
Write-Host "specula — model fetch" -ForegroundColor Green
Write-Host "  destination: $ModelsDir"
Write-Host "  tier:        $Tier ($($models.Count) files, ~$([math]::Round($totalGB,1)) GB total)"
Write-Host ""

# Require curl.exe (ships with Windows 10+, Win11 ARM has it)
$curl = Get-Command curl.exe -ErrorAction SilentlyContinue
if (-not $curl) {
    throw "curl.exe not found. Install Git for Windows or enable the built-in curl."
}

$failures = @()
foreach ($m in $models) {
    $dest = Join-Path $ModelsDir $m.File
    $url  = "https://huggingface.co/$($m.Repo)/resolve/main/$($m.File)"

    Write-Host "── $($m.File)" -ForegroundColor Cyan
    Write-Host "   role: $($m.Role)   (~$($m.SizeGB) GB)"
    Write-Host "   src:  $url"
    Write-Host "   dest: $dest"

    # curl.exe flags:
    #   -L           follow HF's redirect to the CDN
    #   -C -         resume from byte offset of existing partial file
    #   --retry 5    retry transient failures (flaky WiFi, HF 503, etc.)
    #   --retry-delay 5
    #   --fail       non-zero exit on HTTP errors (so we can catch them)
    #   --progress-bar  clean progress output
    & curl.exe -L -C - --retry 5 --retry-delay 5 --fail --progress-bar -o $dest $url

    if ($LASTEXITCODE -ne 0) {
        Write-Warning "   FAILED (curl exit $LASTEXITCODE) — will continue with remaining files"
        $failures += $m.File
    } else {
        $sizeMB = [math]::Round((Get-Item $dest).Length / 1MB, 1)
        Write-Host "   ok ($sizeMB MB)" -ForegroundColor Green
    }
    Write-Host ""
}

Write-Host ""
if ($failures.Count -eq 0) {
    Write-Host "All downloads complete." -ForegroundColor Green
} else {
    Write-Host "Some downloads failed. Re-run this script to resume:" -ForegroundColor Yellow
    $failures | ForEach-Object { Write-Host "    $_" -ForegroundColor Yellow }
    exit 1
}
