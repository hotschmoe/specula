<#
.SYNOPSIS
    Probe a running daily-driver llama-server.

.DESCRIPTION
    Confirms the server is up and responding before you launch opencode.
    Hits /health and /v1/models, prints the model id and a small
    completion-latency sample so you can sanity-check end-to-end.

.PARAMETER Port
    Port the server is listening on. Default 8080.

.EXAMPLE
    .\scripts\check_daily_driver_status.ps1
#>
[CmdletBinding()]
param(
    [int]$Port = 8080
)

$base = "http://127.0.0.1:$Port"

function Try-Get($url, $timeoutSec = 5) {
    try { return Invoke-RestMethod -Uri $url -Method Get -TimeoutSec $timeoutSec }
    catch { return $null }
}

Write-Host ""
Write-Host "Probing $base ..." -ForegroundColor Cyan

$health = Try-Get "$base/health"
if ($null -eq $health) {
    Write-Host "  /health      : NOT REACHABLE - server not running on port $Port" -ForegroundColor Red
    Write-Host ""
    Write-Host "Start it with: .\scripts\serve_daily_driver.ps1"
    exit 1
}
Write-Host "  /health      : $($health.status)" -ForegroundColor Green

$models = Try-Get "$base/v1/models"
if ($null -ne $models -and $models.data) {
    $ids = ($models.data | ForEach-Object { $_.id }) -join ', '
    Write-Host "  /v1/models   : $ids" -ForegroundColor Green
} else {
    Write-Host "  /v1/models   : (no data field, server too old?)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Sending a 16-token completion as smoke test..." -ForegroundColor Cyan
$body = @{
    prompt      = "Hello in one word:"
    n_predict   = 16
    temperature = 0
    seed        = 0
} | ConvertTo-Json

$sw = [System.Diagnostics.Stopwatch]::StartNew()
try {
    $resp = Invoke-RestMethod -Uri "$base/completion" -Method Post `
        -ContentType 'application/json' -Body $body -TimeoutSec 600
} catch {
    Write-Host "  POST /completion failed: $_" -ForegroundColor Red
    exit 1
}
$sw.Stop()

$tg = $resp.timings.predicted_per_second
$pp = $resp.timings.prompt_per_second
Write-Host "  completion   : $($resp.tokens_predicted) tokens in $($sw.Elapsed.TotalSeconds.ToString('F1'))s"
Write-Host "  TG t/s       : $($tg.ToString('F1'))" -ForegroundColor Green
Write-Host "  PP t/s       : $($pp.ToString('F1'))"
Write-Host ""
Write-Host "Server looks healthy. opencode model id: 'llama-cpp-local/daily-driver'" -ForegroundColor Cyan
Write-Host ""
