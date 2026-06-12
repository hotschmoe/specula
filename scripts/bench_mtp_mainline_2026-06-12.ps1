# Session 35: re-run the Qwen3.6-27B MTP sweep on the MAINLINE build
# (MTP merged to mainline: #23643/#24025/#23287), to confirm it
# reproduces session-27's build-opencl-mtp numbers and retire that
# special build. Forked from bench_mtp_server_sweep.ps1 — only the
# binary path, output path, and build stamp differ.
param(
  [string]$Quant = "Q4_0",
  [int]$Port = 8191
)
$ErrorActionPreference = "Continue"

$repo  = "C:\Users\hotschmoe\Documents\GitHub\specula"
$model = "$repo\models\Qwen3.6-27B-MTP-$Quant.gguf"
$bin   = "$repo\llama.cpp\build-opencl\bin\llama-server.exe"   # MAINLINE, not -mtp
$out   = "$repo\results\csv\qwen3_6_27b_${Quant}_mtp_mainline_2026-06-12.md"

"# Qwen3.6-27B-MTP-$Quant MTP sweep via llama-server (MAINLINE)" | Out-File $out -Encoding utf8
"Build: mainline e37abd6b5 (was PR #22673 build in session 27)" | Out-File $out -Append -Encoding utf8
"Model: $model" | Out-File $out -Append -Encoding utf8
"" | Out-File $out -Append -Encoding utf8
"| config | n_max | PP t/s | TG t/s | draft_n | accepted | accept % |" | Out-File $out -Append -Encoding utf8
"|---|---:|---:|---:|---:|---:|---:|" | Out-File $out -Append -Encoding utf8

$prompt = "Explain in detail how a transformer language model performs next token prediction, covering attention, MLPs, and the residual stream. Be specific about dimensions for a 27B parameter model with 5120 hidden size."

function Start-Server($extraArgs, $label) {
  $stdout = "$env:TEMP\srv_${label}_stdout.log"
  $stderr = "$env:TEMP\srv_${label}_stderr.log"
  $allArgs = @("-m",$script:model,"-c","4096","--port",$script:Port,"--host","127.0.0.1","--no-warmup") + $extraArgs
  $p = Start-Process -FilePath $script:bin -ArgumentList $allArgs -RedirectStandardOutput $stdout -RedirectStandardError $stderr -NoNewWindow -PassThru
  for ($i=0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 3
    try {
      $h = Invoke-WebRequest -Uri "http://127.0.0.1:$($script:Port)/health" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
      if ($h.StatusCode -eq 200) { return $p.Id }
    } catch { }
  }
  Write-Host "  ERROR: server $label did not become ready" -ForegroundColor Red
  Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
  Get-Content $stderr -Tail 25 | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkRed }
  return $null
}

function Stop-Server($srvPid) {
  if ($null -ne $srvPid) { Stop-Process -Id $srvPid -Force -ErrorAction SilentlyContinue; Start-Sleep -Seconds 3 }
}

function Run-Bench($label, $n_max) {
  $body = @{prompt=$script:prompt; n_predict=256; temperature=0; stream=$false} | ConvertTo-Json
  try {
    $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$($script:Port)/completion" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 600
    $pp = [Math]::Round($resp.timings.prompt_per_second, 2)
    $tg = [Math]::Round($resp.timings.predicted_per_second, 2)
    $dn = if ($resp.timings.draft_n) { $resp.timings.draft_n } else { 0 }
    $da = if ($resp.timings.draft_n_accepted) { $resp.timings.draft_n_accepted } else { 0 }
    $ar = if ($dn -gt 0) { [Math]::Round(100.0 * $da / $dn, 1) } else { 0 }
    $row = "| $label | $n_max | $pp | $tg | $dn | $da | $ar |"
    $row | Out-File $script:out -Append -Encoding utf8
    Write-Output $row
  } catch {
    "| $label | $n_max | ERROR | ERROR | - | - | - |" | Out-File $script:out -Append -Encoding utf8
    Write-Output "  ERROR on $label : $_"
  }
}

# Focused config set: no-MTP baseline + the session-27 peak (n8) plus
# neighbours to confirm the curve. ngl=0 coprocessor (27B does not
# OpenCL-offload per session 27).
$configs = @(
  @{label="ngl=0 t=18 no-MTP"; args=@("-ngl","0","-t","18","--spec-type","none"); n_max=0},
  @{label="ngl=0 t=18 MTP n4"; args=@("-ngl","0","-t","18","--spec-type","draft-mtp","--spec-draft-n-max","4"); n_max=4},
  @{label="ngl=0 t=18 MTP n8"; args=@("-ngl","0","-t","18","--spec-type","draft-mtp","--spec-draft-n-max","8"); n_max=8},
  @{label="ngl=0 t=18 MTP n12"; args=@("-ngl","0","-t","18","--spec-type","draft-mtp","--spec-draft-n-max","12"); n_max=12}
)

foreach ($c in $configs) {
  Write-Output "--- starting: $($c.label) ---"
  $srvPid = Start-Server $c.args ($c.label -replace "[^a-zA-Z0-9]","_")
  if ($null -ne $srvPid) { Run-Bench $c.label $c.n_max; Stop-Server $srvPid }
}
"---DONE: $out ---"
