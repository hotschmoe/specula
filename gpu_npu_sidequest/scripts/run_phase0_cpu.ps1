# Phase 0 CPU track — verify-shape anchor.
# CPU PP swept over k = 1,2,4,8,16,32,128,512 (one un-chunked k-token
# forward each) plus a TG anchor. Completes the 3-island crossover picture.
# Output: JSON logs in logs\, parsed into results\phase0_cpu.csv.

$ErrorActionPreference = 'Continue'
$root   = 'C:\Users\hotschmoe\Documents\GitHub\specula\gpu_npu_sidequest'
$bench  = 'C:\Users\hotschmoe\Documents\GitHub\specula\llama.cpp\build-cpu\bin\llama-bench.exe'
$model  = 'C:\Users\hotschmoe\Documents\GitHub\specula\models\Qwen3-4B-Q4_0.gguf'
$logdir = Join-Path $root 'logs'

# -t 16 is the documented Qwen3-4B sweet spot on this box (phys_cores-2).
$threads = 16

$ppK = 1,2,4,8,16,32,128,512
foreach ($k in $ppK) {
    $log = Join-Path $logdir "cpu_pp$k.json"
    Write-Host "=== CPU PP k=$k ===" -ForegroundColor Cyan
    & $bench -m $model -ngl 0 -t $threads -p $k -n 0 -b $k -ub $k -r 5 -o json 2>$null | Out-File -FilePath $log -Encoding utf8
    Write-Host "  -> $log"
}

# TG anchor.
$log = Join-Path $logdir "cpu_tg64.json"
Write-Host "=== CPU TG n=64 ===" -ForegroundColor Cyan
& $bench -m $model -ngl 0 -t $threads -p 0 -n 64 -r 5 -o json 2>$null | Out-File -FilePath $log -Encoding utf8
Write-Host "  -> $log"

Write-Host "`nAll CPU runs complete." -ForegroundColor Green
