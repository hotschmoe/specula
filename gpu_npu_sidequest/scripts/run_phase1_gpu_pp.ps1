# Phase 1 GPU track — PP plateau confirmation.
# Extends the Phase 0 GPU PP sweep to large k (128, 256, 512) to test the
# fixed-overhead(~165 ms) + linear(compute) per-pass model.
# Output: JSON logs in logs\, parsed into results\phase1_gpu_pp.csv.

$ErrorActionPreference = 'Continue'
$root   = 'C:\Users\hotschmoe\Documents\GitHub\specula\gpu_npu_sidequest'
$bench  = 'C:\Users\hotschmoe\Documents\GitHub\specula\llama.cpp\build-opencl\bin\llama-bench.exe'
$model  = 'C:\Users\hotschmoe\Documents\GitHub\specula\models\Qwen3-4B-Q4_0.gguf'
$logdir = Join-Path $root 'logs'

# PP sweep: k = 128, 256, 512. -b/-ub >= k so the k-token forward is one
# un-chunked physical batch (single k-wide pass timed).
$ppK = 128,256,512
foreach ($k in $ppK) {
    $log = Join-Path $logdir "gpu_pp$k.json"
    Write-Host "=== PP k=$k ===" -ForegroundColor Cyan
    & $bench -m $model -ngl 99 -p $k -n 0 -b $k -ub $k -r 5 -o json 2>$null | Out-File -FilePath $log -Encoding utf8
    Write-Host "  -> $log"
}

Write-Host "`nAll GPU PP runs complete." -ForegroundColor Green
