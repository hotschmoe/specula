# Phase 0 GPU track — verify-shape crossover microbench.
# Runs llama-bench on the Adreno X2-90 via the OpenCL backend.
# PP (batched k-token forward) swept over k; TG (autoregressive) anchor.
# Output: JSON logs in logs\, parsed downstream into results\phase0_gpu.csv.

$ErrorActionPreference = 'Continue'
$root   = 'C:\Users\hotschmoe\Documents\GitHub\specula\gpu_npu_sidequest'
$bench  = 'C:\Users\hotschmoe\Documents\GitHub\specula\llama.cpp\build-opencl\bin\llama-bench.exe'
$model  = 'C:\Users\hotschmoe\Documents\GitHub\specula\models\Qwen3-4B-Q4_0.gguf'
$logdir = Join-Path $root 'logs'

# PP sweep: k = 1,2,4,8,16,32,64. For each k set -b/-ub >= k so the
# k-token forward is one un-chunked physical batch (we want a single
# k-wide pass timed, not internally split into ub-sized sub-batches).
$ppK = 1,2,4,8,16,32,64
foreach ($k in $ppK) {
    $log = Join-Path $logdir "gpu_pp$k.json"
    Write-Host "=== PP k=$k ===" -ForegroundColor Cyan
    & $bench -m $model -ngl 99 -p $k -n 0 -b $k -ub $k -r 5 -o json 2>$null | Out-File -FilePath $log -Encoding utf8
    Write-Host "  -> $log"
}

# TG anchors: n = 32, 64. Single-token autoregressive forward.
$tgN = 32,64
foreach ($n in $tgN) {
    $log = Join-Path $logdir "gpu_tg$n.json"
    Write-Host "=== TG n=$n ===" -ForegroundColor Cyan
    & $bench -m $model -ngl 99 -p 0 -n $n -r 5 -o json 2>$null | Out-File -FilePath $log -Encoding utf8
    Write-Host "  -> $log"
}

Write-Host "`nAll runs complete." -ForegroundColor Green
