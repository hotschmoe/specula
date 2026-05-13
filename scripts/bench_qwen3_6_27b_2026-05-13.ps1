# Qwen3.6-27B dense baseline matrix — session 26 follow-up.
# Q4_0 fits Adreno's 24 GB global mem; Q8_0 does not so OpenCL is -ngl 0 only.
# Memory bandwidth ceilings @ 228 GB/s:
#   - Q4_0 (16 GB): ~14 t/s
#   - Q8_0 (29 GB): ~8 t/s

param(
  [string]$Quant = "Q4_0"   # Q4_0 or Q8_0
)

# NOT 'Stop' — PS 5.1 wraps native exe stderr in NativeCommandError
# even at exit 0. Check $LASTEXITCODE explicitly if needed.
$ErrorActionPreference = "Continue"

$repo  = "C:\Users\hotschmoe\Documents\GitHub\specula"
$model = "$repo\models\Qwen3.6-27B-MTP-$Quant.gguf"
$out   = "$repo\results\csv\qwen3_6_27b_${Quant}_baseline_2026-05-13.md"
$binCpu = "$repo\llama.cpp\build-cpu-kleidiai\bin\llama-bench.exe"
$binOcl = "$repo\llama.cpp\build-opencl\bin\llama-bench.exe"
$binBatched = "$repo\llama.cpp\build-opencl\bin\llama-batched-bench.exe"

if (-not (Test-Path $model)) {
  Write-Error "Model not found: $model"
  exit 1
}

$sizeGB = (Get-Item $model).Length / 1GB
Write-Output "=== Qwen3.6-27B-MTP-$Quant : $([math]::Round($sizeGB,2)) GB ==="

"# Qwen3.6-27B-MTP-$Quant baseline matrix (2026-05-13)" | Out-File $out -Encoding utf8
"File: $model ($([math]::Round($sizeGB,2)) GB)" | Out-File $out -Append -Encoding utf8
"" | Out-File $out -Append -Encoding utf8

# CPU-kleidiai at -t 18 (35B rule: phys_cores for big models)
Write-Output "--- CPU-kleidiai -t 18 ---"
"## CPU-kleidiai -t 18" | Out-File $out -Append -Encoding utf8
& $binCpu -m $model -p 512 -n 128 -r 1 -t 18 --progress -o md 2>$null | Tee-Object -FilePath $out -Append

Write-Output "--- CPU-kleidiai -t 16 ---"
"" | Out-File $out -Append -Encoding utf8
"## CPU-kleidiai -t 16" | Out-File $out -Append -Encoding utf8
& $binCpu -m $model -p 512 -n 128 -r 1 -t 16 --progress -o md 2>$null | Tee-Object -FilePath $out -Append

# OpenCL -ngl 0 (the night's hero config)
Write-Output "--- OpenCL -ngl 0 -t 18 ---"
"" | Out-File $out -Append -Encoding utf8
"## OpenCL -ngl 0 -t 18 (the night's hero config)" | Out-File $out -Append -Encoding utf8
& $binOcl -m $model -p 512 -n 128 -r 1 -ngl 0 -t 18 --progress -o md 2>$null | Tee-Object -FilePath $out -Append

# OpenCL -ngl 99 — try both with and without the cl_qcom_large_buffer flag.
# Without flag: 24 GB cap → Q8_0 (29 GB) cannot fit. With flag: 44 GB BIOS
# allocation accessible → Q8_0 should fit.
Write-Output "--- OpenCL -ngl 99 -t 16 (no large buffer) ---"
"" | Out-File $out -Append -Encoding utf8
"## OpenCL -ngl 99 -t 16 (default; no large buffer)" | Out-File $out -Append -Encoding utf8
if ($Quant -eq "Q4_0") {
  & $binOcl -m $model -p 512 -n 128 -r 1 -ngl 99 -t 16 --progress -o md 2>$null | Tee-Object -FilePath $out -Append
} else {
  "(skipped: 29 GB > 24 GB Adreno default cap)" | Out-File $out -Append -Encoding utf8
}

Write-Output "--- OpenCL -ngl 99 -t 16 +LARGE_BUFFER ---"
"" | Out-File $out -Append -Encoding utf8
"## OpenCL -ngl 99 -t 16 +GGML_OPENCL_ADRENO_USE_LARGE_BUFFER=1" | Out-File $out -Append -Encoding utf8
$env:GGML_OPENCL_ADRENO_USE_LARGE_BUFFER = "1"
try {
  & $binOcl -m $model -p 512 -n 128 -r 1 -ngl 99 -t 16 --progress -o md 2>$null | Tee-Object -FilePath $out -Append
} catch {
  "ERROR: $_" | Out-File $out -Append -Encoding utf8
}
Remove-Item Env:GGML_OPENCL_ADRENO_USE_LARGE_BUFFER -ErrorAction SilentlyContinue

Write-Output "--- OpenCL -ngl 99 -t 18 -ub 512 +LARGE_BUFFER ---"
"" | Out-File $out -Append -Encoding utf8
"## OpenCL -ngl 99 -t 18 -ub 512 +LARGE_BUFFER" | Out-File $out -Append -Encoding utf8
$env:GGML_OPENCL_ADRENO_USE_LARGE_BUFFER = "1"
try {
  & $binOcl -m $model -p 512 -n 128 -r 1 -ngl 99 -t 18 -ub 512 --progress -o md 2>$null | Tee-Object -FilePath $out -Append
} catch {
  "ERROR: $_" | Out-File $out -Append -Encoding utf8
}
Remove-Item Env:GGML_OPENCL_ADRENO_USE_LARGE_BUFFER -ErrorAction SilentlyContinue

# Concurrency-4 on the likely-best config
Write-Output "--- Concurrency-4 OpenCL -ngl 0 -t 18 ---"
"" | Out-File $out -Append -Encoding utf8
"## Concurrency-4 (4 streams x 512 PP + 128 TG, ctx 4096), OpenCL -ngl 0 -t 18" | Out-File $out -Append -Encoding utf8
& $binBatched -m $model -c 4096 -b 2048 -ub 512 -ngl 0 -t 18 -npp 512 -ntg 128 -npl 4 2>$null | Tee-Object -FilePath $out -Append

Write-Output ""
Write-Output "=== DONE: results in $out ==="
