# Qwen3.6-27B MTP bench -- PR #22673 build (build-opencl-mtp/).
# Compares no-MTP vs --spec-type draft-mtp acceleration on the same binary.

param(
  [string]$Quant = "Q4_0"
)

$ErrorActionPreference = "Continue"

$repo  = "C:\Users\hotschmoe\Documents\GitHub\specula"
$model = "$repo\models\Qwen3.6-27B-MTP-$Quant.gguf"
$out   = "$repo\results\csv\qwen3_6_27b_${Quant}_mtp_2026-05-13.md"
$binBench = "$repo\llama.cpp\build-opencl-mtp\bin\llama-bench.exe"
$binSpec  = "$repo\llama.cpp\build-opencl-mtp\bin\llama-speculative-simple.exe"
$binBatched = "$repo\llama.cpp\build-opencl-mtp\bin\llama-batched-bench.exe"

if (-not (Test-Path $model)) { Write-Error "Model not found: $model"; exit 1 }

$sizeGB = (Get-Item $model).Length / 1GB
Write-Output "=== Qwen3.6-27B-MTP-$Quant : $([math]::Round($sizeGB,2)) GB on PR #22673 (e7b484815) ==="

"# Qwen3.6-27B-MTP-$Quant on PR #22673 build (2026-05-13)" | Out-File $out -Encoding utf8
"Binary: $binBench / $binSpec" | Out-File $out -Append -Encoding utf8
"File: $([math]::Round($sizeGB,2)) GB" | Out-File $out -Append -Encoding utf8
"" | Out-File $out -Append -Encoding utf8

# --- Plain llama-bench baseline (no MTP) -----------------------------------
Write-Output "--- OpenCL -ngl 0 -t 18 (no MTP) ---"
"## OpenCL -ngl 0 -t 18 (no MTP -- coprocessor mode)" | Out-File $out -Append -Encoding utf8
& $binBench -m $model -p 512 -n 128 -r 1 -ngl 0 -t 18 --progress -o md 2>$null | Tee-Object -FilePath $out -Append

Write-Output "--- OpenCL -ngl 0 -t 16 (no MTP) ---"
"" | Out-File $out -Append -Encoding utf8
"## OpenCL -ngl 0 -t 16 (no MTP)" | Out-File $out -Append -Encoding utf8
& $binBench -m $model -p 512 -n 128 -r 1 -ngl 0 -t 16 --progress -o md 2>$null | Tee-Object -FilePath $out -Append

Write-Output "--- OpenCL -ngl 99 -t 16 (no MTP, no LARGE_BUFFER) ---"
"" | Out-File $out -Append -Encoding utf8
"## OpenCL -ngl 99 -t 16 (no MTP, default 24GB cap)" | Out-File $out -Append -Encoding utf8
if ($Quant -eq "Q4_0") {
  & $binBench -m $model -p 512 -n 128 -r 1 -ngl 99 -t 16 --progress -o md 2>$null | Tee-Object -FilePath $out -Append
} else {
  "(skipped: 29 GB > 24 GB default cap)" | Out-File $out -Append -Encoding utf8
}

Write-Output "--- OpenCL -ngl 99 -t 16 +LARGE_BUFFER ---"
"" | Out-File $out -Append -Encoding utf8
"## OpenCL -ngl 99 -t 16 +GGML_OPENCL_ADRENO_USE_LARGE_BUFFER=1" | Out-File $out -Append -Encoding utf8
$env:GGML_OPENCL_ADRENO_USE_LARGE_BUFFER = "1"
& $binBench -m $model -p 512 -n 128 -r 1 -ngl 99 -t 16 --progress -o md 2>$null | Tee-Object -FilePath $out -Append
Remove-Item Env:GGML_OPENCL_ADRENO_USE_LARGE_BUFFER -ErrorAction SilentlyContinue

Write-Output "--- OpenCL -ngl 99 -t 18 +LARGE_BUFFER -ub 512 ---"
"" | Out-File $out -Append -Encoding utf8
"## OpenCL -ngl 99 -t 18 -ub 512 +LARGE_BUFFER" | Out-File $out -Append -Encoding utf8
$env:GGML_OPENCL_ADRENO_USE_LARGE_BUFFER = "1"
& $binBench -m $model -p 512 -n 128 -r 1 -ngl 99 -t 18 -ub 512 --progress -o md 2>$null | Tee-Object -FilePath $out -Append
Remove-Item Env:GGML_OPENCL_ADRENO_USE_LARGE_BUFFER -ErrorAction SilentlyContinue

# --- MTP self-draft via llama-speculative-simple ---------------------------
$prompt = "Explain in detail how a transformer language model performs next token prediction, covering attention, MLPs, and the residual stream. Be specific about dimensions for a 27B parameter model with 5120 hidden size."

Write-Output "--- MTP --spec-type draft-mtp -ngl 0 -t 18 ---"
"" | Out-File $out -Append -Encoding utf8
"## MTP --spec-type draft-mtp -ngl 0 -t 18 --spec-draft-n-max 2 (per unsloth README)" | Out-File $out -Append -Encoding utf8
$r = & $binSpec -m $model -p $prompt -n 256 --spec-type draft-mtp --device none -ngl 0 -t 18 --spec-draft-n-max 2 2>&1 | Out-String
Add-Content $out $r
Write-Output "  Filtered:"
$r -split "`n" | Where-Object { $_ -match "encoded|decoded|n_drafted|n_accept|accept|n_predict|speed" } | Select-Object -First 6 | ForEach-Object { Write-Output "    $($_.Trim())" }

Write-Output "--- MTP --spec-type draft-mtp -ngl 0 -t 18 -nmax 4 ---"
"" | Out-File $out -Append -Encoding utf8
"## MTP -ngl 0 -t 18 --spec-draft-n-max 4" | Out-File $out -Append -Encoding utf8
$r = & $binSpec -m $model -p $prompt -n 256 --spec-type draft-mtp --device none -ngl 0 -t 18 --spec-draft-n-max 4 2>&1 | Out-String
Add-Content $out $r
$r -split "`n" | Where-Object { $_ -match "encoded|decoded|n_drafted|n_accept|accept|n_predict|speed" } | Select-Object -First 6 | ForEach-Object { Write-Output "    $($_.Trim())" }

Write-Output "--- MTP --spec-type draft-mtp -ngl 99 -t 16 +LARGE_BUFFER ---"
"" | Out-File $out -Append -Encoding utf8
"## MTP -ngl 99 -t 16 +LARGE_BUFFER --spec-draft-n-max 2" | Out-File $out -Append -Encoding utf8
$env:GGML_OPENCL_ADRENO_USE_LARGE_BUFFER = "1"
$r = & $binSpec -m $model -p $prompt -n 256 --spec-type draft-mtp --device GPUOpenCL -ngl 99 -t 16 --spec-draft-n-max 2 2>&1 | Out-String
Add-Content $out $r
$r -split "`n" | Where-Object { $_ -match "encoded|decoded|n_drafted|n_accept|accept|n_predict|speed" } | Select-Object -First 6 | ForEach-Object { Write-Output "    $($_.Trim())" }
Remove-Item Env:GGML_OPENCL_ADRENO_USE_LARGE_BUFFER -ErrorAction SilentlyContinue

# --- Baseline equivalent: same prompt, no MTP, for direct comparison -------
Write-Output "--- Baseline (no spec) llama-cli -ngl 0 -t 18 ---"
"" | Out-File $out -Append -Encoding utf8
"## Baseline (no MTP) -ngl 0 -t 18 via llama-speculative-simple" | Out-File $out -Append -Encoding utf8
# Trick: --spec-type none + no draft model = vanilla generation. Use spec-simple binary so timing format matches.
$r = & $binSpec -m $model -p $prompt -n 256 --device none -ngl 0 -t 18 --spec-type none 2>&1 | Out-String
Add-Content $out $r
$r -split "`n" | Where-Object { $_ -match "encoded|decoded|tokens per second|eval time|speed" } | Select-Object -First 6 | ForEach-Object { Write-Output "    $($_.Trim())" }

Write-Output ""
Write-Output "=== DONE: $out ==="
