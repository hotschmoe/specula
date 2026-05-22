# Phase 1 native spec-decode benchmark.
# CPU-only (C-CPU) and GPU-only (C3) speculative decoding via llama-speculative,
# plus target-4B-alone TG baselines via llama-bench. Fixed prompt, n=128, AC.

$ErrorActionPreference = "Continue"
$root   = "C:\Users\hotschmoe\Documents\GitHub\specula"
$cpuBin = "$root\llama.cpp\build-cpu\bin"
$gpuBin = "$root\llama.cpp\build-opencl\bin"
$target = "$root\models\Qwen3-4B-Q4_0.gguf"
$draft  = "$root\models\Qwen3-0.6B-Q8_0.gguf"
$logs   = "$root\gpu_npu_sidequest\logs"
$prompt = "Write a detailed technical explanation of how speculative decoding accelerates large language model inference, covering the draft model, the target model, the verification step, and why acceptance rate matters."

New-Item -ItemType Directory -Force -Path $logs | Out-Null

# --- spec-decode runs ---
foreach ($cfg in @(
    @{name="cpu"; spec="$cpuBin\llama-speculative.exe"; extra=@("-t","16")},
    @{name="gpu"; spec="$gpuBin\llama-speculative.exe"; extra=@("-ngl","99","-ngld","99")}
)) {
    foreach ($k in @(2,4,8)) {
        $tag = "specdecode_$($cfg.name)_k$k"
        Write-Host "=== $tag ==="
        $args = @("-m",$target,"-md",$draft,"-p",$prompt,"-n","128",
                  "--spec-draft-n-max","$k","--spec-draft-n-min","0",
                  "--temp","0","-c","2048") + $cfg.extra
        & $cfg.spec @args 2>&1 | Tee-Object -FilePath "$logs\$tag.log"
        Write-Host ""
    }
}

# --- target-4B-alone TG baselines ---
Write-Host "=== baseline cpu target tg ==="
& "$cpuBin\llama-bench.exe" -m $target -p 0 -n 128 -t 16 -r 5 -o json 2>&1 |
    Tee-Object -FilePath "$logs\baseline_cpu_target_tg.json"
Write-Host "=== baseline gpu target tg ==="
& "$gpuBin\llama-bench.exe" -m $target -ngl 99 -p 0 -n 128 -r 5 -o json 2>&1 |
    Tee-Object -FilePath "$logs\baseline_gpu_target_tg.json"

# --- 0.6B draft per-step cost ---
Write-Host "=== draft cpu tg ==="
& "$cpuBin\llama-bench.exe" -m $draft -p 0 -n 128 -t 16 -r 5 -o json 2>&1 |
    Tee-Object -FilePath "$logs\draft_cpu_tg.json"
Write-Host "=== draft gpu tg ==="
& "$gpuBin\llama-bench.exe" -m $draft -ngl 99 -p 0 -n 128 -r 5 -o json 2>&1 |
    Tee-Object -FilePath "$logs\draft_gpu_tg.json"

Write-Host "DONE"
