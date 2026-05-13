# Probe 1b: same TG-vs-ctx sweep on CPU-kleidiai, for direct comparison
# against the OpenCL -ngl 0 numbers from probe 1.
param([string]$Model = "35B")

$ErrorActionPreference = "Continue"
$repo = "C:\Users\hotschmoe\Documents\GitHub\specula"
$bin = "$repo\llama.cpp\build-cpu-kleidiai\bin\llama-bench.exe"
$out = "$repo\results\csv\probe1b_cpu_kleidiai_longctx_2026-05-13.md"

if ($Model -eq "35B") {
  $m = "$repo\models\Qwen3.6-35B-A3B-MXFP4_MOE.gguf"
  $title = "Qwen3.6-35B-A3B MXFP4 on CPU-kleidiai -t 18"
} elseif ($Model -eq "35B_q4km") {
  $m = "$repo\models\Qwen3.6-35B-A3B-Q4_K_M.gguf"
  $title = "Qwen3.6-35B-A3B Q4_K_M on CPU-kleidiai -t 18"
}

"# Probe 1b: $title, TG vs ctx depth (comparison to probe1 OpenCL -ngl 0)" | Out-File $out -Encoding utf8
"" | Out-File $out -Append -Encoding utf8
& $bin -m $m -p 512 -n 128 -r 1 -d 0,8192,32768,65536,131072 -t 18 --progress -o md 2>$null | Tee-Object -FilePath $out -Append
