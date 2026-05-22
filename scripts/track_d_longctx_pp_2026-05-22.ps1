<#
.SYNOPSIS
  Track D (2026-05-22): long-context PP curve for Qwen3.6-35B-A3B MXFP4.
  Tests whether GPU offload (-ngl 99) overtakes the -ngl 0 coprocessor
  path on prefill as context grows, and sweeps -ub. FA-off: FA is a
  ~35% regression on Adreno OpenCL prefill (measured 2026-05-22).
#>
$ErrorActionPreference = 'Continue'
$root  = Split-Path $PSScriptRoot -Parent
$bench = Join-Path $root 'llama.cpp\build-opencl\bin\llama-bench.exe'
$model = Join-Path $root 'models\Qwen3.6-35B-A3B-MXFP4_MOE.gguf'
$out   = Join-Path $root 'results\csv\track_d_longctx_pp_2026-05-22.md'

"# Track D - long-ctx PP curve, Qwen3.6-35B-A3B MXFP4, OpenCL, -t 16, fa 0" | Out-File $out -Encoding utf8

function Run-Bench($label, [string[]]$ba) {
  "`n## $label" | Out-File $out -Append -Encoding utf8
  & $bench -m $model -t 16 -fa 0 @ba -o md 2>&1 |
    Where-Object { $_ -match '^\|' } | Out-File $out -Append -Encoding utf8
}

# A1: PP curve x ub, mid context (cheap)
Run-Bench 'PP 8192/32768 x ngl{0,99} x ub{512,2048}' @('-ngl','0,99','-ub','512,2048','-p','8192,32768','-n','0','-r','1')
# A2: PP at 128K (expensive)
Run-Bench 'PP 131072 x ngl{0,99}, ub 2048' @('-ngl','0,99','-ub','2048','-p','131072','-n','0','-r','1')
# B: TG at depth (decode slowdown vs context)
Run-Bench 'TG128 at depth 4096/32768 x ngl{0,99}' @('-ngl','0,99','-p','0','-n','128','-d','4096,32768','-r','1')

"`n# done $(Get-Date -Format s)" | Out-File $out -Append -Encoding utf8
