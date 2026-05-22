<#
.SYNOPSIS
  Track D tail (overnight) — completes the long-context PP sweep for
  Qwen3.6-35B-A3B MXFP4 that was stopped early on 2026-05-22.
  A1 already landed in track_d_longctx_pp_2026-05-22.md and showed
  -ngl 99 -ub 2048 winning prefill (174.4 t/s @ pp8192, +17% vs the
  -ngl 0 "blended" default). This finishes the three open points:
    1. the ngl99/ub2048/pp32768 cell A1 dropped (run isolated to see
       if it errors/OOMs);
    2. PP at 131072 for ngl {0,99} — does ngl 99 OOM at 128K
       (20.2 GB model + KV in 24.4 GB GPU memory)?
    3. TG-128 at depth 4K/32K/128K — decode slowdown vs context.
  FA-off throughout: -fa 1 is a ~35% prefill regression on Adreno OpenCL.
  Expect ~90 min. Safe to run unattended.
#>
$ErrorActionPreference = 'Continue'
$root  = Split-Path $PSScriptRoot -Parent
$bench = Join-Path $root 'llama.cpp\build-opencl\bin\llama-bench.exe'
$model = Join-Path $root 'models\Qwen3.6-35B-A3B-MXFP4_MOE.gguf'
$out   = Join-Path $root 'results\csv\track_d_longctx_pp_2026-05-22_tail.md'

"# Track D tail - Qwen3.6-35B-A3B MXFP4, OpenCL, -t 16, fa 0 (overnight)" | Out-File $out -Encoding utf8

function Run-Bench($label, [string[]]$ba) {
  "`n## $label  ($(Get-Date -Format HH:mm))" | Out-File $out -Append -Encoding utf8
  & $bench -m $model -t 16 -fa 0 @ba -o md 2>&1 |
    ForEach-Object { if ($_ -match '^\|') { $_ } elseif ($_ -match 'error|failed|alloc|OOM') { "<!-- $_ -->" } } |
    Out-File $out -Append -Encoding utf8
}

# 1: the missing A1 cell, isolated
Run-Bench 'PP 32768, ngl 99, ub 2048 (missing A1 cell)' @('-ngl','99','-ub','2048','-p','32768','-n','0','-r','1')
# 2: PP at 128K
Run-Bench 'PP 131072 x ngl{0,99}, ub 2048' @('-ngl','0,99','-ub','2048','-p','131072','-n','0','-r','1')
# 3: TG-128 at depth
Run-Bench 'TG128 at depth 4096/32768/131072 x ngl{0,99}' @('-ngl','0,99','-p','0','-n','128','-d','4096,32768,131072','-r','1')

"`n# done $(Get-Date -Format s)" | Out-File $out -Append -Encoding utf8
