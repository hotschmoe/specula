<#
.SYNOPSIS
    Start the daily-driver llama-server for opencode / agent-harness use.

.DESCRIPTION
    Wraps the canonical Qwen3.6-35B-A3B serve config from
    `daily_driver/recipe.md`. As of session 27 (2026-05-13) **the
    default backend is OpenCL with `-ngl 0`** — the "coprocessor"
    mode discovered in the overnight perf sprint (session 26) and
    validated against long context + delta-prefill in session 27.

    Why no longer Vulkan: the prior canonical config
    `GGML_VK_DISABLE_F16=1 + GGML_VK_PREFER_HOST_MEMORY=1` segfaults
    at model load on llama.cpp `856c3adac` (session 25 sweep). With
    only HOST_MEMORY set, Vulkan PP collapses to ~6 t/s. Vulkan is
    deprecated until upstream restores the F16-off path.

    Why OpenCL `-ngl 0` instead of `-ngl 99`: full GPU offload
    cratters TG on this Adreno (TG 16 at -ngl 99 vs TG 28-31 at
    -ngl 0). `-ngl 0` keeps weights in CPU RAM but the OpenCL
    backend stays registered and accelerates some ops.

    Session 27 long-ctx probe (Qwen3.6-35B-A3B MXFP4_MOE, OpenCL -ngl 0 -t 18):
      d=0:    PP 198 / TG 28.7
      d=8k:   PP 132 / TG 24.8
      d=32k:  PP  59 / TG 14.5
      (d=131k preroll exceeded 60-min budget; extrapolation ~TG 5-8)

    Session 27 delta-prefill probe (server with cache_prompt=true):
      Turn 1 cold (5k tokens):       wall 86.8s,  PP 64.8, TG 27.6
      Turn 2 (5k base + 524 delta):  wall 13.5s,  PP 44.0, TG 28.0
      Turn 3 (rerun, full cache):    wall  1.6s,  PP 57.8, TG 28.4
    Cache reuse on OpenCL -ngl 0 works flawlessly — 4941/5461 tokens
    reused on turn 2, full reuse on turn 3.

    No --no-warmup needed (no Vulkan shader-JIT to defer). No
    --flash-attn flag needed (FA-with-f16-KV livelock from the
    old Vulkan recipe doesn't apply on OpenCL -ngl 0).

.PARAMETER Backend
    'opencl' (default, new recommended), 'cpu' (fallback), or
    'vulkan' (deprecated — broken PP on 856c3adac, kept for
    rollback only).

.PARAMETER Port
    Listen port. Default 8080.

.PARAMETER Ctx
    Context size. Default 131072 (128k). Set lower if you want to
    cap memory headroom.

.PARAMETER Threads
    CPU thread count. Default 18 (all physical cores — session 26
    found `-t 18` optimal for the 35B-A3B; the `-t 16` rule applies
    only to ≤14B models on this hardware).

.EXAMPLE
    # Default: OpenCL -ngl 0 canonical (session 27 winner)
    .\scripts\serve_daily_driver.ps1

.EXAMPLE
    # CPU fallback (slightly slower than OpenCL -ngl 0 but simpler env)
    .\scripts\serve_daily_driver.ps1 -Backend cpu

.EXAMPLE
    # Vulkan rollback — only if some future llama.cpp revision restores
    # the DISABLE_F16 path; currently segfaults at load.
    .\scripts\serve_daily_driver.ps1 -Backend vulkan

.EXAMPLE
    # Custom port + smaller ctx
    .\scripts\serve_daily_driver.ps1 -Port 9090 -Ctx 32768
#>
[CmdletBinding()]
param(
    [ValidateSet('opencl', 'cpu', 'vulkan')][string]$Backend = 'opencl',
    [int]$Port = 8080,
    [int]$Ctx  = 131072,
    [int]$Threads = 18
)

$ErrorActionPreference = 'Continue'

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..')

if ($Backend -eq 'opencl') {
    $exe   = Join-Path $repoRoot 'llama.cpp\build-opencl\bin\llama-server.exe'
    $model = Join-Path $repoRoot 'models\Qwen3.6-35B-A3B-MXFP4_MOE.gguf'
    # -ngl 0: OpenCL backend registered, no model layers offloaded.
    # The Adreno path still accelerates select ops while weights stay
    # in CPU RAM. Session 26-27 measured this as 10-20% TG over pure
    # CPU at every context depth, with no shader-JIT first-request hit.
    $extra = @('-ngl', '0', '-t', $Threads)
} elseif ($Backend -eq 'vulkan') {
    $exe   = Join-Path $repoRoot 'llama.cpp\build-vulkan\bin\llama-server.exe'
    $model = Join-Path $repoRoot 'models\Qwen3.6-35B-A3B-MXFP4_MOE.gguf'
    $extra = @(
        '-ngl', '99',
        '--flash-attn', 'off',
        '--no-warmup'
    )
    $env:GGML_VK_DISABLE_F16        = '1'
    $env:GGML_VK_PREFER_HOST_MEMORY = '1'
    Write-Host "  WARNING: Vulkan -DISABLE_F16 path segfaults on llama.cpp 856c3adac;" -ForegroundColor Yellow
    Write-Host "           server will likely fail at model load. Use -Backend opencl." -ForegroundColor Yellow
} else {
    # cpu fallback - simpler env, slightly lower TG vs opencl -ngl 0
    $exe   = Join-Path $repoRoot 'llama.cpp\build-cpu-kleidiai\bin\llama-server.exe'
    $model = Join-Path $repoRoot 'models\Qwen3.6-35B-A3B-Q4_K_M.gguf'
    $extra = @('-t', $Threads)
}

if (-not (Test-Path $exe))   { throw "Server binary not found: $exe" }
if (-not (Test-Path $model)) { throw "Model not found: $model -- fetch via daily_driver/recipe.md step 1" }

$serverArgs = @(
    '-m', $model,
    '-c', $Ctx,
    '--alias', 'daily-driver',
    '--host', '127.0.0.1',
    '--port', $Port
) + $extra

Write-Host ""
Write-Host "daily-driver llama-server"            -ForegroundColor Cyan
Write-Host "  backend : $Backend"
Write-Host "  model   : $(Split-Path $model -Leaf)"
Write-Host "  ctx     : $Ctx"
Write-Host "  endpoint: http://127.0.0.1:$Port  (OpenAI-compatible at /v1)"
if ($Backend -eq 'opencl') {
    Write-Host "  note    : -ngl 0 = OpenCL backend as coprocessor, weights stay in CPU RAM" -ForegroundColor Cyan
    Write-Host "            measured TG @ d=0/8k/32k: 28.7 / 24.8 / 14.5 t/s on 35B-A3B MXFP4"
    Write-Host "            cold start ~30-90 s for first reasonable prompt (no shader-JIT hit)"
}
if ($Backend -eq 'vulkan') {
    Write-Host "  note    : --no-warmup means the FIRST request will be slow" -ForegroundColor Yellow
    Write-Host "            (Vulkan shader JIT + 16k cold prefill ~ 5-6 min)"
    Write-Host "            AND: F16-off path broken on 856c3adac, may segfault at load" -ForegroundColor Red
}
Write-Host ""
Write-Host "Point opencode (or any OpenAI-compat client) at:"
Write-Host "  base URL: http://127.0.0.1:$Port/v1"   -ForegroundColor Green
Write-Host "  api key : (any non-empty string)"
Write-Host ""
Write-Host "Ctrl+C to stop."
Write-Host ""

& $exe @serverArgs
