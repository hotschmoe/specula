param(
    [string]$Model = '.\models\Qwen3-0.6B-Q8_0.gguf',
    [string]$Tag   = '0.6B',
    # Comma-separated row labels to run. Default = full matrix.
    [string]$Only  = 'B0,B1,B2,B3,B4,B5'
)

# Adreno Vulkan env-var matrix via llama-bench.
# See docs/adreno_debugging.md. Use `-Only` to skip known-bad rows
# (e.g. on Qwen3-8B the fp16-on rows B0/B1/B2 crash or are unusably
# slow; run `-Only B3,B4,B5`).

$bench  = '.\llama.cpp\build-vulkan\bin\llama-bench.exe'
$model  = $Model
$outdir = '.\results'
New-Item -ItemType Directory -Force -Path $outdir | Out-Null

$wanted = $Only -split ',' | ForEach-Object { $_.Trim() }

$rows = @(
    @{ label = 'B0-baseline';   vars = @{}; ngl = 99 },
    @{ label = 'B1-nocoop';     vars = @{ GGML_VK_DISABLE_COOPMAT = '1' }; ngl = 99 },
    @{ label = 'B2-nocoop12';   vars = @{ GGML_VK_DISABLE_COOPMAT = '1'; GGML_VK_DISABLE_COOPMAT2 = '1' }; ngl = 99 },
    @{ label = 'B3-nof16';      vars = @{ GGML_VK_DISABLE_F16 = '1' }; ngl = 99 },
    @{ label = 'B4-allsafe';    vars = @{ GGML_VK_DISABLE_COOPMAT = '1'; GGML_VK_DISABLE_COOPMAT2 = '1'; GGML_VK_DISABLE_F16 = '1' }; ngl = 99 },
    @{ label = 'B5-ngl0';       vars = @{}; ngl = 0 }
)

foreach ($row in $rows) {
    $label = $row.label
    $short = ($label -split '-')[0]
    if ($wanted -notcontains $short) { continue }
    $log = Join-Path $outdir "adreno-perf-$Tag-$label.log"
    Write-Host "=== $label ===" -ForegroundColor Cyan

    $setParts = @()
    foreach ($k in $row.vars.Keys) { $setParts += "set `"$k=$($row.vars[$k])`"" }
    $setClause = if ($setParts.Count) { ($setParts -join ' && ') + ' && ' } else { '' }

    $header = "### $label ($($setParts -join ';')) ngl=$($row.ngl)  started $(Get-Date -Format o)"
    Set-Content -Path $log -Value $header -Encoding utf8

    $cmdLine = "$setClause`"$bench`" -m `"$model`" -ngl $($row.ngl) -p 128,512 -n 64 -r 2 >> `"$log`" 2>&1"
    Write-Host "cmd /c $cmdLine"
    & cmd.exe /c $cmdLine
    Write-Host "  exit=$LASTEXITCODE  log=$log"
}

Write-Host "`nMatrix run complete." -ForegroundColor Green
