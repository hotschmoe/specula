<#
.SYNOPSIS
    Build the SME probe as a native ARM64 executable, then run it.

.DESCRIPTION
    Uses the same clang-via-vcvarsarm64 recipe as build_llama_cpp.ps1.
    Compiles sme_probe.c + sme_probe.S with -march=armv9.2-a+sme so the
    assembler accepts the SME mnemonics, links as a single .exe, and
    runs it so we can read the per-instruction fault table.
#>
[CmdletBinding()]
param(
    [string]$LlvmRoot = 'C:\Program Files\LLVM',
    [string]$VsRoot   = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools'
)

$ErrorActionPreference = 'Continue'
$here = $PSScriptRoot

$vcvars = Join-Path $VsRoot 'VC\Auxiliary\Build\vcvarsarm64.bat'
if (-not (Test-Path $vcvars)) { throw "vcvarsarm64.bat not found at $vcvars" }

$clang = Join-Path $LlvmRoot 'bin\clang.exe'
if (-not (Test-Path $clang)) { throw "clang.exe not found at $clang" }

# Import vcvarsarm64 into this session (same trick build_llama_cpp.ps1 uses).
$envDump = cmd.exe /c "`"$vcvars`" >nul 2>&1 && set"
foreach ($line in $envDump) {
    $eq = $line.IndexOf('=')
    if ($eq -gt 0) {
        Set-Item -Path "env:$($line.Substring(0,$eq))" -Value $line.Substring($eq+1)
    }
}

$cfile = Join-Path $here 'sme_probe.c'
$sfile = Join-Path $here 'sme_probe.S'
$exe   = Join-Path $here 'sme_probe.exe'

# clang_rt builtins — needed for the same linkage reason build_llama_cpp.ps1 passes it.
$rt = Get-ChildItem -Path (Join-Path $LlvmRoot 'lib\clang') -Directory |
      Sort-Object { [int]($_.Name -replace '\D','') } -Descending |
      ForEach-Object { Join-Path $_.FullName 'lib\windows\clang_rt.builtins-aarch64.lib' } |
      Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $rt) { throw "clang_rt.builtins-aarch64.lib not found" }

Write-Host "Compiling $cfile + $sfile -> $exe" -ForegroundColor Cyan

# -march needs quoting — PS 5.1 splits bare `armv9.2-a+sme` at the dot
# (interprets `.2-a+sme` as a separate positional arg, which clang
# then treats as a filename and errors out with "no such file").
# Building the argv array explicitly and expanding with @args avoids
# that heuristic.
# -O0 keeps the trampolines verbatim (no instruction merging/reordering
# that could shift the expected PC offset our VEH relies on).
$clangArgs = @(
    '-O0',
    '-march=armv9.2-a+sme',
    '-o', $exe,
    $cfile, $sfile,
    '-lkernel32', '-lmsvcrt',
    $rt
)
& $clang @clangArgs
if ($LASTEXITCODE -ne 0) { throw "clang build failed (exit $LASTEXITCODE)" }

Write-Host ""
Write-Host "Running $exe" -ForegroundColor Cyan
Write-Host ""
& $exe
$code = $LASTEXITCODE
Write-Host ""
Write-Host "sme_probe exit: $code" -ForegroundColor ($(if ($code -eq 0) { 'Green' } else { 'Yellow' }))
