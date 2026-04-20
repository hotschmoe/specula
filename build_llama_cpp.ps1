<#
.SYNOPSIS
    Clone (or update) llama.cpp as a sibling checkout and build it for
    native Windows ARM64 with all backends we care about.

.DESCRIPTION
    Builds ONE binary with Vulkan + OpenCL/Adreno enabled so you can switch
    backends at runtime via --device. The Hexagon NPU backend is built
    separately because it currently requires the Qualcomm toolchain docker
    image and a different build recipe — see docs/backend/snapdragon/README.md
    in llama.cpp for those instructions.

.PARAMETER Commit
    Specific llama.cpp commit to check out. If empty, uses whatever HEAD is
    after `git pull`.

.PARAMETER Jobs
    Parallel build jobs. Default 12 (2 prime + 10 perf cores is a reasonable
    split for X2 Elite Extreme).

.EXAMPLE
    .\build_llama_cpp.ps1
    .\build_llama_cpp.ps1 -Commit abc1234
#>
[CmdletBinding()]
param(
    [string]$Commit = '',
    [int]$Jobs = 12,
    [string]$RepoDir = (Join-Path $PSScriptRoot '..\llama.cpp')
)

$ErrorActionPreference = 'Stop'

# --- Clone or update ---------------------------------------------------------
if (-not (Test-Path $RepoDir)) {
    Write-Host "Cloning llama.cpp..." -ForegroundColor Cyan
    git clone https://github.com/ggml-org/llama.cpp $RepoDir
}
Push-Location $RepoDir
try {
    if ($Commit) {
        git fetch origin
        git checkout $Commit
    } else {
        git pull --ff-only
    }
    $currentCommit = (git rev-parse HEAD).Trim()
    Write-Host "llama.cpp @ $currentCommit" -ForegroundColor Green

    # --- Configure: Vulkan + OpenCL/Adreno in one binary ---------------------
    $buildDir = "build-vulkan-opencl"
    cmake -B $buildDir `
        -DCMAKE_BUILD_TYPE=Release `
        -DGGML_VULKAN=ON `
        -DGGML_OPENCL=ON `
        -DGGML_OPENCL_USE_ADRENO_KERNELS=ON `
        -DLLAMA_CURL=OFF

    # --- Build ----------------------------------------------------------------
    cmake --build $buildDir --config Release -j $Jobs `
        --target llama-cli llama-bench llama-speculative llama-speculative-simple llama-server

    # --- Report ---------------------------------------------------------------
    $bins = Get-ChildItem "$buildDir\bin\Release\*.exe" -ErrorAction SilentlyContinue
    if (-not $bins) { $bins = Get-ChildItem "$buildDir\bin\*.exe" -ErrorAction SilentlyContinue }
    Write-Host ""
    Write-Host "Built binaries:" -ForegroundColor Green
    $bins | ForEach-Object { Write-Host "  $($_.FullName)" }

    # Record commit used for later result reproducibility
    $commitFile = Join-Path $PSScriptRoot '..\LLAMA_CPP_COMMIT.txt'
    "llama.cpp commit: $currentCommit`nbuilt: $(Get-Date -Format o)" | Out-File -FilePath $commitFile -Encoding utf8
}
finally {
    Pop-Location
}
