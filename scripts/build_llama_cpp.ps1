<#
.SYNOPSIS
    Clone (or update) llama.cpp as a sibling checkout and build it for
    native Windows ARM64 under one of several specula presets.

.DESCRIPTION
    llama.cpp rejects MSVC on ARM64, so builds must use clang invoked from
    a vcvarsarm64 environment. This script sources vcvarsarm64, captures
    the resulting env, runs cmake+ninja with clang, optionally patches
    KleidiAI's .S sources for clang-on-Windows (needed only when
    -DGGML_CPU_KLEIDIAI=ON), then copies the runtime DLLs
    (msvcp140/vcruntime140/libomp140.aarch64) next to the built binaries.

    Build output goes to <repo>\build-<preset>\. Each preset gets its own
    directory so multiple configurations can coexist.

.PARAMETER Preset
    Which build configuration to produce:
        cpu             - CPU only, KleidiAI off. Matches the known-good
                          baseline recipe in gguf_models/LOCAL_LLM_NOTES.md.
        cpu-kleidiai    - CPU + KleidiAI ON for the Phase 1 SME2 retry.
                          Applies the clang-on-Windows .S guard patch.
        vulkan-opencl   - Vulkan + OpenCL/Adreno GPU backends enabled.
                          Target device is selected at runtime via
                          llama-cli / llama-bench flags.
        hexagon         - Placeholder. Hexagon NPU backend requires the
                          Qualcomm docker toolchain; this preset errors
                          out with a pointer to the llama.cpp docs.

.PARAMETER Commit
    Specific llama.cpp commit to check out. If empty, `git pull --ff-only`
    is run and the resulting HEAD is recorded.

.PARAMETER RepoDir
    Path to the llama.cpp checkout. Defaults to a sibling of specula/.

.PARAMETER Jobs
    Parallel build jobs. Default 12.

.PARAMETER LlvmRoot
    LLVM install root. Defaults to `C:\Program Files\LLVM`.

.PARAMETER VsRoot
    Visual Studio Build Tools root (for vcvarsarm64.bat). Defaults to the
    2022 BuildTools install path.

.EXAMPLE
    .\build_llama_cpp.ps1 -Preset vulkan-opencl
    .\build_llama_cpp.ps1 -Preset cpu-kleidiai -Commit abc1234
#>
[CmdletBinding()]
param(
    [ValidateSet('cpu', 'cpu-kleidiai', 'vulkan-opencl', 'hexagon')]
    [string]$Preset = 'vulkan-opencl',
    [string]$Commit = '',
    [string]$RepoDir = (Join-Path $PSScriptRoot '..\llama.cpp'),
    [int]$Jobs = 12,
    [string]$LlvmRoot = 'C:\Program Files\LLVM',
    [string]$VsRoot = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools'
)

$ErrorActionPreference = 'Stop'

# --- Hexagon is out of scope for this script ---------------------------------
if ($Preset -eq 'hexagon') {
    throw "The Hexagon NPU backend needs the Qualcomm toolchain docker image. See llama.cpp/docs/backend/snapdragon/README.md and build it separately; this script does not drive that flow."
}

# --- Locate vcvarsarm64 and clang --------------------------------------------
$vcvars = Join-Path $VsRoot 'VC\Auxiliary\Build\vcvarsarm64.bat'
if (-not (Test-Path $vcvars)) { throw "vcvarsarm64.bat not found at $vcvars. Set -VsRoot or install VS BuildTools." }

$clangC   = Join-Path $LlvmRoot 'bin\clang.exe'
$clangCxx = Join-Path $LlvmRoot 'bin\clang++.exe'
$clangRc  = Join-Path $LlvmRoot 'bin\llvm-rc.exe'
foreach ($p in @($clangC, $clangCxx, $clangRc)) {
    if (-not (Test-Path $p)) { throw "Not found: $p. Set -LlvmRoot or install LLVM." }
}

# clang_rt builtins path is version-scoped (e.g. ...\lib\clang\22\...). Auto-detect.
$rtCandidates = Get-ChildItem -Path (Join-Path $LlvmRoot 'lib\clang') -Directory -ErrorAction SilentlyContinue |
    Sort-Object { [int]($_.Name -replace '\D','') } -Descending |
    ForEach-Object { Join-Path $_.FullName 'lib\windows\clang_rt.builtins-aarch64.lib' } |
    Where-Object { Test-Path $_ }
if (-not $rtCandidates) { throw "clang_rt.builtins-aarch64.lib not found under $LlvmRoot\lib\clang\*\lib\windows\. Is LLVM installed with Windows ARM runtime support?" }
$clangRtLib = $rtCandidates[0]

# --- Import vcvarsarm64 env into this PowerShell session ---------------------
# vcvarsarm64 is a cmd.exe .bat. Run it in a cmd subshell, dump env, then
# project differences back into $env:*.
Write-Host "Importing vcvarsarm64 environment..." -ForegroundColor Cyan
$envDump = cmd.exe /c "`"$vcvars`" >nul 2>&1 && set"
if ($LASTEXITCODE -ne 0) { throw "vcvarsarm64.bat failed (exit $LASTEXITCODE)" }
foreach ($line in $envDump) {
    $eq = $line.IndexOf('=')
    if ($eq -gt 0) {
        $k = $line.Substring(0, $eq)
        $v = $line.Substring($eq + 1)
        Set-Item -Path "env:$k" -Value $v
    }
}

# --- Clone or update ---------------------------------------------------------
if (-not (Test-Path $RepoDir)) {
    Write-Host "Cloning llama.cpp into $RepoDir..." -ForegroundColor Cyan
    git clone https://github.com/ggml-org/llama.cpp $RepoDir
    if ($LASTEXITCODE -ne 0) { throw "git clone failed" }
}

Push-Location $RepoDir
try {
    if ($Commit) {
        git fetch origin
        git checkout $Commit
    } else {
        git pull --ff-only
    }
    if ($LASTEXITCODE -ne 0) { throw "git update failed" }
    $currentCommit = (git rev-parse HEAD).Trim()
    Write-Host "llama.cpp @ $currentCommit" -ForegroundColor Green

    # --- Preset -> cmake flags -----------------------------------------------
    $buildDir = "build-$Preset"
    switch ($Preset) {
        'cpu' {
            $presetFlags = @('-DGGML_CPU_KLEIDIAI=OFF')
            $patchKleidiai = $false
        }
        'cpu-kleidiai' {
            $presetFlags = @('-DGGML_CPU_KLEIDIAI=ON')
            $patchKleidiai = $true
        }
        'vulkan-opencl' {
            $presetFlags = @(
                '-DGGML_VULKAN=ON',
                '-DGGML_OPENCL=ON',
                '-DGGML_OPENCL_USE_ADRENO_KERNELS=ON',
                '-DGGML_CPU_KLEIDIAI=OFF'
            )
            $patchKleidiai = $false
        }
    }

    $commonFlags = @(
        "-B", $buildDir,
        "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_C_COMPILER=$clangC",
        "-DCMAKE_CXX_COMPILER=$clangCxx",
        "-DCMAKE_RC_COMPILER=$clangRc",
        '-DCMAKE_C_FLAGS=-DNOMINMAX -DWIN32_LEAN_AND_MEAN',
        '-DCMAKE_CXX_FLAGS=-DNOMINMAX -DWIN32_LEAN_AND_MEAN',
        "-DCMAKE_EXE_LINKER_FLAGS=`"$clangRtLib`"",
        "-DCMAKE_SHARED_LINKER_FLAGS=`"$clangRtLib`"",
        '-DGGML_NATIVE=ON',
        '-DLLAMA_CURL=OFF',
        '-DLLAMA_BUILD_TESTS=OFF'
    )

    Write-Host "Configuring $buildDir (preset: $Preset)..." -ForegroundColor Cyan
    & cmake @commonFlags @presetFlags
    if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }

    # --- KleidiAI patch for clang-on-Windows ---------------------------------
    # Must run AFTER configure (which fetches KleidiAI sources) and BEFORE build.
    if ($patchKleidiai) {
        $kleidiSrc = Join-Path $RepoDir "$buildDir\_deps\kleidiai_download-src\kai"
        if (-not (Test-Path $kleidiSrc)) {
            Write-Warning "Expected KleidiAI sources at $kleidiSrc but did not find them; skipping patch. The build may fail on .S assembly."
        } else {
            $patchScript = Join-Path $PSScriptRoot 'patch_kleidiai.py'
            if (-not (Test-Path $patchScript)) { throw "patch_kleidiai.py not found next to this script at $patchScript" }
            Write-Host "Patching KleidiAI .S files for clang-on-Windows..." -ForegroundColor Cyan
            & python $patchScript $kleidiSrc
            if ($LASTEXITCODE -ne 0) { throw "KleidiAI patch failed" }
        }
    }

    # --- Build ---------------------------------------------------------------
    $targets = @(
        'llama-cli',
        'llama-bench',
        'llama-batched-bench',
        'llama-speculative',
        'llama-speculative-simple',
        'llama-server'
    )
    Write-Host "Building $($targets -join ', ')..." -ForegroundColor Cyan
    & cmake --build $buildDir --config Release -j $Jobs --target @targets
    if ($LASTEXITCODE -ne 0) { throw "cmake build failed" }

    # --- Copy runtime DLLs next to the binaries ------------------------------
    # Without these, the exes fail with STATUS_DLL_NOT_FOUND (0xC0000135) at
    # launch. See gguf_models/LOCAL_LLM_NOTES.md.
    $binDir = Join-Path $RepoDir "$buildDir\bin"
    if (-not (Test-Path $binDir)) {
        # Older cmake layouts put exes under build\bin\Release\
        $altBin = Join-Path $RepoDir "$buildDir\bin\Release"
        if (Test-Path $altBin) { $binDir = $altBin }
    }

    $dllNames = @('msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll', 'libomp140.aarch64.dll')
    $searchRoots = @(
        "$env:VCToolsRedistDir\arm64\Microsoft.VC143.CRT",
        "$env:VCToolsRedistDir\arm64\Microsoft.VC143.OpenMP"
    )
    $copied = @()
    $missing = @()
    foreach ($dll in $dllNames) {
        $found = $null
        foreach ($root in $searchRoots) {
            if (-not $root) { continue }
            $c = Get-ChildItem -Path $root -Filter $dll -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($c) { $found = $c.FullName; break }
        }
        if (-not $found) {
            # Fallback: search the VC Tools install as a whole
            if ($env:VCINSTALLDIR) {
                $c = Get-ChildItem -Path $env:VCINSTALLDIR -Filter $dll -Recurse -ErrorAction SilentlyContinue |
                    Where-Object { $_.FullName -match '\\arm64\\' } |
                    Select-Object -First 1
                if ($c) { $found = $c.FullName }
            }
        }
        if ($found) {
            Copy-Item -Path $found -Destination $binDir -Force
            $copied += $dll
        } else {
            $missing += $dll
        }
    }
    if ($copied.Count)  { Write-Host "Copied runtime DLLs: $($copied -join ', ')" -ForegroundColor Green }
    if ($missing.Count) { Write-Warning "Could not locate: $($missing -join ', '). Binaries may fail with STATUS_DLL_NOT_FOUND until these are placed next to them." }

    # --- Report --------------------------------------------------------------
    $bins = Get-ChildItem "$binDir\llama-*.exe" -ErrorAction SilentlyContinue
    Write-Host ""
    Write-Host "Built binaries in $binDir" -ForegroundColor Green
    $bins | ForEach-Object { Write-Host "  $($_.Name)" }

    # Record commit + preset per build dir for reproducibility in CSV rows.
    $stampFile = Join-Path $RepoDir "$buildDir\SPECULA_BUILD.txt"
    @(
        "preset:     $Preset",
        "commit:     $currentCommit",
        "built:      $(Get-Date -Format o)",
        "clang_rt:   $clangRtLib",
        "vs_root:    $VsRoot",
        "llvm_root:  $LlvmRoot"
    ) | Out-File -FilePath $stampFile -Encoding utf8
    Write-Host "Wrote $stampFile" -ForegroundColor Green
}
finally {
    Pop-Location
}
