# Compiles SolveBlackwoodKernel.cu -> SolveBlackwoodKernel.ptx (the production kernel).
#
# BlackwoodGpuEngine loads the .ptx by relative path at startup, so run this from the
# repository root before the first run, and again after any change to the .cu.
#
# -arch: set to your GPU's compute capability. compute_120 is Blackwood-era Ada/Blackwell;
# use e.g. compute_86 (Ampere) or compute_89 (Ada) if nvcc rejects it. Override with:
#   .\build-kernel.ps1 -Arch compute_86
param([string]$Arch = "compute_120")

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

# nvcc needs cl.exe on PATH even for -ptx-only output.
if ($null -eq (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        $found = Get-ChildItem -Path "$vsPath\VC\Tools\MSVC" -Filter cl.exe -Recurse -ErrorAction SilentlyContinue |
                 Where-Object { $_.FullName -match "Hostx64\\x64" } | Select-Object -First 1
        if ($null -ne $found) { $env:PATH = "$($found.Directory.FullName);$env:PATH" }
    }
    if ($null -eq (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        throw "cl.exe not found -- nvcc cannot compile without the MSVC toolchain on PATH."
    }
}

nvcc -ptx -O3 -arch=$Arch `
    (Join-Path $root "SolveBlackwoodKernel.cu") `
    -o (Join-Path $root "SolveBlackwoodKernel.ptx")

if ($LASTEXITCODE -ne 0) { throw "nvcc failed with exit code $LASTEXITCODE" }
Write-Output "Built SolveBlackwoodKernel.ptx (arch $Arch)"
