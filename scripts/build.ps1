<#
.SYNOPSIS
  Configures and builds ooc_rtnn with Ninja inside a Visual Studio x64 developer shell.
.EXAMPLE
  ./scripts/build.ps1                 # Release build into ./build
  ./scripts/build.ps1 -Config Debug -Clean
  ./scripts/build.ps1 -Test           # build, then run ctest
#>
param(
  [ValidateSet("Release", "RelWithDebInfo", "Debug")] [string] $Config = "Release",
  [string] $BuildDir = "build",
  [switch] $Clean,
  [switch] $Test,
  [string] $CudaArch = "89"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

if (-not $env:VSCMD_ARG_TGT_ARCH) {
  $installer = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer"
  $env:PATH = "$installer;$env:PATH"  # VsDevCmd itself calls vswhere
  $vswhere = Join-Path $installer "vswhere.exe"
  $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
  if (-not $vs) { throw "Visual Studio (or Build Tools) with the C++ x64 toolset was not found" }
  Import-Module (Join-Path $vs "Common7\Tools\Microsoft.VisualStudio.DevShell.dll")
  Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments "-arch=x64 -host_arch=x64" | Out-Null
}

$build = Join-Path $root $BuildDir
if ($Clean -and (Test-Path $build)) { Remove-Item -Recurse -Force $build }

cmake -S $root -B $build -G Ninja "-DCMAKE_BUILD_TYPE=$Config" "-DCMAKE_CUDA_ARCHITECTURES=$CudaArch"
if ($LASTEXITCODE) { exit $LASTEXITCODE }
cmake --build $build --parallel
if ($LASTEXITCODE) { exit $LASTEXITCODE }
if ($Test) {
  ctest --test-dir $build --output-on-failure
  exit $LASTEXITCODE
}
