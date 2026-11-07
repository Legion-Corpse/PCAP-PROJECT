$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    & "$PSScriptRoot/build.ps1" -Test
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
    $env:OOC_RTNN = "$projectRoot/build_c17/ooc_rtnn.exe"
    & "$projectRoot/.venv/Scripts/python.exe" -m pytest -q -o addopts=
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
    & "$projectRoot/.venv/Scripts/python.exe" bench/c17_evidence.py --validate results/c17_20261029_v2/manifest.json
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
    & "$projectRoot/.venv/Scripts/python.exe" bench/c17_lifecycle.py --validate results/c17_lifecycle_20261029/manifest.json
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
    & "$PSScriptRoot/demo.ps1" -Method all
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
} finally { Pop-Location }
