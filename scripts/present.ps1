<# Runs the five demonstration steps in order, pausing between them. #>
param([switch]$NoPause)
$ErrorActionPreference = 'Stop'
$steps = @(
    @{ Title = '1/5 Correctness: four methods, same six points, checked against brute force';
       Run = { & "$PSScriptRoot/demo.ps1" } },
    @{ Title = '2/5 Streamed tiles: actual ownership and query routing';
       Run = { & "$PSScriptRoot/demo.ps1" -Method rt-ooc } },
    @{ Title = '3/5 Bounded lists: one stored ID, counts unchanged';
       Run = { & "$PSScriptRoot/demo.ps1" -Method rt-ooc -MaxIds 1 } },
    @{ Title = '4/5 Executor budget on one million points';
       Run = { & "$PSScriptRoot/budget.ps1" } },
    @{ Title = '5/5 Speed versus device memory on one million points';
       Run = { & "$PSScriptRoot/compare.ps1" } }
)
foreach ($step in $steps) {
    Write-Host "`n==== $($step.Title) ====" -ForegroundColor Cyan
    & $step.Run
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
    if (-not $NoPause -and $step -ne $steps[-1]) { Read-Host 'Press Enter for the next step' | Out-Null }
}
