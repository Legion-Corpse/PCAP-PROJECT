<# Executor-budget demonstration on a larger input: an oversized tile is rejected before any
   measurement, then streamed RT completes the same search under shrinking device budgets. #>
param([string]$Points = 'uniform:1M', [string]$TilePoints = '16K', [string]$OversizedTile = '1M',
      [ValidateRange(1,4096)][int]$BudgetMB = 8, [ValidateRange(1,4096)][int]$SmallBudgetMB = 2)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $root 'build_c17/ooc_rtnn.exe'
if (-not (Test-Path -LiteralPath $exe)) { throw 'Build first: ./scripts/build.ps1' }
$outDir = Join-Path $root 'results/raw'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$common = @('--points', $Points, '--max-neighbors', '0', '--verify', 'sample', '--verify-samples', '512')

function Invoke-Run([string]$label, [string[]]$extra) {
    $json = Join-Path $outDir ('budget_demo_' + [guid]::NewGuid().ToString('N') + '.jsonl')
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $text = @(& $exe @common @extra --json $json 2>&1 | ForEach-Object { "$_" })
    $code = $LASTEXITCODE
    $ErrorActionPreference = $previous
    $rows = @(if (Test-Path -LiteralPath $json) { Get-Content -LiteralPath $json | ForEach-Object { $_ | ConvertFrom-Json } })
    [PSCustomObject]@{ Label=$label; Code=$code; Text=$text; Rows=$rows }
}

function Get-Row($run, [string]$budget) {
    if ($run.Code -or $run.Rows.Count -ne 1) { $run.Text | Write-Host; throw "$($run.Label) failed" }
    $r = $run.Rows[0]
    if ($r.status -ne 'ok' -or $r.verify_checked -le 0 -or $r.verify_count_mismatches -or $r.verify_list_errors) {
        throw "Verification failed for $($run.Label)"
    }
    [PSCustomObject]@{ Run=$run.Label; Budget=$budget; Tiles=$r.tiles; Loads=$r.loads;
        Executor_MiB=[math]::Round($r.executor_bytes/1MB,3); PinnedTile_MiB=[math]::Round($r.host_tile_bytes/1MB,3);
        Search_ms=[math]::Round($r.search_ms,1); Neighbors=$r.neighbors_total; Checked=$r.verify_checked }
}

Write-Host "Budget demonstration on $Points (count-only, first search on a fresh index, 512 sampled checks)."
Write-Host "`n1) Resident RT keeps the whole index on the GPU (no executor budget applies)."
$resident = Get-Row (Invoke-Run 'rt-incore' @('--method', 'rt-incore')) 'none'

Write-Host "2) Streamed RT, $OversizedTile-point tiles, $BudgetMB MiB budget: two slots cannot fit."
$rejected = Invoke-Run 'rt-ooc oversized' @('--method', 'rt-ooc', '--tile-points', $OversizedTile, '--budget-mb', "$BudgetMB")
if (-not $rejected.Code -or $rejected.Rows.Count) { throw 'Expected the oversized-tile run to be rejected' }
$rejected.Text | Where-Object { $_ -match 'error' } | ForEach-Object { Write-Host "   -> $_" }
Write-Host '   Rejected before measurement; no result row was written.'

Write-Host "3) Streamed RT, $TilePoints-point tiles, $BudgetMB MiB and then $SmallBudgetMB MiB budgets."
$fit = Get-Row (Invoke-Run "rt-ooc $BudgetMB MiB" @('--method', 'rt-ooc', '--tile-points', $TilePoints, '--budget-mb', "$BudgetMB")) "$BudgetMB MiB"
$small = Get-Row (Invoke-Run "rt-ooc $SmallBudgetMB MiB" @('--method', 'rt-ooc', '--tile-points', $TilePoints, '--budget-mb', "$SmallBudgetMB")) "$SmallBudgetMB MiB"

$table = @($resident, $fit, $small)
$table | Format-Table -AutoSize
if (@($table | ForEach-Object { $_.Neighbors } | Select-Object -Unique).Count -ne 1) {
    throw 'Total neighbor counts differ between runs'
}
Write-Host 'All three runs found the same total neighbor count and passed sampled verification.'
Write-Host 'A smaller budget gives smaller query windows, so tiles are reloaded more often (Loads).'
Write-Host 'Executor_MiB is the checked device working set; PinnedTile_MiB is the host tile store it streams from.'
Write-Host 'Neither column is whole-process memory. One run per row; use the report for timing comparisons.'
Write-Host "Raw rows: $outDir\budget_demo_*.jsonl"
