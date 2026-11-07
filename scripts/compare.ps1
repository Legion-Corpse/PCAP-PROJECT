<# Larger speed/memory demonstration. Preserves fresh raw rows; not the paper benchmark. #>
param([string]$Points = 'uniform:1M', [string]$TilePoints = '16K',
      [ValidateRange(1,4096)][int]$BudgetMB = 8,
      [ValidateRange(2,20)][int]$Repeat = 3)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $root 'build_c17/ooc_rtnn.exe'
if (-not (Test-Path -LiteralPath $exe)) { throw 'Build first: ./scripts/build.ps1' }
$outDir = Join-Path $root 'results/raw'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$json = Join-Path $outDir ('live_comparison_' + [guid]::NewGuid().ToString('N') + '.jsonl')
Write-Host "Live comparison: $Points; first call excluded from retained-search median."
& $exe --points $Points --method cuda-grid,rt-incore,rt-ooc --tile-points $TilePoints `
    --budget-mb $BudgetMB --window 65536 --max-neighbors 0 --repeat $Repeat `
    --verify sample --verify-samples 512 --json $json
if ($LASTEXITCODE) { exit $LASTEXITCODE }
$rows = @(Get-Content -LiteralPath $json | ForEach-Object { $_ | ConvertFrom-Json })
if ($rows.Count -ne 3 * $Repeat) { throw 'Incomplete live comparison output' }
$table = foreach ($method in @('cuda-grid','rt-incore','rt-ooc')) {
    $all = @($rows | Where-Object { $_.method -eq $method })
    if (@($all | Where-Object { $_.status -ne 'ok' -or $_.verify_checked -le 0 -or
        $_.verify_count_mismatches -ne 0 -or $_.verify_list_errors -ne 0 }).Count) {
        throw "Verification failed for $method"
    }
    $times = @($all | Where-Object { $_.repeat -gt 0 } | ForEach-Object { [double]$_.search_ms } | Sort-Object)
    $middle = [int][math]::Floor($times.Count / 2)
    $median = if ($times.Count % 2) { $times[$middle] } else { ($times[$middle-1] + $times[$middle]) / 2 }
    $device = ($all | Measure-Object executor_bytes -Maximum).Maximum
    if ($device -le 0) { throw 'Rebuild the application to obtain device allocation accounting.' }
    [PSCustomObject]@{ Method=$method; Build_ms=[math]::Round($all[0].index_ms,3);
        SearchMedian_ms=[math]::Round($median,3); Device_MiB=[math]::Round($device/1MB,3);
        PinnedTile_MiB=[math]::Round($all[0].host_tile_bytes/1MB,3) }
}
$table | Format-Table -AutoSize
Write-Host "Raw measurements: $json"
Write-Host 'One process per backend; one warm-up and repeated calls on a retained index. Verification is sampled.'
Write-Host 'Memory: explicit device search buffers and separate pinned host tiles; excludes context, construction, other host arrays, and total process peaks.'
Write-Host 'Report only the speed/memory trade-off shown by this run; it can change with workload and tile settings.'
