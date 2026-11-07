<# Small, fully verified input/output demo. -Method all prints a compact side-by-side view;
   a single method prints its full output, including the RT tile structure. Tiny-input timings
   are dominated by GPU setup, so they are hidden unless -Timing is given. #>
param([ValidateSet('cpu-grid','cuda-grid','rt-incore','rt-ooc','all')]
      [string]$Method = 'all', [ValidateRange(0,1024)][int]$MaxIds = 8,
      [ValidateRange(1,4294967295)][long]$TilePoints = 2, [switch]$Timing)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $root 'build_c17/radius_demo.exe'
if (-not (Test-Path -LiteralPath $exe)) {
    throw 'Build the project first: ./scripts/build.ps1'
}
$methods = if ($Method -eq 'all') { @('cpu-grid','cuda-grid','rt-incore','rt-ooc') } else { @($Method) }
$timingLine = '^(Timing|Memory) '
$summary = @()
$answers = [ordered]@{}
foreach ($backend in $methods) {
    $output = @(& $exe $backend (Join-Path $root 'examples/points.xyz') (Join-Path $root 'examples/queries.xyz') $MaxIds $TilePoints)
    $code = $LASTEXITCODE
    if ($code) { $output | ForEach-Object { Write-Host $_ }; exit $code }
    if ($methods.Count -eq 1) {
        $output | Where-Object { $Timing -or $_ -notmatch $timingLine } |
            ForEach-Object { Write-Host $(if ($Timing) { $_ } else { $_ -replace ' \| executor bytes: \d+' }) }
    } else {
        if (-not $answers.Count) {
            Write-Host "Radius: 1 | max stored IDs: $MaxIds | streamed tile capacity: $TilePoints"
            $output | Where-Object { $_ -match '^Stored points|^\s+\d+: ' } | ForEach-Object { Write-Host $_ }
        }
        Write-Host "`n[$backend]"
        # Byte counts on six points reflect fixed RT overhead, not the streamed working set.
        $output | Where-Object { $_ -match '^\s+Q\d+ \(|^PASS:|^RT tiles:' } |
            ForEach-Object { Write-Host ($_ -replace ' \| executor bytes: \d+') }
    }
    $answers[$backend] = @($output | Where-Object { $_ -match '^\s+Q\d+ \(' } | ForEach-Object { $_.Trim() })
    $timingRow = $output | Where-Object { $_ -match '^Timing \(ms\):' }
    if ($timingRow -notmatch 'build=([0-9.]+) search=([0-9.]+) build\+search=([0-9.]+) verification=([0-9.]+)') {
        throw "Timing output missing for $backend"
    }
    $row = [PSCustomObject]@{ Method=$backend; Build_ms=$Matches[1]; Search_ms=$Matches[2];
                            Build_Search_ms=$Matches[3]; Verification_ms=$Matches[4];
                            Device_KiB=0; PinnedTile_KiB=0 }
    $memory = $output | Where-Object { $_ -match '^Memory \(bytes\):' }
    if ($memory -notmatch 'device=(\d+) pinned_host_tiles=(\d+)') {
        throw "Memory output missing for $backend"
    }
    $row.Device_KiB = [math]::Round([double]$Matches[1] / 1024, 3)
    $row.PinnedTile_KiB = [math]::Round([double]$Matches[2] / 1024, 3)
    $summary += $row
}
if ($methods.Count -gt 1) {
    $counts = @($answers.Values | ForEach-Object { ($_ | ForEach-Object { $_ -replace ' IDs=.*' }) -join '|' })
    $lists = @($answers.Values | ForEach-Object { $_ -join '|' })
    if (@($counts | Select-Object -Unique).Count -ne 1) { throw 'Methods returned different counts' }
    Write-Host "`nAll $($methods.Count) methods returned identical counts, each verified against brute force."
    if (@($lists | Select-Object -Unique).Count -eq 1) {
        Write-Host 'Displayed ID lists are also identical.'
    } else {
        Write-Host 'Truncated lists may hold different valid IDs; each was checked for range, radius, and uniqueness.'
    }
    Write-Host 'Tile structure: ./scripts/demo.ps1 -Method rt-ooc'
}
if ($Timing) {
    Write-Host "`nMethod timing and memory summary (single tiny demo run):"
    $summary | Format-Table -AutoSize
    Write-Host 'GPU setup and run-to-run noise dominate small inputs; use benchmark results for speed comparisons.'
    Write-Host 'Device_KiB is explicit device search storage; CPU grid uses host memory, reported as zero device bytes.'
    Write-Host 'PinnedTile_KiB is only the RT host tile store. Neither column is whole-process memory.'
}
