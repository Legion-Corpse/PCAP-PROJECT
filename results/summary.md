# Measured results

Generated from the validated primary and lifecycle manifests. Time values are medians in milliseconds unless noted. The main matrix uses five independent processes per configuration. Each row has sampled verification for 512 deterministic query positions. Cold first answer ends before verification; retained search reuses the index and includes that backend's query ordering, routing, transfer, execution, and result handling.

The test machine was a Windows 11 laptop with an RTX 4060 Laptop GPU and 8 GB device memory. GPU clocks were not locked. The serial CPU grid is a reference baseline, not a tuned multicore implementation.

## Primary comparison

| Workload | Backend | Build | Retained search | Cold first answer |
|---|---|---:|---:|---:|
| Uniform 100K | cpu-grid | 3.68 | 94.55 | 112.85 |
| Uniform 100K | cuda-grid | 80.16 | 3.00 | 83.25 |
| Uniform 100K | rt-incore | 203.19 | 12.82 | 212.89 |
| Uniform 100K | rt-ooc | 209.75 | 12.25 | 222.45 |
| Uniform 1M | cpu-grid | 37.18 | 1431.38 | 1380.11 |
| Uniform 1M | cuda-grid | 86.04 | 26.68 | 112.15 |
| Uniform 1M | rt-incore | 349.01 | 154.98 | 510.34 |
| Uniform 1M | rt-ooc | 438.05 | 327.85 | 761.22 |
| Uniform 4M | cpu-grid | 174.66 | 8985.10 | 8899.23 |
| Uniform 4M | cuda-grid | 115.91 | 487.38 | 612.04 |
| Uniform 4M | rt-incore | 1564.73 | 911.06 | 2419.92 |
| Uniform 4M | rt-ooc | 1150.95 | 2823.02 | 3817.69 |
| Scene-like 200K | cpu-grid | 12.42 | 173.19 | 190.59 |
| Scene-like 200K | cuda-grid | 78.03 | 4.82 | 82.80 |
| Scene-like 200K | rt-incore | 261.44 | 26.00 | 283.69 |
| Scene-like 200K | rt-ooc | 231.09 | 29.89 | 259.63 |
| Armadillo 172,974 | cpu-grid | 10.55 | 90.52 | 100.10 |
| Armadillo 172,974 | cuda-grid | 94.74 | 1.59 | 96.55 |
| Armadillo 172,974 | rt-incore | 239.89 | 15.72 | 257.86 |
| Armadillo 172,974 | rt-ooc | 223.69 | 18.06 | 261.40 |

The CUDA linked-cell grid has the shortest retained-search median in all five tested workloads. This is a result for the implementations and machine above; it does not compare against an optimized multicore CPU library or other GPUs.

## Streamed RT capacity and planner

| 1M configuration | Retained search | Observed range |
|---|---:|---:|
| 64K/on (default) | 327.85 | 262.40 to 350.14 |
| 16K/on | 714.25 | 651.22 to 848.10 |
| 256K/on | 196.50 | 167.82 to 252.29 |
| 16K/off | 787.84 | 751.35 to 865.30 |
| 64K/off | 330.78 | 314.78 to 394.50 |
| automatic | 163.69 | 149.15 to 226.02 |

Automatic selection uses a finite calibration set for same-point, count-only queries. Calibration time is part of setup. Predictions exclude host sorting, routing, and scatter. The 1M automatic trial selected a one-million-point tile, so this result is not evidence of a multi-tile streaming speedup.

## Explicit allocation quantities at 4M points

| Backend | Executor buffers MiB | Pinned tile store MiB |
|---|---:|---:|
| rt-incore | 76.598 | 70.598 |
| rt-ooc | 9.267 | 70.462 |

Executor bytes do not include context state, construction, calibration, input arrays, sorting temporaries, or total process memory. The Armadillo reconstruction fits physical device memory and does not establish real-data performance beyond VRAM.

## Retained-index lifecycle

Each lifecycle process builds once and makes seventeen query calls. Prefix time is summed within each trial before taking a cross-trial median. The cold prefix includes the first query call; the warm prefix excludes it. Values below include construction once and are seconds.

| 4M backend | One cold batch | Sixteen cold batches | One warm batch | Sixteen warm batches |
|---|---:|---:|---:|---:|
| cuda-grid | 0.614 | 8.209 | 0.613 | 8.208 |
| rt-incore | 1.865 | 13.855 | 1.899 | 13.942 |
| rt-ooc | 3.698 | 47.777 | 3.954 | 48.454 |

Lifecycle results describe repeated queries against a fixed retained index. They do not claim a universal amortization break-even point.

## Reproducibility and detailed results

The primary release contains 250 rows from 125 independent processes and 25 configurations. The lifecycle release contains 510 rows from 30 processes and six configurations. Validators recompute summaries and verify source, executable, and raw-result hashes.

See the [primary manifest](c17_20261029_v2/manifest.json) and [lifecycle manifest](c17_lifecycle_20261029/manifest.json) for the measured build identities, source snapshots, and raw data.
