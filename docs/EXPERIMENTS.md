# Experiments

All experiments are scripted in `bench/run_experiments.py` and plotted by
`bench/plot_results.py`; raw JSON lines land in `results/raw/` (git-ignored),
figures in `results/figures/`, and every table in `results/summary.md`.

```powershell
.\.venv\Scripts\python.exe bench\run_experiments.py           # full suite
.\.venv\Scripts\python.exe bench\run_experiments.py --quick   # small sizes
.\.venv\Scripts\python.exe bench\plot_results.py
```

## Setup

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU (Ada, sm_89, 24 SMs, 8 GB, PCIe 4.0 x8), driver 617.42, WDDM |
| CPU / RAM | 32 hardware threads, 64 GB |
| Software | Windows 11, CUDA 13.4, OptiX 9.1 (headers from NVIDIA/optix-dev), MSVC 19.44, OpenMP (`/openmp:llvm`) |
| Data | synthetic, seeded: `uniform` (unit cube), `clustered` (64 Gaussian clusters, log-uniform widths, Zipf weights, 5% noise), `scene` (ground plane + building facades + spheres: 2-manifold data like LiDAR) |
| Queries | `Q = P` unless stated; radius chosen for ~32 expected neighbors at uniform density |

**Timing.** `search_ms` is end-to-end from host queries to host results:
query reordering, routing, all transfers, traversal, and scattering the
results back to the original order. One-time index construction
(`index_ms`: partition, calibration, tile builds; or point sort + GAS build for
in-core) is reported separately. `exec_ms` is the GPU pipeline alone.
Repeated runs report the best run. Every run is verified (`--verify sample`
checks 5–20K random queries against the exact CPU grid; the test suites check
everything with exact equality).

**Device budget.** `--budget-mb` caps the memory the out-of-core search may
use (tile slots + window buffers + ordering scratch), emulating smaller GPUs
on the same hardware. In-core methods always get the whole GPU.

## Questions and experiments

| # | Question (synopsis) | Experiment | Figure |
|---|---|---|---|
| E1 | How do the backends compare on different data? | `baselines` | `baselines_throughput.png` |
| E2 | Can hardware traversal stay effective once the structure exceeds device memory? | `scaling` (5M–160M points) | `scaling.png` |
| E3 | Does the cost model pick a good tile size? (Eq. 1) | `tile_size` (depth 2–10, G = 1 and 8) | `tile_size_model.png` |
| E4 | Does Morton reordering improve tile reuse? | `ablation_order` (4 orders × LRU/Belady, 4 slots) | `ablation_order.png` |
| E5 | Does double buffering hide transfers? | `ablation_overlap` (1–8 slots, overlap on/off) | `ablation_overlap.png` |
| E6 | How does performance degrade with less memory? | `budget` (64 MB – all) | `budget.png` |
| E7 | Point bucketing (our addition) | `group_size` (G = 1..32) | `group_size.png` |
| E8 | Duplicate points or duplicate rays at tile borders? | `halo` (k = 8..512) | `halo_modes.png` |
| E9 | Neighbor lists instead of counts | `lists` (K = 16, 64) | table only |

Results and discussion: see the README ("Results") and `results/summary.md`.
