# Out-of-core RT-core fixed-radius neighbor search

This is our Parallel Computing and Architecture Programming project (Group 15). For every query point, the program counts the points that lie within a radius r and, if asked, keeps a bounded list of their original IDs. The host code is written in C and the GPU parts are CUDA kernels and OptiX device programs. We compare four backends that all follow the same output contract:

| Backend | What it does |
|---|---|
| `cpu-grid` | Serial uniform grid on the CPU (our reference) |
| `cuda-grid` | Linked-cell grid built and queried with CUDA kernels |
| `rt-incore` | One compacted OptiX acceleration structure kept resident on the GPU |
| `rt-ooc` | Morton-ordered tiles kept in pinned host memory and streamed through one or two device slots under a memory budget |

## Building

We tested the build on Windows 11 with an RTX 4060 Laptop GPU, CUDA 13.4, MSVC 19.44 and the OptiX 9.1 headers (CMake fetches the headers, so no OptiX SDK install is needed). You need the Visual Studio C++ build tools, the CUDA Toolkit, CMake and Ninja. The default GPU architecture is 89; pass `-CudaArch` to target a different one.

```powershell
./scripts/build.ps1 -Test
```

This builds `build_c17/ooc_rtnn.exe` (the main program), `build_c17/radius_demo.exe` (a small demo) and the C unit tests, and then runs the tests. Both executables link the same search library.

## Small demonstration

```powershell
./scripts/demo.ps1
./scripts/demo.ps1 -Method rt-ooc -MaxIds 1
./scripts/demo.ps1 -Method rt-ooc -TilePoints 1
```

The first command runs all four backends on the six points and three queries in [examples](examples/), with radius one. Every backend reports the counts 4, 2 and 0, with the neighbor lists `[0, 1, 2, 4]`, `[1, 3]` and `[]`, and each result is checked against brute force. The second command keeps only one ID per query, so you can see that the list is truncated while the counts stay complete. The third one splits the points into single-point tiles; the tile printout shows which points each tile owns and which tiles each query is routed to, and the answers do not change. Add `-Timing` to see build and search times (on six points these mostly measure GPU setup).

Two larger demonstrations use one million points:

```powershell
./scripts/budget.ps1
./scripts/compare.ps1
```

`budget.ps1` first shows a tile that is too large for an 8 MiB executor budget being rejected, and then runs the same search under 8 MiB and 2 MiB budgets (the totals are identical, but the smaller budget needs more tile reloads). `compare.ps1` runs the CUDA grid and both RT backends on the same input and prints the retained-search time and explicit device bytes of each. `present.ps1` runs all of the above in order.

## Running your own searches

```powershell
./build_c17/ooc_rtnn.exe --points uniform:100K --method cpu-grid,cuda-grid,rt-incore,rt-ooc --verify full
./build_c17/ooc_rtnn.exe --points scene:200K --radius 0.015 --method rt-ooc --tile-points 64K --budget-mb 64 --group 8 --max-neighbors 16 --repeat 3 --verify full
./build_c17/ooc_rtnn.exe --help
```

Synthetic inputs need no files (`uniform:100K`, `clustered:100K:5`, `scene:100K`). For stored data, use `--points file:PATH` and optionally `--queries file:PATH`; without `--queries` the point set is queried against itself. The loader reads these formats:

| Extension | Layout |
|---|---|
| `.bin` | little-endian float32 x, y, z (12 bytes per point) |
| `.kitti` | little-endian float32 x, y, z, intensity (16 bytes; intensity is ignored) |
| `.ply` | ASCII or little-endian binary PLY vertices |
| `.xyz`, `.txt`, `.csv`, `.pts` | one coordinate triple per line, separated by spaces, commas or semicolons; lines starting with `#` are skipped |

Point IDs follow the input row order, starting at zero. Non-finite coordinates, empty files, truncated binary records and unsupported PLY layouts are rejected with an error message and a nonzero exit code.

Some options worth knowing about:

- `--budget-mb` caps the streamed executor's explicit device allocations (slots, query windows, counts, ID lists and routing arrays). The CUDA context, OptiX pipeline, construction, calibration and host arrays are not part of this cap.
- `--tile-points 0` lets the planner pick a tile capacity from a short calibration run; any other value skips calibration.
- `--slots 1|2` and `--overlap on|off` control the transfer/compute pipeline.
- `--max-neighbors K` keeps at most K IDs per query; the count is always complete.
- `--json PATH` appends one measurement row per run.

## How the streamed backend works

Points are sorted once by Morton code and cut into contiguous chunks, so each point belongs to exactly one tile. For each tile we build a compacted OptiX acceleration structure over buckets of G points and store it, together with the tile's points, in a pinned host blob. At query time the queries are also Morton-sorted and processed in windows. A window is routed to every tile whose radius-expanded bounding box it may touch, the tile is copied into a free device slot and relocated, and the OptiX programs count the bucket members that pass the squared-distance test. Since a point lives in only one tile, a query can visit several tiles without any duplicate removal. CUDA events keep a slot (and its pinned staging area) from being overwritten while an earlier copy or launch still uses it. Finally, results are scattered back to the caller's query order.

The distance test uses single-precision operations in a fixed order on both host and device (contraction is disabled on the host, and the device uses explicitly rounded intrinsics), so all backends agree exactly, including at the radius boundary.

## Tests and measurements

The Python tests need the project virtual environment:

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements-lock.txt
./scripts/verify.ps1
```

`verify.ps1` builds the project, runs the C tests and the pytest suite, validates both stored measurement releases and runs the demo on all four backends.

The measurements used in our report are in `results/`. `results/c17_20261029_v2/` holds the main comparison (250 rows from 125 independent processes) and `results/c17_lifecycle_20261029/` holds the repeated-batch runs on retained indexes. Each folder keeps the exact commands, raw rows, a source snapshot and a manifest with hashes, which `bench/c17_evidence.py --validate` and `bench/c17_lifecycle.py --validate` check. [results/summary.md](results/summary.md) is generated from them by `bench/write_c17_summary.py`. To record a new release, build first, run `bench/c17_record_build.py` from the compiler environment, and give the benchmark driver a new output directory. `bench/prepare_armadillo.py` downloads and converts the Stanford Armadillo scan (into `data/`, which Git ignores).

In short, the CUDA grid had the shortest retained-search time in every workload we measured, and the streamed RT backend kept its explicit device working set small at the price of routing and transfer costs. All numbers come from one laptop GPU, and the real scan we used fits in device memory, so we would not generalise them beyond that setting.

## Layout

| Path | Contents |
|---|---|
| `src_c/` | C host code, CUDA kernels and OptiX device programs |
| `examples/` | the small demo program and its point/query files |
| `scripts/` | build, verification and demo scripts |
| `tests/` | C unit tests and pytest checks for the CLI, demo and measurement validators |
| `bench/` | benchmark drivers, build recording, dataset preparation and summary generation |
| `results/` | the measurement releases used in the report |
| `cmake/` | embeds the compiled OptiX module into the library |
