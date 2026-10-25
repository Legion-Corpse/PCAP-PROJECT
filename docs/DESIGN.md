# Design

This document explains how the out-of-core RT-core fixed-radius neighbor search
is built, why each piece exists, and where it lives in the code.

## 1. Problem

Points `P = {p_1..p_N} ⊂ R^3`, queries `Q = {q_1..q_M} ⊂ R^3`, radius `r`.
For every query, report `N(q) = { p ∈ P : ||p − q|| ≤ r }` — as an exact count,
and optionally the first `K` ids. The acceleration structure for `P` does not
fit in device memory (or is not allowed to: `--budget-mb` emulates a smaller GPU).

## 2. RT-core formulation (in-core building block)

`src/rt/device_programs.cu`, `src/rt/optix_engine.cpp`

* Every point is an OptiX **custom primitive** whose AABB is the cube of
  half-width `R` around it. A query is a ray starting at `q` with
  `tmax = 1e-16`: BVH traversal on the RT core reports exactly the boxes that
  contain `q`.
* The **intersection program** performs the exact `||p − q||² ≤ r²` test and
  never calls `optixReportIntersection`, so traversal continues and visits
  every candidate. Any-hit and closest-hit are disabled
  (`OPTIX_RAY_FLAG_DISABLE_ANYHIT | DISABLE_CLOSESTHIT`,
  `OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL` so a primitive is never
  visited twice).
* The running count lives in a payload register and **continues across
  tiles**: the ray-gen program loads `counts[q]` before the trace and stores it
  after, so the same program serves one tile or many.
* GAS build flags: `PREFER_FAST_TRACE | ALLOW_COMPACTION`; every GAS is compacted.

### Point bucketing (improvement over RTNN)

With `--group G`, a primitive is a run of `G` Morton-adjacent points and its
AABB is their bounding box grown by `R`; the intersection program tests all `G`
points. Each RT-core → SM hand-off (the expensive part of custom-primitive
traversal) then does `G` cheap distance tests instead of one. Effects measured
in `results/summary.md`:

* GAS size drops from ~19 B/point (G = 1) to ~1.3 B/point (G = 16), so each
  tile streams ~2× fewer bytes (points dominate: 16 B/point);
* BVH build time drops ~8×;
* traversal is up to ~2.5× faster on clustered data (many points per box),
  roughly neutral for G ≤ 4 on uniform data, slower for large G there.

Partial groups are padded with NaN points (`dist2` with NaN is never `≤ r²`).

## 3. Exactness

All backends must return *identical* counts — the tests compare them with
`==`, including a lattice where many pairs sit at distance exactly `r`.

* **One distance formula everywhere.** `dist2()` (`src/common/vec_math.h`)
  uses `__fsub_rn/__fmul_rn/__fadd_rn` on the device. nvcc would otherwise
  contract `a*b+c` into an FMA while MSVC (`/fp:precise`) does not, and a pair
  at distance ≈ r could be a neighbor on one backend and not on another.
* **Conservative candidate filters.** Grid cells, AABB half-widths and halo
  distances use `R = r(1 + 1e-5) + 4·ε·max|coord|` (`conservativeRadius`), so
  every pair accepted by the float test is guaranteed to be a candidate even
  after rounding of box bounds, cell coordinates, or the RT core's box test.
* Cell coordinates are computed in double.

## 4. Out-of-core pipeline

```
points ──► kd-tree (median splits, built once to 16K-point leaves)
             │  cost model picks the cut depth d   (src/ooc/cost_model.*)
             ▼
          tiles = frontier at depth d   ──► per tile: owned (+ halo) points,
             │                               Morton sort, NaN pad, GAS build,
             │                               compaction ──► pinned host blob
             ▼                                               (src/ooc/tile_store.*)
queries ──► plan: order, windows, routing, cache policy       (src/ooc/planner.*, gpu_order.cu)
             ▼
          pipelined execution on 3 streams                     (src/ooc/executor.*)
```

### 4.1 Spatial tiling (`src/ooc/kdtree.*`)

A median-split kd-tree is built once over a packed copy of the points; every
subtree's points are contiguous. A **tiling is a cut of that tree**: all
nodes at depth `d` (plus shallower leaves). The cost model can therefore
compare tile sizes without re-partitioning. Cells have infinite outer faces,
so every query has exactly one owner tile. Large nodes (> 2^20 points) are
split with all threads (sampled median + stable parallel partition); smaller
nodes are split in parallel with each other. Degenerate inputs (ties on the
split plane, identical points) fall back to "split just above the minimum" or
an oversized leaf.

### 4.2 Halos and the ownership rule (`src/ooc/tile_store.*`)

The synopsis overlaps tiles by `r` and removes duplicates afterwards. We
avoid producing duplicates in the first place with an **ownership rule**, and
offer two dual strategies:

| `--halo` | tile stores | a query is routed to | duplication |
|---|---|---|---|
| `points` (default) | owned points + every point within `R` of the cell | the one tile whose cell contains it | points (bytes streamed) |
| `queries` | owned points only (tiles are disjoint) | every tile whose cell is within `R` | rays (traversals) |

Either way each (query, point) pair is evaluated in exactly one tile, so the
partial counts simply add up and no de-duplication pass exists. Halo points
are found by a kd descent per point with per-axis pruning plus an exact
point-to-box distance test; `--halo auto` lets the cost model choose.

### 4.3 Host-resident compact BVHs

Each tile's GAS is built on the device **one tile at a time**, compacted, and
copied with its points into one contiguous pinned blob
`[compacted GAS | pad | float4 points]` inside a pinned bump arena (one
`cudaHostAlloc` per 256 MB instead of one per tile). Host preparation of tile
`t+1` (gather, Morton sort, padding) overlaps the device build of tile `t`.
Loading a tile is then a single DMA plus `optixAccelRelocate`, which fixes
the GAS's internal pointers for the slot it landed in
(`optixAccelGetRelocationInfo` is recorded at build time).

### 4.4 Query ordering (`src/ooc/planner.cpp`, `src/ooc/gpu_order.cu`)

All queries are known up front, and reordering never changes the result.

* `tiled` (default): queries grouped by owner tile in kd DFS order (which is
  itself a space-filling order) and **Morton-sorted inside each tile**. Done
  on the GPU in budget-sized chunks: one kernel does the kd descent and the
  Morton key, CUB sorts by `(tile, morton)`, and only the 4-byte permutation
  comes back; the host concatenates the per-tile runs of all chunks. Windows
  then cover contiguous runs of tiles, every tile is streamed exactly once,
  the per-tile query lists are contiguous slices (no index lists are uploaded
  at all), and each window's coordinates are gathered by the host *while the
  GPU works on the previous window*.
* `morton`: one global Morton (Z-order) sort — the synopsis' scheme.
* `random`, `original`: ablation baselines.

### 4.5 Windows and routing

The ordered queries are cut into windows (bounded device memory for queries,
counts and K-neighbor rows; double buffered). Each window is routed with a
stable parallel counting sort into per-tile lists. Tiles likely still
resident (the last `slots` distinct tiles used) are visited first.

### 4.6 Tile cache and eviction

The device holds `S` tile slots (as many as the budget allows). Because the
whole access sequence is known before execution, the planner simulates the
cache offline and can use **Belady's MIN** (evict the tile whose next use is
farthest in the future) instead of LRU.

Fewest loads is not the same as fastest, though. Slots used by the last
`L = min(2, S−1)` steps are never victims: `L ≥ 1` lets the next transfer
overlap the current traversal, and `L = 2` keeps a second transfer in flight.
Plain MIN tends to evict the slot used two steps earlier (when windows sweep
the tiles in order, that tile is needed again soonest *and* is the most
recently freed), which caps prefetching at one tile; with 4 slots and
transfer-bound windows it was ~13% slower than LRU at equal load counts. The
constrained version keeps MIN's lower load count (e.g. 330 vs 337 loads for
global Morton order) without that stall.

### 4.7 Pipelined execution (double buffering)

```
copy stream     │ win q+params │ tile A │ tile B │        │ tile C │ ...
compute stream  │              │ reloc A, launch A │ reloc B, launch B │ ...
d2h stream      │                                        │ counts │
```

* **copy**: DMA only — window queries + all launch parameters of the window
  (one copy), tile blobs.
* **compute**: for a freshly loaded tile, `optixAccelRelocate` and a 1-thread
  kernel that writes the new traversable handle into a device slot table
  (`LaunchParams::handleSlot`); then `optixLaunch`.
* **d2h**: window results.

Two hazards were found by measurement and removed: (1) relocation is a kernel;
on the copy stream it waited for the SMs held by the running traversal and
serialized the pipeline; (2) a small per-launch parameter `memcpy` on the
compute stream queued behind multi-MB tile copies on the same copy engine.
Events `slotFree[s]` (last launch that read slot `s`) and `tileReady[s]`
(transfer done) order everything; window buffers are double buffered, so the
host packs window `w+1` and scatters window `w−1` while the GPU runs `w`.

## 5. Cost model (`src/ooc/cost_model.*`)

For tile `t` with compacted size `S_t` and `|Q_t|` queries, double buffering
hides the next transfer when (synopsis Eq. 1)

```
|Q_t| · τ(n_t)  ≥  S_t / B_h2d
```

and the pipelined time is estimated as

```
T(d) = x_0 + Σ_t max(y_t, x_{t+1})
x_t  = S_t / B_h2d + c_reloc
y_t  = c_launch · steps_t + |Q_t| · τ(n_t)
```

* `B_h2d`: pinned-memory bandwidth, measured (64 MB copies).
* `τ(n)`: traversal time per query, measured on probe tiles of 2^16 and 2^19
  points taken from four regions of the kd order (Morton-ordered, with the
  real group size and K), interpolated in `log2 n`.
* `c_launch`: per-launch latency **and tail**, measured as the excess of a
  4096-query launch over its per-query share — small tiles under-fill the GPU,
  which is what makes very fine tilings slow.
* `S_t` from the measured GAS bytes per point and per-tile point counts; per
  tile halo sizes and query loads are estimated from 1M point / 256K query
  samples pushed through the tree (`nodeCoverage`, `ownerCounts`).
* Budget: `slots = ⌊(budget − window buffers) / max S_t⌋`, requiring ≥ 2 to
  double buffer.

The model evaluates every depth (and both halo modes with `--halo auto`) and
picks the fastest feasible one. Over a 4–1024-tile sweep its absolute error is
within −13% … +10%, and its chosen tile count runs within 5% of the best one
measured (`results/figures/tile_size_model.png`). In the budget sweep, it
predicted 677 ms for a measured 680 ms at 64 MB.

## 6. Baselines

* `cpu-grid` (`src/cpu/cpu_search.cpp`): OpenMP uniform grid, cell ≥ R, 27-cell
  stencil, deterministic counting sort. Also the correctness oracle.
* `brute`: O(NM), tests only.
* `cuda-grid` (`src/cuda/grid_search.cu`): in-core GPU cell list (CUB radix sort
  by cell, prefix-summed cell starts, 9 contiguous row ranges per query,
  Morton-sorted queries).
* `rt-incore` (`src/rt/incore_search.cpp`): one GAS over all points
  (RTNN-style with `--group 1`), Morton-sorted points and queries on the GPU.

## 7. Engineering notes

* OptiX 9.1 headers come from github.com/NVIDIA/optix-dev at configure time;
  the device programs are compiled to OptiX-IR and embedded in the binary.
* MSVC 19.44 overflows its stack compiling some `#pragma omp` loops with
  heavily inlined bodies; those loops go through a type-erased
  `parallelFor(n, grain, body)` helper (`src/common/host_utils.h`).
* Windows first-touch page faults cost 5–12× the bandwidth of warm pages; hot
  scratch (radix-sort buffers) is kept per thread.
* `CUDA_MODULE_LOADING=EAGER` is set by the CLI so first kernel calls are not
  charged with lazy module loading.
