#include "ooc/tile_store.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <future>

#include "common/host_utils.h"
#include "common/timer.h"
#include "cuda/kernels.h"
#include "rt/optix_engine.h"

namespace ooc {
namespace {

const float4 kNanPoint = [] {
  float4 v;
  std::memset(&v, 0xff, sizeof(v));
  return v;
}();

// halo[t] = indices (into sortedPoints) of non-owned points within R of tile t's cell.
void computeHalos(const KdTree& tree, const std::vector<float4>& pts, const TileSet& ts, std::vector<uint32_t>& offsets,
                  std::vector<uint32_t>& halo) {
  const size_t T = ts.tiles.size();
  const int64_t n = int64_t(pts.size());
  std::vector<uint32_t> ownerOf(pts.size());
#pragma omp parallel for schedule(dynamic, 1)
  for (int64_t t = 0; t < int64_t(T); ++t) {
    const TileHost& tile = ts.tiles[size_t(t)];
    std::fill(ownerOf.begin() + tile.ownedBegin, ownerOf.begin() + tile.ownedBegin + tile.ownedCount, uint32_t(t));
  }
  std::vector<std::atomic<uint32_t>> counts(T);
  for (auto& c : counts) c.store(0, std::memory_order_relaxed);
#pragma omp parallel for schedule(dynamic, 4096)
  for (int64_t i = 0; i < n; ++i) {
    const uint32_t own = ownerOf[size_t(i)];
    tree.within(xyz(pts[size_t(i)]), ts.R, ts.tileOfNode, [&](int t) {
      if (uint32_t(t) != own) counts[size_t(t)].fetch_add(1, std::memory_order_relaxed);
    });
  }
  offsets.assign(T + 1, 0);
  for (size_t t = 0; t < T; ++t) offsets[t + 1] = offsets[t] + counts[t].load();
  halo.resize(offsets[T]);
  std::vector<std::atomic<uint32_t>> cursor(T);
  for (size_t t = 0; t < T; ++t) cursor[t].store(offsets[t], std::memory_order_relaxed);
#pragma omp parallel for schedule(dynamic, 4096)
  for (int64_t i = 0; i < n; ++i) {
    const uint32_t own = ownerOf[size_t(i)];
    tree.within(xyz(pts[size_t(i)]), ts.R, ts.tileOfNode, [&](int t) {
      if (uint32_t(t) != own) halo[cursor[size_t(t)].fetch_add(1, std::memory_order_relaxed)] = uint32_t(i);
    });
  }
#pragma omp parallel for schedule(dynamic, 1)
  for (int64_t t = 0; t < int64_t(T); ++t)  // deterministic layout regardless of thread timing
    std::sort(halo.begin() + offsets[size_t(t)], halo.begin() + offsets[size_t(t) + 1]);
}

}  // namespace

char* PinnedArena::alloc(size_t bytes) {
  bytes = alignUp(bytes, 256);
  if (chunks_.empty() || used_ + bytes > chunks_.back().bytes()) {
    chunks_.emplace_back(std::max(chunkBytes_, bytes));
    used_ = 0;
  }
  char* p = chunks_.back().as<char>() + used_;
  used_ += bytes;
  return p;
}

size_t PinnedArena::reservedBytes() const {
  size_t total = 0;
  for (const auto& c : chunks_) total += c.bytes();
  return total;
}

TileSet buildTiles(OptixEngine& engine, const KdTree& tree, const std::vector<float4>& pts, int depth, HaloMode mode,
                   float R, uint32_t group, Metrics& m) {
  NvtxRange range("build_tiles");
  TileSet ts;
  ts.halo = mode;
  ts.depth = depth;
  ts.R = R;
  ts.group = std::max(1u, group);
  ts.frontier = tree.frontier(depth);
  ts.tileOfNode = tree.tileMap(ts.frontier);
  ts.tiles.resize(ts.frontier.size());
  for (size_t t = 0; t < ts.frontier.size(); ++t) {
    const KdNode& n = tree.nodes()[size_t(ts.frontier[t])];
    ts.tiles[t].node = ts.frontier[t];
    ts.tiles[t].ownedBegin = n.begin;
    ts.tiles[t].ownedCount = n.count();
  }

  Timer timer;
  std::vector<uint32_t> haloOffsets(ts.tiles.size() + 1, 0), halo;
  if (mode == HaloMode::Points) computeHalos(tree, pts, ts, haloOffsets, halo);
  m.set("halo_ms", timer.ms());
  ts.haloPoints = halo.size();

  for (size_t t = 0; t < ts.tiles.size(); ++t) {
    auto& tile = ts.tiles[t];
    tile.numPoints = tile.ownedCount + (haloOffsets[t + 1] - haloOffsets[t]);
    tile.paddedPoints = uint32_t(paddedCount(tile.numPoints, ts.group));
    ts.maxTilePoints = std::max(ts.maxTilePoints, tile.paddedPoints);
  }

  // Two-stage pipeline: a worker thread assembles tile t+1 (gather owned +
  // halo points, Morton sort, NaN padding) while this thread builds tile t on
  // the device and stores the compacted GAS in the pinned arena.
  timer.reset();
  CudaStream stream;
  GasScratch scratch;
  const uint32_t maxPts = std::max<uint32_t>(ts.maxTilePoints, 1);
  DeviceBuffer dPoints(sizeof(float4) * maxPts);
  DeviceBuffer dCompact(engine.gasOutputBound(maxPts));
  PinnedBuffer staging[2] = {PinnedBuffer(sizeof(float4) * maxPts), PinnedBuffer(sizeof(float4) * maxPts)};
  double prepMs = 0;
  auto prepare = [&](size_t t) {
    Timer pt;
    const TileHost& tile = ts.tiles[t];
    float4* host = staging[t % 2].as<float4>();
    std::memcpy(host, pts.data() + tile.ownedBegin, sizeof(float4) * tile.ownedCount);
    const uint32_t h0 = haloOffsets[t];
    parallelFor(int64_t(haloOffsets[t + 1] - h0), 1 << 15, [&](int64_t b0, int64_t e0) {
      for (int64_t j = b0; j < e0; ++j) host[tile.ownedCount + size_t(j)] = pts[halo[h0 + size_t(j)]];
    });
    // Morton order inside the tile makes each run of `group` points compact.
    if (ts.group > 1) mortonSortInPlace(host, tile.numPoints);
    for (uint32_t j = tile.numPoints; j < tile.paddedPoints; ++j) host[j] = kNanPoint;
    prepMs += pt.ms();
  };
  double gpuMs = 0;
  std::future<void> pending;
  if (!ts.tiles.empty()) pending = std::async(std::launch::async, prepare, size_t(0));
  for (size_t t = 0; t < ts.tiles.size(); ++t) {
    pending.get();
    if (t + 1 < ts.tiles.size()) pending = std::async(std::launch::async, prepare, t + 1);
    Timer gt;
    auto& tile = ts.tiles[t];
    const float4* host = staging[t % 2].as<float4>();
    CUDA_CHECK(cudaMemcpyAsync(dPoints.get(), host, sizeof(float4) * tile.paddedPoints, cudaMemcpyHostToDevice, stream));
    const GasInfo gas = engine.buildGas(dPoints.as<float4>(), tile.numPoints, R, dCompact.dptr(), dCompact.bytes(),
                                        scratch, stream, ts.group);
    tile.gasBytes = gas.bytes;
    tile.pointsOffset = alignUp(gas.bytes, 256);
    tile.blobBytes = tile.pointsOffset + sizeof(float4) * tile.paddedPoints;
    char* blob = ts.arena.alloc(tile.blobBytes);
    CUDA_CHECK(cudaMemcpyAsync(blob, dCompact.get(), gas.bytes, cudaMemcpyDeviceToHost, stream));
    std::memcpy(blob + tile.pointsOffset, host, sizeof(float4) * tile.paddedPoints);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    tile.blob = blob;
    tile.reloc = engine.relocationInfo(gas.handle);
    ts.maxBlobBytes = std::max(ts.maxBlobBytes, tile.blobBytes);
    ts.totalBlobBytes += tile.blobBytes;
    ts.totalGasBytes += tile.gasBytes;
    gpuMs += gt.ms();
  }
  m.set("tile_prep_ms", prepMs);
  m.set("tile_gpu_ms", gpuMs);
  m.set("tile_build_ms", timer.ms());
  m.set("tiles", double(ts.tiles.size()));
  m.set("tile_depth", double(depth));
  m.set("tile_max_points", double(ts.maxTilePoints));
  m.set("tile_avg_points", ts.tiles.empty() ? 0.0 : double(pts.size() + ts.haloPoints) / double(ts.tiles.size()));
  m.set("halo_points", double(ts.haloPoints));
  m.set("halo_fraction", pts.empty() ? 0.0 : double(ts.haloPoints) / double(pts.size()));
  m.set("tile_store_bytes", double(ts.totalBlobBytes));
  m.set("gas_bytes_total", double(ts.totalGasBytes));
  m.set("gas_bytes_per_point", double(ts.totalGasBytes) / double(std::max<uint64_t>(1, pts.size() + ts.haloPoints)));
  m.set("points_per_prim", double(ts.group));
  return ts;
}

}  // namespace ooc
