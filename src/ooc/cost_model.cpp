#include "ooc/cost_model.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "common/host_utils.h"
#include "common/morton.h"
#include "cuda/buffers.h"
#include "cuda/kernels.h"
#include "rt/optix_engine.h"

namespace ooc {
namespace {

std::vector<float3> strideSample(const std::vector<float3>& v, size_t maxCount, uint64_t seed) {
  if (v.size() <= maxCount) return v;
  const size_t stride = (v.size() + maxCount - 1) / maxCount;
  std::vector<float3> s;
  s.reserve(maxCount);
  for (size_t i = size_t(seed % stride); i < v.size(); i += stride) s.push_back(v[i]);
  return s;
}

float timeLaunches(OptixEngine& engine, uint64_t dParams, uint32_t width, int reps, cudaStream_t s, cudaEvent_t a,
                   cudaEvent_t b) {
  CUDA_CHECK(cudaEventRecord(a, s));
  for (int r = 0; r < reps; ++r) engine.launch(dParams, width, s);
  CUDA_CHECK(cudaEventRecord(b, s));
  CUDA_CHECK(cudaEventSynchronize(b));
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
  return ms;
}

}  // namespace

double Calibration::tauNs(double n) const {
  if (n <= 1) return tauNs0;
  const double t = (std::log2(n) - std::log2(n0)) / std::max(1e-9, std::log2(n1) - std::log2(n0));
  return std::max(0.25 * std::min(tauNs0, tauNs1), tauNs0 + t * (tauNs1 - tauNs0));
}

Calibration CostModel::calibrate(OptixEngine& engine, const std::vector<float4>& pts, float radius, float R,
                                 uint32_t K, uint32_t group) {
  group = std::max(1u, group);
  Calibration c;
  CudaStream s;
  CudaEvent a(true), b(true);

  {  // pinned host -> device bandwidth
    const size_t bytes = size_t(64) << 20;
    PinnedBuffer h(bytes);
    DeviceBuffer d(bytes);
    std::memset(h.get(), 1, bytes);
    CUDA_CHECK(cudaMemcpyAsync(d.get(), h.get(), bytes, cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaEventRecord(a, s));
    for (int i = 0; i < 4; ++i) CUDA_CHECK(cudaMemcpyAsync(d.get(), h.get(), bytes, cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaEventRecord(b, s));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    c.h2dGBps = 4.0 * double(bytes) / (double(ms) * 1e6);
  }

  const size_t N = pts.size();
  if (N == 0) return c;
  auto padPointsNaNHost = [](float4* p, size_t count) { std::memset(p, 0xff, sizeof(float4) * count); };
  c.n1 = double(std::min<size_t>(size_t(1) << 19, N));
  c.n0 = std::max(1.0, std::floor(std::min(double(1 << 16), c.n1 / 8)));
  const float r2 = radius * radius;
  DeviceBuffer dParams(sizeof(LaunchParams));
  double neighborSum = 0, neighborCount = 0, overheadSum = 0;
  int overheadCount = 0;
  GasScratch scratch;
  DeviceBuffer gas, dPts, dQ, dCounts, dNbrs;
  for (int size = 0; size < 2; ++size) {
    const size_t n = size_t(size == 0 ? c.n0 : c.n1);
    double tauSum = 0;
    int probes = 0;
    for (int k = 0; k < 4; ++k) {  // four regions spread over the kd order
      const size_t off = std::min(N - n, N * size_t(k) / 4);
      // Probe tile prepared like a real one: Morton order, NaN-padded groups.
      const size_t padded = paddedCount(n, group);
      std::vector<float4> chunk(padded);
      std::copy(pts.begin() + off, pts.begin() + off + n, chunk.begin());
      mortonSortInPlace(chunk.data(), n);
      padPointsNaNHost(chunk.data() + n, padded - n);
      dPts.alloc(sizeof(float4) * padded);
      CUDA_CHECK(cudaMemcpyAsync(dPts.get(), chunk.data(), sizeof(float4) * padded, cudaMemcpyHostToDevice, s));
      const GasInfo info = engine.buildGasOwned(dPts.as<float4>(), uint32_t(n), R, gas, scratch, s, group);
      if (size == 1 && k == 0) c.gasBytesPerPrim = double(info.bytes) / double(n);
      // Probe queries: a strided subset of the probe points, already in Morton order.
      const size_t q = std::min<size_t>(n, size_t(1) << 18);
      std::vector<float4> hq(q);
      for (size_t i = 0; i < q; ++i) hq[i] = chunk[i * n / q];
      dQ.alloc(sizeof(float4) * q);
      dCounts.alloc(4 * q);
      if (K) dNbrs.alloc(4ull * K * q);
      CUDA_CHECK(cudaMemcpyAsync(dQ.get(), hq.data(), sizeof(float4) * q, cudaMemcpyHostToDevice, s));
      CUDA_CHECK(cudaMemsetAsync(dCounts.get(), 0, 4 * q, s));
      LaunchParams lp{};
      lp.handle = info.handle;
      lp.points = dPts.as<float4>();
      lp.group = group;
      lp.queries = dQ.as<float4>();
      lp.queryList = nullptr;
      lp.counts = dCounts.as<uint32_t>();
      lp.neighbors = K ? dNbrs.as<uint32_t>() : nullptr;
      lp.numQueries = uint32_t(q);
      lp.maxNeighbors = K;
      lp.radius2 = r2;
      CUDA_CHECK(cudaMemcpyAsync(dParams.get(), &lp, sizeof(lp), cudaMemcpyHostToDevice, s));
      engine.launch(dParams.dptr(), uint32_t(q), s);  // warm-up (clocks, caches); its counts give |N(q)|
      std::vector<uint32_t> counts(q);
      CUDA_CHECK(cudaMemcpyAsync(counts.data(), dCounts.get(), 4 * q, cudaMemcpyDeviceToHost, s));
      CUDA_CHECK(cudaStreamSynchronize(s));
      for (uint32_t v : counts) neighborSum += v;
      neighborCount += double(q);
      engine.launch(dParams.dptr(), uint32_t(q), s);
      const float ms = timeLaunches(engine, dParams.dptr(), uint32_t(q), 3, s, a, b);
      const double tau = double(ms) * 1e6 / (3.0 * double(q));
      tauSum += tau;
      ++probes;
      if (size == 1) {
        // Fixed cost of a launch that cannot fill the GPU (latency + tail),
        // measured as the excess of a small launch over its per-query share.
        const uint32_t qs = uint32_t(std::min<size_t>(q, 4096));
        const float small = timeLaunches(engine, dParams.dptr(), qs, 8, s, a, b);
        overheadSum += std::max(0.0, double(small) * 1e3 / 8.0 - double(qs) * tau * 1e-3);
        ++overheadCount;
      }
    }
    (size == 0 ? c.tauNs0 : c.tauNs1) = tauSum / probes;
  }
  c.probeNeighbors = neighborSum / std::max(1.0, neighborCount);

  {  // fixed per-launch and per-relocation costs (tiny launches on the last probe)
    LaunchParams lp{};
    CUDA_CHECK(cudaMemcpy(&lp, dParams.get(), sizeof(lp), cudaMemcpyDeviceToHost));
    lp.numQueries = 1;
    CUDA_CHECK(cudaMemcpy(dParams.get(), &lp, sizeof(lp), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(a, s));
    for (int r = 0; r < 50; ++r) engine.launch(dParams.dptr(), 1, s);
    CUDA_CHECK(cudaEventRecord(b, s));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    c.launchUs = std::max(double(ms) * 1e3 / 50.0, overheadCount ? overheadSum / overheadCount : 0.0);
    const OptixRelocationInfo info = engine.relocationInfo(lp.handle);
    CUDA_CHECK(cudaEventRecord(a, s));
    for (int r = 0; r < 20; ++r) engine.relocate(info, gas.dptr(), gas.bytes(), s);
    CUDA_CHECK(cudaEventRecord(b, s));
    CUDA_CHECK(cudaEventSynchronize(b));
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    c.relocUs = double(ms) * 1e3 / 20.0;
  }
  return c;
}

CostModel::CostModel(const KdTree& tree, const std::vector<float3>& points, const std::vector<float3>& queries,
                     float R, uint64_t seed)
    : tree_(tree), N_(double(points.size())), M_(double(queries.size())) {
  const auto ps = strideSample(points, size_t(1) << 20, seed);
  const auto qs = strideSample(queries, size_t(1) << 18, seed + 1);
  pointScale_ = ps.empty() ? 0.0 : N_ / double(ps.size());
  queryScale_ = qs.empty() ? 0.0 : M_ / double(qs.size());
  pointCover_ = tree.nodeCoverage(ps, R);
  queryOwner_ = tree.ownerCounts(qs);
  queryCover_ = tree.nodeCoverage(qs, R);
}

DepthEstimate CostModel::estimate(int depth, HaloMode halo, const Calibration& cal, size_t budget, size_t windowBytes,
                                  int slotsOverride, size_t W) const {
  DepthEstimate e;
  e.depth = depth;
  e.halo = halo;
  const auto frontier = tree_.frontier(depth);
  struct T { double bytes, q, pts; };
  std::vector<T> tiles;
  tiles.reserve(frontier.size());
  double storedPts = 0, pairs = 0;
  for (int id : frontier) {
    const KdNode& n = tree_.nodes()[size_t(id)];
    double pts = double(n.count()), q;
    if (halo == HaloMode::Points) {
      pts = std::max(pts, double(pointCover_[size_t(id)]) * pointScale_);
      q = double(queryOwner_[size_t(id)]) * queryScale_;
    } else {
      q = double(queryCover_[size_t(id)]) * queryScale_;
    }
    if (pts <= 0) continue;
    const double bytes = pts * (cal.gasBytesPerPrim + 16.0) + 256.0;
    tiles.push_back({bytes, q, pts});
    storedPts += pts;
    pairs += q;
    e.maxTileBytes = std::max(e.maxTileBytes, bytes);
    e.totalBytes += bytes;
  }
  e.tiles = tiles.size();
  e.dupFactor = halo == HaloMode::Points ? storedPts / std::max(1.0, N_) : pairs / std::max(1.0, M_);
  const double avail = double(budget) - double(windowBytes);
  if (slotsOverride > 0) e.slots = slotsOverride;
  else e.slots = avail > 0 ? int(std::min<double>(double(e.tiles), std::floor(avail / std::max(1.0, e.maxTileBytes)))) : 0;
  e.feasible = e.slots >= 1 && double(e.slots) * e.maxTileBytes <= avail;

  const double bw = cal.h2dGBps * 1e6;  // bytes per ms
  std::vector<double> x(tiles.size()), y(tiles.size());
  size_t hidden = 0;
  for (size_t t = 0; t < tiles.size(); ++t) {
    x[t] = tiles[t].bytes / bw + cal.relocUs * 1e-3;
    const double steps = 1.0 + tiles[t].q / double(std::max<size_t>(1, W));
    y[t] = cal.launchUs * 1e-3 * steps + tiles[t].q * cal.tauNs(tiles[t].pts) * 1e-6;
    e.transferMs += x[t];
    e.traverseMs += y[t];
  }
  double total = 0;
  if (e.slots >= 2 && !tiles.empty()) {
    total = x[0];
    for (size_t t = 0; t < tiles.size(); ++t) {
      const double next = t + 1 < tiles.size() ? x[t + 1] : 0.0;
      total += std::max(y[t], next);
      if (t + 1 < tiles.size() && y[t] >= next) ++hidden;
    }
    e.eq1Fraction = tiles.size() > 1 ? double(hidden) / double(tiles.size() - 1) : 1.0;
  } else {
    total = e.transferMs + e.traverseMs;
  }
  // Query upload, routing lists and counts download (on their own streams, kept conservative).
  total += (16.0 * M_ + 4.0 * pairs + 4.0 * M_) / bw;
  e.predictedMs = total;
  return e;
}

DepthEstimate CostModel::choose(const Calibration& cal, HaloMode halo, bool haloAuto, size_t budget,
                                size_t windowBytes, int slotsOverride, size_t W, std::vector<DepthEstimate>* all) const {
  DepthEstimate best;
  bool have = false;
  auto better = [](const DepthEstimate& a, const DepthEstimate& b) {
    const bool pa = a.slots >= 2, pb = b.slots >= 2;  // prefer configurations that can double buffer
    if (pa != pb) return pa;
    return a.predictedMs < b.predictedMs;
  };
  const HaloMode modes[2] = {HaloMode::Points, HaloMode::Queries};
  for (HaloMode h : modes) {
    if (!haloAuto && h != halo) continue;
    for (int d = 0; d <= tree_.maxDepth(); ++d) {
      const DepthEstimate e = estimate(d, h, cal, budget, windowBytes, slotsOverride, W);
      if (all) all->push_back(e);
      if (!e.feasible) continue;
      if (!have || better(e, best)) { best = e; have = true; }
    }
  }
  if (!have) throw std::runtime_error("device budget too small for any tiling (even the smallest tile does not fit)");
  return best;
}

}  // namespace ooc
