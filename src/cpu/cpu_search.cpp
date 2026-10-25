#include "cpu/cpu_search.h"

#include <omp.h>

#include <algorithm>
#include <cmath>

#include "common/timer.h"

namespace ooc {
namespace {

constexpr int64_t kMaxCells = int64_t(1) << 25;

struct CpuGrid {
  Aabb box;
  double cell = 0.0;
  int64_t dims[3] = {1, 1, 1};
  std::vector<uint32_t> start;  // numCells + 1
  std::vector<float3> pts;      // points sorted by cell
  std::vector<uint32_t> ids;    // original index of pts[i]

  int64_t coord(float v, int axis) const {
    const float lo = axis == 0 ? box.lo.x : (axis == 1 ? box.lo.y : box.lo.z);
    return int64_t(std::floor((double(v) - double(lo)) / cell));
  }
  int64_t linear(int64_t x, int64_t y, int64_t z) const { return (z * dims[1] + y) * dims[0] + x; }
};

CpuGrid buildGrid(const std::vector<float3>& points, float r) {
  CpuGrid g;
  g.box = computeBounds(points);
  if (!g.box.valid()) g.box.extend(make_f3(0, 0, 0));
  g.cell = std::max(double(conservativeRadius(r, g.box)), 1e-30);
  const float3 e = g.box.extent();
  for (;;) {
    g.dims[0] = int64_t(double(e.x) / g.cell) + 1;
    g.dims[1] = int64_t(double(e.y) / g.cell) + 1;
    g.dims[2] = int64_t(double(e.z) / g.cell) + 1;
    if (g.dims[0] * g.dims[1] * g.dims[2] <= kMaxCells) break;
    g.cell *= 1.25;  // larger cells keep the 27-cell stencil exact, just less selective
  }
  const int64_t numCells = g.dims[0] * g.dims[1] * g.dims[2];
  const int64_t n = int64_t(points.size());
  std::vector<uint32_t> cellOf(static_cast<size_t>(n));
  std::vector<uint32_t> counts(size_t(numCells), 0u);
#pragma omp parallel for
  for (int64_t i = 0; i < n; ++i) {
    const float3 p = points[size_t(i)];
    const int64_t c = g.linear(std::min(g.coord(p.x, 0), g.dims[0] - 1), std::min(g.coord(p.y, 1), g.dims[1] - 1),
                               std::min(g.coord(p.z, 2), g.dims[2] - 1));
    cellOf[size_t(i)] = uint32_t(c);
#pragma omp atomic
    counts[size_t(c)]++;
  }
  g.start.assign(size_t(numCells) + 1, 0u);
  for (int64_t c = 0; c < numCells; ++c) g.start[size_t(c) + 1] = g.start[size_t(c)] + counts[size_t(c)];
  std::vector<uint32_t> cursor(g.start.begin(), g.start.end() - 1);
  g.pts.resize(size_t(n));
  g.ids.resize(size_t(n));
  for (int64_t i = 0; i < n; ++i) {  // serial fill keeps the in-cell order deterministic
    const uint32_t pos = cursor[cellOf[size_t(i)]]++;
    g.pts[pos] = points[size_t(i)];
    g.ids[pos] = uint32_t(i);
  }
  return g;
}

// Visits all points of the 27-cell neighbourhood that pass the exact distance test.
template <class Fn>
void forEachNeighbor(const CpuGrid& g, float3 q, float r2, Fn&& fn) {
  const int64_t cx = g.coord(q.x, 0), cy = g.coord(q.y, 1), cz = g.coord(q.z, 2);
  const int64_t x0 = std::max<int64_t>(cx - 1, 0), x1 = std::min<int64_t>(cx + 1, g.dims[0] - 1);
  const int64_t y0 = std::max<int64_t>(cy - 1, 0), y1 = std::min<int64_t>(cy + 1, g.dims[1] - 1);
  const int64_t z0 = std::max<int64_t>(cz - 1, 0), z1 = std::min<int64_t>(cz + 1, g.dims[2] - 1);
  for (int64_t z = z0; z <= z1; ++z)
    for (int64_t y = y0; y <= y1; ++y) {
      const int64_t row = g.linear(0, y, z);
      const uint32_t b = g.start[size_t(row + x0)], e = g.start[size_t(row + x1 + 1)];
      for (uint32_t j = b; j < e; ++j)
        if (dist2(q, g.pts[j]) <= r2) fn(g.ids[j]);
    }
}

}  // namespace

void cpuBruteForce(const std::vector<float3>& points, const std::vector<float3>& queries, const SearchParams& params,
                   SearchResult& out) {
  out.reset(queries.size(), params.maxNeighbors);
  const float r2 = params.radius * params.radius;
  const int64_t m = int64_t(queries.size());
  const uint32_t K = params.maxNeighbors;
#pragma omp parallel for schedule(dynamic, 64)
  for (int64_t qi = 0; qi < m; ++qi) {
    const float3 q = queries[size_t(qi)];
    uint32_t c = 0;
    for (size_t j = 0; j < points.size(); ++j) {
      if (dist2(q, points[j]) <= r2) {
        if (c < K) out.neighbors[size_t(qi) * K + c] = uint32_t(j);
        ++c;
      }
    }
    out.counts[size_t(qi)] = c;
  }
}

void cpuGridSearch(const std::vector<float3>& points, const std::vector<float3>& queries, const SearchParams& params,
                   SearchResult& out, Metrics* metrics) {
  Timer t;
  const CpuGrid g = buildGrid(points, params.radius);
  const double buildMs = t.ms();
  out.reset(queries.size(), params.maxNeighbors);
  const float r2 = params.radius * params.radius;
  const uint32_t K = params.maxNeighbors;
  const int64_t m = int64_t(queries.size());
  t.reset();
#pragma omp parallel for schedule(dynamic, 1024)
  for (int64_t qi = 0; qi < m; ++qi) {
    uint32_t c = 0;
    uint32_t* dst = K ? out.neighbors.data() + size_t(qi) * K : nullptr;
    forEachNeighbor(g, queries[size_t(qi)], r2, [&](uint32_t id) {
      if (c < K) dst[c] = id;
      ++c;
    });
    out.counts[size_t(qi)] = c;
  }
  if (metrics) {
    metrics->set("build_ms", buildMs);
    metrics->set("query_ms", t.ms());
    metrics->set("grid_cells", double(g.dims[0] * g.dims[1] * g.dims[2]));
    metrics->set("threads", double(omp_get_max_threads()));
  }
}

std::vector<uint32_t> cpuGridCounts(const std::vector<float3>& points, const std::vector<float3>& queries,
                                    const std::vector<size_t>& subset, float radius) {
  const CpuGrid g = buildGrid(points, radius);
  const float r2 = radius * radius;
  std::vector<uint32_t> counts(subset.size());
  const int64_t s = int64_t(subset.size());
#pragma omp parallel for schedule(dynamic, 256)
  for (int64_t i = 0; i < s; ++i) {
    uint32_t c = 0;
    forEachNeighbor(g, queries[subset[size_t(i)]], r2, [&](uint32_t) { ++c; });
    counts[size_t(i)] = c;
  }
  return counts;
}

}  // namespace ooc
