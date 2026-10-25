#include "cuda/grid_search.h"

#include <cub/device/device_scan.cuh>

#include <algorithm>
#include <cmath>

#include "common/check.h"
#include "common/host_utils.h"
#include "common/timer.h"
#include "cuda/buffers.h"
#include "cuda/kernels.h"

namespace ooc {
namespace {

constexpr int kBlock = 256;
constexpr int64_t kMaxCells = int64_t(1) << 26;

struct GridDesc {
  float3 lo;
  double cell;
  int dims[3];
};

__device__ __forceinline__ int cellCoord(float v, float lo, double cell) {
  return int(floor((double(v) - double(lo)) / cell));
}

__global__ void cellKeysKernel(const float4* __restrict__ pts, uint32_t* __restrict__ keys, uint32_t* __restrict__ vals,
                               uint32_t* __restrict__ counts, uint32_t n, GridDesc g) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float4 p = pts[i];
  const int x = min(cellCoord(p.x, g.lo.x, g.cell), g.dims[0] - 1);
  const int y = min(cellCoord(p.y, g.lo.y, g.cell), g.dims[1] - 1);
  const int z = min(cellCoord(p.z, g.lo.z, g.cell), g.dims[2] - 1);
  const uint32_t c = uint32_t((size_t(z) * g.dims[1] + y) * g.dims[0] + x);
  keys[i] = c;
  vals[i] = i;
  atomicAdd(&counts[c], 1u);
}

__global__ void gridQueryKernel(const float4* __restrict__ pts, const uint32_t* __restrict__ cellStart,
                                const float4* __restrict__ queries, uint32_t n, GridDesc g, float r2, uint32_t K,
                                uint32_t* __restrict__ counts, uint32_t* __restrict__ nbrs) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float3 q = xyz(queries[i]);
  const int cx = cellCoord(q.x, g.lo.x, g.cell), cy = cellCoord(q.y, g.lo.y, g.cell), cz = cellCoord(q.z, g.lo.z, g.cell);
  const int x0 = max(cx - 1, 0), x1 = min(cx + 1, g.dims[0] - 1);
  const int y0 = max(cy - 1, 0), y1 = min(cy + 1, g.dims[1] - 1);
  const int z0 = max(cz - 1, 0), z1 = min(cz + 1, g.dims[2] - 1);
  uint32_t c = 0;
  uint32_t* dst = K ? nbrs + size_t(i) * K : nullptr;
  if (x0 <= x1) {
    for (int z = z0; z <= z1; ++z)
      for (int y = y0; y <= y1; ++y) {
        const size_t row = (size_t(z) * g.dims[1] + y) * g.dims[0];
        const uint32_t b = cellStart[row + x0], e = cellStart[row + x1 + 1];
        for (uint32_t j = b; j < e; ++j) {
          const float4 p = pts[j];
          if (dist2(q, xyz(p)) <= r2) {
            if (c < K) dst[c] = __float_as_uint(p.w);
            ++c;
          }
        }
      }
  }
  counts[i] = c;
}

}  // namespace

void cudaGridSearch(const std::vector<float3>& points, const std::vector<float3>& queries, const SearchParams& params,
                    SearchResult& out, Metrics& m, size_t windowQueries) {
  NvtxRange range("cuda_grid");
  const uint32_t n = uint32_t(points.size()), nq = uint32_t(queries.size());
  const uint32_t K = params.maxNeighbors;
  out.reset(nq, K);
  CudaStream stream;
  CudaEvent e0(true), e1(true), e2(true), e3(true);

  // ---- grid geometry (same rules as the CPU reference) ----
  Aabb box = computeBounds(points);
  if (!box.valid()) box.extend(make_f3(0, 0, 0));
  GridDesc g{box.lo, std::max(double(conservativeRadius(params.radius, box)), 1e-30), {1, 1, 1}};
  const float3 e = box.extent();
  for (;;) {
    g.dims[0] = int(double(e.x) / g.cell) + 1;
    g.dims[1] = int(double(e.y) / g.cell) + 1;
    g.dims[2] = int(double(e.z) / g.cell) + 1;
    if (int64_t(g.dims[0]) * g.dims[1] * g.dims[2] <= kMaxCells) break;
    g.cell *= 1.25;
  }
  const uint32_t numCells = uint32_t(int64_t(g.dims[0]) * g.dims[1] * g.dims[2]);
  int cellBits = 1;
  while ((uint64_t(1) << cellBits) < numCells) ++cellBits;

  // ---- upload + build ----
  Timer wall;
  std::vector<float4> packed(n);
  packPoints(points.data(), nullptr, n, packed.data());
  DeviceBuffer dPts(sizeof(float4) * n), dSorted(sizeof(float4) * n);
  DeviceBuffer dKeys(4ull * n), dVals(4ull * n), dKeysAlt(4ull * n), dValsAlt(4ull * n);
  DeviceBuffer dCellStart(4ull * (numCells + 1));
  CUDA_CHECK(cudaEventRecord(e0, stream));
  CUDA_CHECK(cudaMemcpyAsync(dPts.get(), packed.data(), sizeof(float4) * n, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaEventRecord(e1, stream));
  {
    DeviceBuffer dCounts(4ull * (numCells + 1));
    CUDA_CHECK(cudaMemsetAsync(dCounts.get(), 0, 4ull * (numCells + 1), stream));
    if (n) {
      cellKeysKernel<<<(n + kBlock - 1) / kBlock, kBlock, 0, stream>>>(dPts.as<float4>(), dKeys.as<uint32_t>(),
                                                                       dVals.as<uint32_t>(), dCounts.as<uint32_t>(), n, g);
      CUDA_CHECK_LAST();
    }
    size_t tempBytes = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, tempBytes, dCounts.as<uint32_t>(), dCellStart.as<uint32_t>(),
                                             numCells + 1, stream));
    DeviceBuffer temp(tempBytes);
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(temp.get(), tempBytes, dCounts.as<uint32_t>(), dCellStart.as<uint32_t>(),
                                             numCells + 1, stream));
    radixSortPairs32(dKeys.as<uint32_t>(), dVals.as<uint32_t>(), dKeysAlt.as<uint32_t>(), dValsAlt.as<uint32_t>(), n,
                     cellBits, stream);
    launchGatherFloat4(dPts.as<float4>(), dVals.as<uint32_t>(), dSorted.as<float4>(), n, stream);
    CUDA_CHECK(cudaEventRecord(e2, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  dPts.release(); dKeys.release(); dVals.release(); dKeysAlt.release(); dValsAlt.release();
  float h2dMs = 0, buildMs = 0;
  CUDA_CHECK(cudaEventElapsedTime(&h2dMs, e0, e1));
  CUDA_CHECK(cudaEventElapsedTime(&buildMs, e1, e2));

  // ---- queries: Morton order for warp coherence ----
  Timer searchT, sortT;
  DeviceBuffer dQSorted;
  std::vector<uint32_t> order;
  mortonSortOnDevice(queries, dQSorted, order, stream);
  const double sortMs = sortT.ms();

  // ---- search in windows so K-neighbor output never has to fit at once ----
  size_t freeB = 0, totalB = 0;
  CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));
  const size_t perQuery = 4ull * (1 + K);
  size_t window = windowQueries ? windowQueries : std::max<size_t>(1 << 16, (freeB / 2) / perQuery);
  window = std::min<size_t>(window, std::max<uint32_t>(nq, 1));
  DeviceBuffer dCounts(4ull * window), dNbrs(K ? 4ull * K * window : 0);
  PinnedBuffer hCounts(4ull * window), hNbrs(K ? 4ull * K * window : 0);
  const float r2 = params.radius * params.radius;
  double queryMs = 0, d2hMs = 0;
  for (size_t w0 = 0; w0 < nq; w0 += window) {
    const uint32_t cnt = uint32_t(std::min<size_t>(window, nq - w0));
    CUDA_CHECK(cudaEventRecord(e0, stream));
    gridQueryKernel<<<(cnt + kBlock - 1) / kBlock, kBlock, 0, stream>>>(
        dSorted.as<float4>(), dCellStart.as<uint32_t>(), dQSorted.as<float4>() + w0, cnt, g, r2, K,
        dCounts.as<uint32_t>(), dNbrs.as<uint32_t>());
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaEventRecord(e1, stream));
    CUDA_CHECK(cudaMemcpyAsync(hCounts.get(), dCounts.get(), 4ull * cnt, cudaMemcpyDeviceToHost, stream));
    if (K) CUDA_CHECK(cudaMemcpyAsync(hNbrs.get(), dNbrs.get(), 4ull * K * cnt, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaEventRecord(e3, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    float a = 0, b = 0;
    CUDA_CHECK(cudaEventElapsedTime(&a, e0, e1));
    CUDA_CHECK(cudaEventElapsedTime(&b, e1, e3));
    queryMs += a;
    d2hMs += b;
    scatterResults(order.data() + w0, cnt, hCounts.as<uint32_t>(), hNbrs.as<uint32_t>(), out);
  }
  m.set("h2d_ms", h2dMs);
  m.set("build_ms", buildMs);
  m.set("query_sort_ms", sortMs);
  m.set("query_ms", queryMs);
  m.set("d2h_ms", d2hMs);
  m.set("search_ms", searchT.ms());
  m.set("index_ms", h2dMs + buildMs);
  m.set("wall_ms", wall.ms());
  m.set("grid_cells", double(numCells));
  m.set("device_bytes_points", double(sizeof(float4) * n + 4ull * (numCells + 1)));
}

}  // namespace ooc
