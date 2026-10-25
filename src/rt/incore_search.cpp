#include "rt/incore_search.h"

#include <algorithm>

#include "common/host_utils.h"
#include "common/timer.h"
#include "cuda/buffers.h"
#include "cuda/kernels.h"
#include "rt/optix_engine.h"

namespace ooc {

void rtInCoreSearch(OptixEngine& engine, const std::vector<float3>& points, const std::vector<float3>& queries,
                    const SearchParams& params, const InCoreOptions& opts, SearchResult& out, Metrics& m) {
  NvtxRange range("rt_incore");
  const uint32_t n = uint32_t(points.size()), nq = uint32_t(queries.size());
  const uint32_t K = params.maxNeighbors;
  out.reset(nq, K);
  CudaStream stream;
  CudaEvent e0(true), e1(true), e2(true);
  Timer wall;

  const Aabb box = computeBounds(points);
  const float R = conservativeRadius(params.radius, box);

  // Points go to the device in Morton order (needed for compact primitive
  // groups, and better memory locality for the intersection program).
  const uint32_t G = std::max(1u, params.pointsPerPrim);
  DeviceBuffer dPts;
  std::vector<uint32_t> pointOrder;
  CUDA_CHECK(cudaEventRecord(e0, stream));
  mortonSortOnDevice(points, dPts, pointOrder, stream, paddedCount(n, G));
  CUDA_CHECK(cudaEventRecord(e1, stream));
  GasScratch scratch;
  DeviceBuffer gas;
  const GasInfo info = engine.buildGasOwned(dPts.as<float4>(), n, R, gas, scratch, stream, G);
  CUDA_CHECK(cudaEventRecord(e2, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  float h2dMs = 0, buildMs = 0;
  CUDA_CHECK(cudaEventElapsedTime(&h2dMs, e0, e1));
  CUDA_CHECK(cudaEventElapsedTime(&buildMs, e1, e2));
  pointOrder.clear();
  pointOrder.shrink_to_fit();

  Timer searchT, sortT;
  // Morton order is computed on the device (the in-core setting assumes the
  // queries fit); other orders are prepared on the host and streamed per window.
  const bool deviceSort = opts.order == QueryOrder::Morton || opts.order == QueryOrder::Tiled;  // no tiles here
  std::vector<uint32_t> order;
  DeviceBuffer dSorted;
  if (deviceSort) mortonSortOnDevice(queries, dSorted, order, stream);
  else order = queryOrder(queries, opts.order, 1);
  const double sortMs = sortT.ms();

  size_t freeB = 0, totalB = 0;
  CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));
  const size_t perQuery = 2 * (16 + 4ull * (1 + K));
  size_t window = opts.windowQueries ? opts.windowQueries : std::max<size_t>(1 << 16, (freeB / 2) / perQuery);
  window = std::min<size_t>(window, std::max<uint32_t>(nq, 1));
  DeviceBuffer dQ(deviceSort ? 0 : sizeof(float4) * window), dCounts(4 * window), dNbrs(K ? 4ull * K * window : 0);
  DeviceBuffer dParams(sizeof(LaunchParams));
  PinnedBuffer hQ(deviceSort ? 0 : sizeof(float4) * window), hCounts(4 * window), hNbrs(K ? 4ull * K * window : 0);
  PinnedBuffer hParams(sizeof(LaunchParams));
  const float r2 = params.radius * params.radius;
  double queryMs = 0, transferMs = 0;
  for (size_t w0 = 0; w0 < nq; w0 += window) {
    const uint32_t cnt = uint32_t(std::min<size_t>(window, nq - w0));
    LaunchParams& lp = *hParams.as<LaunchParams>();
    lp = LaunchParams{};
    lp.handle = info.handle;
    lp.points = dPts.as<float4>();
    lp.group = G;
    lp.queries = deviceSort ? dSorted.as<float4>() + w0 : dQ.as<float4>();
    lp.queryList = nullptr;
    lp.counts = dCounts.as<uint32_t>();
    lp.neighbors = K ? dNbrs.as<uint32_t>() : nullptr;
    lp.numQueries = cnt;
    lp.maxNeighbors = K;
    lp.radius2 = r2;
    CUDA_CHECK(cudaEventRecord(e0, stream));
    if (!deviceSort) {
      packPoints(queries.data(), order.data() + w0, cnt, hQ.as<float4>());
      CUDA_CHECK(cudaMemcpyAsync(dQ.get(), hQ.get(), sizeof(float4) * cnt, cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(cudaMemcpyAsync(dParams.get(), &lp, sizeof(lp), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(dCounts.get(), 0, 4ull * cnt, stream));
    CUDA_CHECK(cudaEventRecord(e1, stream));
    engine.launch(dParams.dptr(), cnt, stream);
    CUDA_CHECK(cudaEventRecord(e2, stream));
    CUDA_CHECK(cudaMemcpyAsync(hCounts.get(), dCounts.get(), 4ull * cnt, cudaMemcpyDeviceToHost, stream));
    if (K) CUDA_CHECK(cudaMemcpyAsync(hNbrs.get(), dNbrs.get(), 4ull * K * cnt, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    float a = 0, b = 0;
    CUDA_CHECK(cudaEventElapsedTime(&a, e0, e1));
    CUDA_CHECK(cudaEventElapsedTime(&b, e1, e2));
    transferMs += a;
    queryMs += b;
    scatterResults(order.data() + w0, cnt, hCounts.as<uint32_t>(), hNbrs.as<uint32_t>(), out);
  }
  m.set("h2d_ms", h2dMs);
  m.set("build_ms", buildMs);
  m.set("query_sort_ms", sortMs);
  m.set("query_ms", queryMs);
  m.set("query_h2d_ms", transferMs);
  m.set("search_ms", searchT.ms());
  m.set("index_ms", h2dMs + buildMs);
  m.set("wall_ms", wall.ms());
  m.set("gas_bytes", double(info.bytes));
  m.set("gas_bytes_per_point", n ? double(info.bytes) / n : 0.0);
  m.set("points_per_prim", double(G));
  m.set("device_bytes_resident", double(info.bytes + sizeof(float4) * paddedCount(n, G)));
}

}  // namespace ooc
