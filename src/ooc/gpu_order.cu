#include "ooc/gpu_order.h"

#include <cub/device/device_radix_sort.cuh>

#include <algorithm>
#include <cstring>

#include "common/check.h"
#include "common/dataset.h"
#include "common/host_utils.h"
#include "common/morton.h"
#include "common/timer.h"
#include "cuda/buffers.h"
#include "cuda/kernels.h"

namespace ooc {
namespace {

constexpr int kBlock = 256;

struct DeviceNode {
  float split;
  int32_t axis, left, right;
  int32_t tile;  // >= 0: this node is a tile of the frontier
};

__global__ void ownerKeyKernel(const float4* __restrict__ q, uint64_t* __restrict__ keys, uint32_t* __restrict__ vals,
                               uint32_t n, const DeviceNode* __restrict__ nodes, Aabb box, uint32_t numTiles,
                               uint32_t* __restrict__ counts) {
  extern __shared__ uint32_t hist[];
  const bool shared = numTiles <= 8192;
  if (shared)
    for (uint32_t t = threadIdx.x; t < numTiles; t += blockDim.x) hist[t] = 0;
  __syncthreads();
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const float3 p = xyz(q[i]);
    int id = 0;
    while (nodes[id].tile < 0) {  // same rule as KdTree::owner
      const DeviceNode nd = nodes[id];
      id = comp(p, nd.axis) < nd.split ? nd.left : nd.right;
    }
    const uint32_t t = uint32_t(nodes[id].tile);
    keys[i] = (uint64_t(t) << kOrderKeyBits) | (mortonCode(p, box) >> (63 - kOrderKeyBits));
    vals[i] = i;
    if (shared) atomicAdd(&hist[t], 1u);
    else atomicAdd(&counts[t], 1u);
  }
  if (shared) {
    __syncthreads();
    for (uint32_t t = threadIdx.x; t < numTiles; t += blockDim.x)
      if (hist[t]) atomicAdd(&counts[t], hist[t]);
  }
}

}  // namespace

TiledOrderResult gpuTiledOrder(const std::vector<float3>& queries, const KdTree& tree,
                               const std::vector<int>& tileOfNode, size_t numTiles, size_t budgetBytes) {
  NvtxRange range("gpu_tiled_order");
  TiledOrderResult r;
  const size_t M = queries.size();
  const uint32_t T = uint32_t(numTiles);
  if (M == 0) return r;
  Timer total;

  std::vector<DeviceNode> hn(tree.nodes().size());
  for (size_t i = 0; i < hn.size(); ++i) {
    const KdNode& n = tree.nodes()[i];
    hn[i] = DeviceNode{n.split, n.axis, n.left, n.right, tileOfNode[i]};
  }
  int ownerBits = 1;
  while ((uint64_t(1) << ownerBits) < T) ++ownerBits;
  const int endBit = kOrderKeyBits + ownerBits;
  const Aabb box = computeBounds(queries);

  // ~56 bytes of device memory per query in flight (queries, keys, values, CUB scratch).
  // Chunks are capped at 8M queries: larger ones only grow the pinned staging
  // (page-locking hundreds of MB costs more than the extra chunk launches).
  const size_t C = std::min(M, std::clamp<size_t>(budgetBytes / 72, size_t(1) << 16, size_t(8) << 20));
  CudaStream s;
  DeviceBuffer dNodes(sizeof(DeviceNode) * hn.size()), dCounts(4ull * T);
  DeviceBuffer dQ(16 * C), dKeys(8 * C), dKeysAlt(8 * C), dVals(4 * C), dValsAlt(4 * C);
  size_t tempBytes = 0;
  {
    cub::DoubleBuffer<uint64_t> k(dKeys.as<uint64_t>(), dKeysAlt.as<uint64_t>());
    cub::DoubleBuffer<uint32_t> v(dVals.as<uint32_t>(), dValsAlt.as<uint32_t>());
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, tempBytes, k, v, int(C), 0, endBit, s));
  }
  DeviceBuffer dTemp(tempBytes);
  PinnedBuffer hQ(16 * C), hIdx(4 * C), hCounts(4ull * T);
  CUDA_CHECK(cudaMemcpyAsync(dNodes.get(), hn.data(), sizeof(DeviceNode) * hn.size(), cudaMemcpyHostToDevice, s));

  // All chunk outputs, in chunk order; chunkTileStart[c * (T+1) + t] = run start inside chunk c.
  std::vector<uint32_t> idxAll(M);
  const size_t chunks = (M + C - 1) / C;
  std::vector<uint32_t> chunkTileStart(chunks * (T + 1));
  const size_t sharedBytes = T <= 8192 ? 4ull * T : 0;
  for (size_t c = 0; c < chunks; ++c) {
    const size_t c0 = c * C;
    const uint32_t cnt = uint32_t(std::min(C, M - c0));
    packPoints(queries.data() + c0, nullptr, cnt, hQ.as<float4>());
    CUDA_CHECK(cudaMemcpyAsync(dQ.get(), hQ.get(), 16ull * cnt, cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemsetAsync(dCounts.get(), 0, 4ull * T, s));
    ownerKeyKernel<<<(cnt + kBlock - 1) / kBlock, kBlock, sharedBytes, s>>>(
        dQ.as<float4>(), dKeys.as<uint64_t>(), dVals.as<uint32_t>(), cnt, dNodes.as<DeviceNode>(), box, T,
        dCounts.as<uint32_t>());
    CUDA_CHECK_LAST();
    cub::DoubleBuffer<uint64_t> k(dKeys.as<uint64_t>(), dKeysAlt.as<uint64_t>());
    cub::DoubleBuffer<uint32_t> v(dVals.as<uint32_t>(), dValsAlt.as<uint32_t>());
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(dTemp.get(), tempBytes, k, v, int(cnt), 0, endBit, s));
    CUDA_CHECK(cudaMemcpyAsync(hIdx.get(), v.Current(), 4ull * cnt, cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaMemcpyAsync(hCounts.get(), dCounts.get(), 4ull * T, cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    const uint32_t* idx = hIdx.as<uint32_t>();
    parallelFor(cnt, 1 << 16, [&](int64_t b, int64_t e) {
      for (int64_t i = b; i < e; ++i) idxAll[c0 + size_t(i)] = uint32_t(c0) + idx[i];
    });
    uint32_t run = 0;
    for (uint32_t t = 0; t < T; ++t) {
      chunkTileStart[c * (T + 1) + t] = run;
      run += hCounts.as<uint32_t>()[t];
    }
    chunkTileStart[c * (T + 1) + T] = run;
  }
  r.deviceMs = total.ms();

  // Concatenate the per-tile runs of all chunks.
  Timer host;
  std::vector<uint32_t> tileOffset(size_t(T) + 1, 0);
  for (uint32_t t = 0; t < T; ++t) {
    size_t n = 0;
    for (size_t c = 0; c < chunks; ++c) n += chunkTileStart[c * (T + 1) + t + 1] - chunkTileStart[c * (T + 1) + t];
    tileOffset[t + 1] = tileOffset[t] + uint32_t(n);
  }
  r.order.resize(M);
  r.tileStart.assign(tileOffset.begin(), tileOffset.end());
  parallelFor(T, 1, [&](int64_t b, int64_t e) {
    for (int64_t t = b; t < e; ++t) {
      size_t dst = tileOffset[size_t(t)];
      for (size_t c = 0; c < chunks; ++c) {
        const size_t s0 = c * C + chunkTileStart[c * (T + 1) + size_t(t)];
        const size_t n = chunkTileStart[c * (T + 1) + size_t(t) + 1] - chunkTileStart[c * (T + 1) + size_t(t)];
        std::memcpy(r.order.data() + dst, idxAll.data() + s0, 4 * n);
        dst += n;
      }
    }
  });
  r.hostMs = host.ms();
  return r;
}

}  // namespace ooc
