#include "cuda/kernels.h"

#include <cub/device/device_radix_sort.cuh>

#include "common/check.h"
#include "common/dataset.h"
#include "common/host_utils.h"
#include "common/morton.h"
#include "cuda/buffers.h"

namespace ooc {
namespace {

constexpr int kBlock = 256;
inline unsigned gridFor(uint32_t n) { return (n + kBlock - 1) / kBlock; }

__global__ void makeAabbsKernel(const float4* __restrict__ pts, AabbF6* __restrict__ out, uint32_t numPrims,
                                uint32_t group, float h) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numPrims) return;
  float lx = INFINITY, ly = INFINITY, lz = INFINITY, hx = -INFINITY, hy = -INFINITY, hz = -INFINITY;
  for (uint32_t j = 0; j < group; ++j) {
    const float4 p = pts[size_t(i) * group + j];
    if (isnan(p.x)) continue;
    lx = fminf(lx, p.x); ly = fminf(ly, p.y); lz = fminf(lz, p.z);
    hx = fmaxf(hx, p.x); hy = fmaxf(hy, p.y); hz = fmaxf(hz, p.z);
  }
  out[i] = AabbF6{lx - h, ly - h, lz - h, hx + h, hy + h, hz + h};
}

__global__ void mortonKeysKernel(const float4* __restrict__ pts, uint64_t* __restrict__ keys, uint32_t* __restrict__ vals,
                                 uint32_t n, Aabb box) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  keys[i] = mortonCode(xyz(pts[i]), box) >> (63 - kOrderKeyBits);
  vals[i] = i;
}

__global__ void writeU64Kernel(uint64_t* dst, uint64_t v) { *dst = v; }

__global__ void gatherKernel(const float4* __restrict__ src, const uint32_t* __restrict__ perm, float4* __restrict__ dst,
                             uint32_t n) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = src[perm[i]];
}

template <class K>
void sortPairs(K* keys, uint32_t* vals, K* keysAlt, uint32_t* valsAlt, uint32_t n, int endBit, cudaStream_t s) {
  if (n == 0) return;
  cub::DoubleBuffer<K> dk(keys, keysAlt);
  cub::DoubleBuffer<uint32_t> dv(vals, valsAlt);
  size_t tempBytes = 0;
  CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, tempBytes, dk, dv, n, 0, endBit, s));
  DeviceBuffer temp(tempBytes);
  CUDA_CHECK(cub::DeviceRadixSort::SortPairs(temp.get(), tempBytes, dk, dv, n, 0, endBit, s));
  if (dk.Current() != keys) CUDA_CHECK(cudaMemcpyAsync(keys, dk.Current(), sizeof(K) * n, cudaMemcpyDeviceToDevice, s));
  if (dv.Current() != vals)
    CUDA_CHECK(cudaMemcpyAsync(vals, dv.Current(), sizeof(uint32_t) * n, cudaMemcpyDeviceToDevice, s));
  CUDA_CHECK(cudaStreamSynchronize(s));  // `temp` is freed on return
}

}  // namespace

void launchMakeAabbs(const float4* points, AabbF6* aabbs, uint32_t numPrims, uint32_t group, float halfWidth,
                     cudaStream_t stream) {
  if (numPrims == 0) return;
  makeAabbsKernel<<<gridFor(numPrims), kBlock, 0, stream>>>(points, aabbs, numPrims, group, halfWidth);
  CUDA_CHECK_LAST();
}

void padPointsNaN(float4* dst, size_t n, size_t capacity, cudaStream_t stream) {
  if (capacity > n) CUDA_CHECK(cudaMemsetAsync(dst + n, 0xff, sizeof(float4) * (capacity - n), stream));
}

void launchMortonKeys(const float4* pts, uint64_t* keys, uint32_t* vals, uint32_t n, Aabb box, cudaStream_t stream) {
  if (n == 0) return;
  mortonKeysKernel<<<gridFor(n), kBlock, 0, stream>>>(pts, keys, vals, n, box);
  CUDA_CHECK_LAST();
}

void launchWriteU64(uint64_t* dst, uint64_t value, cudaStream_t stream) {
  writeU64Kernel<<<1, 1, 0, stream>>>(dst, value);
  CUDA_CHECK_LAST();
}

void launchGatherFloat4(const float4* src, const uint32_t* perm, float4* dst, uint32_t n, cudaStream_t stream) {
  if (n == 0) return;
  gatherKernel<<<gridFor(n), kBlock, 0, stream>>>(src, perm, dst, n);
  CUDA_CHECK_LAST();
}

void radixSortPairs64(uint64_t* keys, uint32_t* vals, uint64_t* keysAlt, uint32_t* valsAlt, uint32_t n, int endBit,
                      cudaStream_t stream) {
  sortPairs(keys, vals, keysAlt, valsAlt, n, endBit, stream);
}

void radixSortPairs32(uint32_t* keys, uint32_t* vals, uint32_t* keysAlt, uint32_t* valsAlt, uint32_t n, int endBit,
                      cudaStream_t stream) {
  sortPairs(keys, vals, keysAlt, valsAlt, n, endBit, stream);
}

void mortonSortOnDevice(const std::vector<float3>& queries, DeviceBuffer& dSorted, std::vector<uint32_t>& order,
                        cudaStream_t stream, size_t capacity) {
  const uint32_t nq = uint32_t(queries.size());
  PinnedBuffer packed(sizeof(float4) * std::max<uint32_t>(nq, 1));
  packPoints(queries.data(), nullptr, nq, packed.as<float4>());
  capacity = std::max<size_t>({capacity, nq, 1});
  dSorted.alloc(sizeof(float4) * capacity);
  padPointsNaN(dSorted.as<float4>(), nq, capacity, stream);
  DeviceBuffer dQ(sizeof(float4) * std::max<uint32_t>(nq, 1));
  DeviceBuffer keys(8ull * nq), keysAlt(8ull * nq), vals(4ull * nq), valsAlt(4ull * nq);
  CUDA_CHECK(cudaMemcpyAsync(dQ.get(), packed.get(), sizeof(float4) * nq, cudaMemcpyHostToDevice, stream));
  launchMortonKeys(dQ.as<float4>(), keys.as<uint64_t>(), vals.as<uint32_t>(), nq, computeBounds(queries), stream);
  radixSortPairs64(keys.as<uint64_t>(), vals.as<uint32_t>(), keysAlt.as<uint64_t>(), valsAlt.as<uint32_t>(), nq,
                   kOrderKeyBits, stream);
  launchGatherFloat4(dQ.as<float4>(), vals.as<uint32_t>(), dSorted.as<float4>(), nq, stream);
  order.resize(nq);
  CUDA_CHECK(cudaMemcpyAsync(order.data(), vals.get(), 4ull * nq, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace ooc
