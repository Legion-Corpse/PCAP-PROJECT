// Host-callable wrappers around small CUDA kernels.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include "common/vec_math.h"
#include "cuda/buffers.h"

namespace ooc {

// Layout-compatible with OptixAabb.
struct AabbF6 { float minX, minY, minZ, maxX, maxY, maxZ; };

// One box per group of `group` consecutive points: their bounds grown by
// `halfWidth` (NaN padding points are skipped). numPrims = ceil(n / group).
void launchMakeAabbs(const float4* points, AabbF6* aabbs, uint32_t numPrims, uint32_t group, float halfWidth,
                     cudaStream_t stream);

// Fills dst[n, capacity) with NaN points (id kNoNeighbor) so groups are whole.
void padPointsNaN(float4* dst, size_t n, size_t capacity, cudaStream_t stream);
inline size_t paddedCount(size_t n, uint32_t group) { return (n + group - 1) / group * group; }

// keys[i] = top kOrderKeyBits bits of the Morton code of xyz(pts[i]) in `box`, vals[i] = i.
void launchMortonKeys(const float4* pts, uint64_t* keys, uint32_t* vals, uint32_t n, Aabb box, cudaStream_t stream);

// dst[i] = src[perm[i]]
void launchGatherFloat4(const float4* src, const uint32_t* perm, float4* dst, uint32_t n, cudaStream_t stream);

// Sorts (key, value) pairs by key with CUB radix sort; keys/values are sorted in place
// using the provided double-buffer scratch (same sizes). Returns nothing; stream-ordered.
void radixSortPairs64(uint64_t* keys, uint32_t* vals, uint64_t* keysAlt, uint32_t* valsAlt, uint32_t n, int endBit,
                      cudaStream_t stream);
void radixSortPairs32(uint32_t* keys, uint32_t* vals, uint32_t* keysAlt, uint32_t* valsAlt, uint32_t n, int endBit,
                      cudaStream_t stream);

// *dst = value, as a 1-thread kernel: keeps tiny parameter updates off the copy
// engines (which are busy streaming tiles) and in compute-stream order.
void launchWriteU64(uint64_t* dst, uint64_t value, cudaStream_t stream);

// Morton keys are truncated to this many bits (12 per axis) for ordering:
// finer than any tile or warp footprint, and half the radix-sort passes.
constexpr int kOrderKeyBits = 36;

// Uploads `queries`, sorts them by Morton code on the device and leaves the
// sorted copy (xyz + original index bits) in `dSorted`; `order[i]` is the
// original index of sorted query i. Needs ~44 bytes of temporary device memory
// per query.
// `capacity` (>= n) sizes dSorted; the tail is NaN-padded.
void mortonSortOnDevice(const std::vector<float3>& queries, DeviceBuffer& dSorted, std::vector<uint32_t>& order,
                        cudaStream_t stream, size_t capacity = 0);

}  // namespace ooc
