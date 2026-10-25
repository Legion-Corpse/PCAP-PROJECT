// Parallel host-side helpers used around device transfers.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include "common/search_types.h"
#include "common/vec_math.h"

namespace ooc {

// Runs body(begin, end) over [0, n) in chunks of `grain` on all OpenMP threads.
// Type-erased on purpose: MSVC 19.44 overflows its stack compiling some
// #pragma omp loops whose (heavily inlined) bodies are written in place.
void parallelFor(int64_t n, int64_t grain, const std::function<void(int64_t, int64_t)>& body);

// dst[i] = {src[idx[i]].xyz, bits(idx[i])}; idx == nullptr means identity.
void packPoints(const float3* src, const uint32_t* idx, size_t n, float4* dst);

// Writes window-local results back to original query order:
//   out.counts[order[i]] = counts[i]; neighbor rows likewise (when K > 0).
void scatterResults(const uint32_t* order, size_t n, const uint32_t* counts, const uint32_t* neighbors,
                    SearchResult& out);

// Stable parallel LSD radix sort of (key, value) pairs on the low `keyBits` bits.
void radixSortPairs(std::vector<uint64_t>& keys, std::vector<uint32_t>& vals, int keyBits);

// Single-threaded LSD radix sort of n (key, value) pairs on the low `keyBits`
// bits, using caller-provided scratch of the same length. For many small,
// independent sorts run in parallel (e.g. one per tile).
void radixSortPairsSerial(uint64_t* keys, uint32_t* vals, uint64_t* tmpKeys, uint32_t* tmpVals, size_t n, int keyBits);

// Reorders points (xyz + id) along the Morton curve of their own bounding box.
void mortonSortInPlace(float4* pts, size_t n);

// Indices 0..n-1 stably sorted by key (parallel radix sort).
std::vector<uint32_t> sortedOrder(const std::vector<uint64_t>& keys, int keyBits = 64);

}  // namespace ooc
