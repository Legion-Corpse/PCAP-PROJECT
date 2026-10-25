// CPU reference implementations (OpenMP). They define correctness for the GPU
// backends and serve as the multicore baseline.
#pragma once

#include "common/dataset.h"
#include "common/search_types.h"

namespace ooc {

// O(|P| * |Q|); only for small inputs and unit tests.
void cpuBruteForce(const std::vector<float3>& points, const std::vector<float3>& queries,
                   const SearchParams& params, SearchResult& out);

// Uniform cell grid with cell size >= r, 27-cell stencil. Queries may lie
// anywhere (also outside the point bounds).
void cpuGridSearch(const std::vector<float3>& points, const std::vector<float3>& queries,
                   const SearchParams& params, SearchResult& out, Metrics* metrics = nullptr);

// Exact counts for a subset of queries, used for sampled verification of large runs.
std::vector<uint32_t> cpuGridCounts(const std::vector<float3>& points, const std::vector<float3>& queries,
                                    const std::vector<size_t>& subset, float radius);

}  // namespace ooc
