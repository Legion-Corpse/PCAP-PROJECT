// In-core CUDA uniform-grid fixed-radius search: the standard non-RT GPU baseline
// (cell lists as in cuNSearch / FRNN libraries, Morton-sorted queries).
#pragma once

#include "common/dataset.h"
#include "common/search_types.h"

namespace ooc {

// Throws DeviceOutOfMemory when the point set does not fit on the device.
// `windowQueries` bounds how many queries are in flight (0 = automatic).
void cudaGridSearch(const std::vector<float3>& points, const std::vector<float3>& queries, const SearchParams& params,
                    SearchResult& out, Metrics& metrics, size_t windowQueries = 0);

}  // namespace ooc
