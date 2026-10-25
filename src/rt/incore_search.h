// In-core RT-core search (RTNN-style): one GAS over all points, resident on the
// device. This is the upper bound the out-of-core path is compared against,
// and the configuration that fails once the structure exceeds device memory.
#pragma once

#include "common/dataset.h"
#include "common/search_types.h"
#include "ooc/planner.h"

namespace ooc {

class OptixEngine;

struct InCoreOptions {
  QueryOrder order = QueryOrder::Morton;
  size_t windowQueries = 0;  // 0 = automatic
};

// Throws DeviceOutOfMemory when the GAS (or its build) does not fit.
void rtInCoreSearch(OptixEngine& engine, const std::vector<float3>& points, const std::vector<float3>& queries,
                    const SearchParams& params, const InCoreOptions& opts, SearchResult& out, Metrics& metrics);

}  // namespace ooc
