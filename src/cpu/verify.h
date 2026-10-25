// Result verification against the CPU reference.
#pragma once

#include <string>

#include "common/search_types.h"
#include "common/vec_math.h"

namespace ooc {

struct VerifyReport {
  size_t checked = 0;
  size_t countMismatches = 0;
  size_t listErrors = 0;
  std::string firstError;
  bool ok() const { return countMismatches == 0 && listErrors == 0; }
};

// Every query: exact counts; neighbor rows must be valid, distinct, and equal
// to the reference set whenever nothing was truncated.
VerifyReport verifyAgainst(const std::vector<float3>& points, const std::vector<float3>& queries,
                           const SearchParams& params, const SearchResult& reference, const SearchResult& got);

// Random query indices for sampled verification (all of them if samples >= m).
std::vector<size_t> sampleQueries(size_t m, size_t samples, uint64_t seed);

// Checks the queries in `subset` against reference counts (from cpuGridCounts),
// plus neighbor-row validity. Cheap enough for 100M-point runs, and the
// reference can be shared by every backend of one invocation.
VerifyReport verifySampled(const std::vector<float3>& points, const std::vector<float3>& queries,
                           const SearchParams& params, const SearchResult& got, const std::vector<size_t>& subset,
                           const std::vector<uint32_t>& reference);

}  // namespace ooc
