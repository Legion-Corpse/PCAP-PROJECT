#include "cpu/verify.h"

#include <algorithm>
#include <random>
#include <sstream>

#include "cpu/cpu_search.h"

namespace ooc {
namespace {

// Checks row q of `got`: min(count, K) distinct true neighbors, rest padding.
bool rowValid(const std::vector<float3>& points, float3 q, float r2, const SearchResult& got, size_t qi,
              std::string& why) {
  const uint32_t K = got.K;
  const uint32_t n = std::min(got.counts[qi], K);
  const uint32_t* row = got.neighbors.data() + qi * K;
  std::vector<uint32_t> ids(row, row + n);
  for (uint32_t id : ids) {
    if (id >= points.size()) { why = "invalid id"; return false; }
    if (dist2(points[id], q) > r2) { why = "id outside radius"; return false; }
  }
  std::sort(ids.begin(), ids.end());
  if (std::adjacent_find(ids.begin(), ids.end()) != ids.end()) { why = "duplicate id"; return false; }
  for (uint32_t j = n; j < K; ++j)
    if (row[j] != kNoNeighbor) { why = "padding overwritten"; return false; }
  return true;
}

void note(VerifyReport& r, size_t qi, const std::string& what) {
  if (!r.firstError.empty()) return;
  std::ostringstream os;
  os << "query " << qi << ": " << what;
  r.firstError = os.str();
}

}  // namespace

VerifyReport verifyAgainst(const std::vector<float3>& points, const std::vector<float3>& queries,
                           const SearchParams& params, const SearchResult& ref, const SearchResult& got) {
  VerifyReport r;
  const float r2 = params.radius * params.radius;
  r.checked = queries.size();
  for (size_t qi = 0; qi < queries.size(); ++qi) {
    if (ref.counts[qi] != got.counts[qi]) {
      ++r.countMismatches;
      note(r, qi, "count " + std::to_string(got.counts[qi]) + " != reference " + std::to_string(ref.counts[qi]));
      continue;
    }
    if (!got.K) continue;
    std::string why;
    if (!rowValid(points, queries[qi], r2, got, qi, why)) {
      ++r.listErrors;
      note(r, qi, why);
      continue;
    }
    if (got.counts[qi] <= got.K && ref.K == got.K) {
      const uint32_t n = got.counts[qi];
      std::vector<uint32_t> a(got.neighbors.begin() + qi * got.K, got.neighbors.begin() + qi * got.K + n);
      std::vector<uint32_t> b(ref.neighbors.begin() + qi * ref.K, ref.neighbors.begin() + qi * ref.K + n);
      std::sort(a.begin(), a.end());
      std::sort(b.begin(), b.end());
      if (a != b) {
        ++r.listErrors;
        note(r, qi, "neighbor set differs from reference");
      }
    }
  }
  return r;
}

std::vector<size_t> sampleQueries(size_t m, size_t samples, uint64_t seed) {
  std::vector<size_t> subset;
  if (samples >= m) {
    subset.resize(m);
    for (size_t i = 0; i < m; ++i) subset[i] = i;
  } else if (m) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<size_t> pick(0, m - 1);
    subset.resize(samples);
    for (auto& s : subset) s = pick(rng);
  }
  return subset;
}

VerifyReport verifySampled(const std::vector<float3>& points, const std::vector<float3>& queries,
                           const SearchParams& params, const SearchResult& got, const std::vector<size_t>& subset,
                           const std::vector<uint32_t>& ref) {
  VerifyReport r;
  const float r2 = params.radius * params.radius;
  r.checked = subset.size();
  for (size_t i = 0; i < subset.size(); ++i) {
    const size_t qi = subset[i];
    if (ref[i] != got.counts[qi]) {
      ++r.countMismatches;
      note(r, qi, "count " + std::to_string(got.counts[qi]) + " != reference " + std::to_string(ref[i]));
      continue;
    }
    std::string why;
    if (got.K && !rowValid(points, queries[qi], r2, got, qi, why)) {
      ++r.listErrors;
      note(r, qi, why);
    }
  }
  return r;
}

}  // namespace ooc
