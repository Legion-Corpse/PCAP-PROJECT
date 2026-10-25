#include "common/host_utils.h"

#include <algorithm>
#include <cstring>

#include "common/morton.h"
#include "cuda/kernels.h"

namespace ooc {

void parallelFor(int64_t n, int64_t grain, const std::function<void(int64_t, int64_t)>& body) {
  if (n <= 0) return;
  const int64_t g = grain > 0 ? grain : 1;
  const int64_t chunks = (n + g - 1) / g;
#pragma omp parallel for schedule(dynamic, 1)
  for (int64_t c = 0; c < chunks; ++c) body(c * g, c * g + g < n ? c * g + g : n);
}

void packPoints(const float3* src, const uint32_t* idx, size_t n, float4* dst) {
  const int64_t m = int64_t(n);
#pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < m; ++i) {
    const uint32_t id = idx ? idx[i] : uint32_t(i);
    const float3 p = src[id];
    float w;
    std::memcpy(&w, &id, sizeof(w));
    dst[i] = float4{p.x, p.y, p.z, w};
  }
}

void scatterResults(const uint32_t* order, size_t n, const uint32_t* counts, const uint32_t* neighbors,
                    SearchResult& out) {
  const int64_t m = int64_t(n);
  const size_t K = out.K;
#pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < m; ++i) {
    const uint32_t q = order[i];
    out.counts[q] = counts[i];
    // Only the first min(count, K) slots were written on the device; the rest
    // keep the kNoNeighbor padding from SearchResult::reset.
    const size_t used = std::min<size_t>(counts[i], K);
    if (used) std::memcpy(&out.neighbors[size_t(q) * K], neighbors + size_t(i) * K, used * sizeof(uint32_t));
  }
}

void radixSortPairs(std::vector<uint64_t>& keys, std::vector<uint32_t>& vals, int keyBits) {
  constexpr int kDigit = 12, kBuckets = 1 << kDigit;  // 36-bit order keys: 3 passes
  const int64_t n = int64_t(keys.size());
  if (n < 2) return;
  const int passes = (std::min(64, std::max(1, keyBits)) + kDigit - 1) / kDigit;
  const int64_t parts = std::min<int64_t>(256, std::max<int64_t>(1, n / (1 << 15)));
  const int64_t span = (n + parts - 1) / parts;
  // Scratch is kept per thread: re-faulting hundreds of MB of fresh pages on
  // every call costs more than the sort itself on Windows.
  // (Bound to references: inside the parallel lambdas a bare thread_local name
  // would resolve to the worker thread's instance.)
  thread_local std::vector<uint64_t> tlKeys;
  thread_local std::vector<uint32_t> tlVals;
  thread_local std::vector<int64_t> tlHist;
  std::vector<uint64_t>& k2 = tlKeys;
  std::vector<uint32_t>& v2 = tlVals;
  std::vector<int64_t>& hist = tlHist;
  k2.resize(keys.size());
  v2.resize(vals.size());
  hist.resize(size_t(parts) * kBuckets);
  for (int pass = 0; pass < passes; ++pass) {
    const int shift = pass * kDigit;
    parallelFor(parts, 1, [&](int64_t b, int64_t e) {
      for (int64_t p = b; p < e; ++p) {
        int64_t* h = hist.data() + p * kBuckets;
        std::fill(h, h + kBuckets, 0);
        for (int64_t i = p * span, end = std::min(n, (p + 1) * span); i < end; ++i)
          ++h[(keys[size_t(i)] >> shift) & (kBuckets - 1)];
      }
    });
    int64_t run = 0;  // digit-major, part-minor: keeps the sort stable
    for (int d = 0; d < kBuckets; ++d)
      for (int64_t p = 0; p < parts; ++p) {
        int64_t& h = hist[size_t(p * kBuckets + d)];
        const int64_t c = h;
        h = run;
        run += c;
      }
    parallelFor(parts, 1, [&](int64_t b, int64_t e) {
      for (int64_t p = b; p < e; ++p) {
        int64_t* h = hist.data() + p * kBuckets;
        for (int64_t i = p * span, end = std::min(n, (p + 1) * span); i < end; ++i) {
          const int64_t pos = h[(keys[size_t(i)] >> shift) & (kBuckets - 1)]++;
          k2[size_t(pos)] = keys[size_t(i)];
          v2[size_t(pos)] = vals[size_t(i)];
        }
      }
    });
    keys.swap(k2);
    vals.swap(v2);
  }
}

void radixSortPairsSerial(uint64_t* keys, uint32_t* vals, uint64_t* tmpKeys, uint32_t* tmpVals, size_t n, int keyBits) {
  constexpr int kDigit = 12, kBuckets = 1 << kDigit;
  const int passes = (std::min(64, std::max(1, keyBits)) + kDigit - 1) / kDigit;
  uint32_t count[kBuckets];
  for (int pass = 0; pass < passes; ++pass) {
    const int shift = pass * kDigit;
    std::fill(count, count + kBuckets, 0u);
    for (size_t i = 0; i < n; ++i) ++count[(keys[i] >> shift) & (kBuckets - 1)];
    uint32_t run = 0;
    for (int d = 0; d < kBuckets; ++d) {
      const uint32_t c = count[d];
      count[d] = run;
      run += c;
    }
    for (size_t i = 0; i < n; ++i) {
      const uint32_t pos = count[(keys[i] >> shift) & (kBuckets - 1)]++;
      tmpKeys[pos] = keys[i];
      tmpVals[pos] = vals[i];
    }
    std::swap(keys, tmpKeys);
    std::swap(vals, tmpVals);
  }
  if (passes % 2) {  // result is in the caller's scratch: copy it back
    std::copy(keys, keys + n, tmpKeys);
    std::copy(vals, vals + n, tmpVals);
  }
}

std::vector<uint32_t> sortedOrder(const std::vector<uint64_t>& keys, int keyBits) {
  std::vector<uint64_t> k(keys);
  std::vector<uint32_t> order(keys.size());
  parallelFor(int64_t(order.size()), 1 << 16, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; ++i) order[size_t(i)] = uint32_t(i);
  });
  radixSortPairs(k, order, keyBits);
  return order;
}

void mortonSortInPlace(float4* pts, size_t n) {
  if (n < 2) return;
  Aabb box;
  for (size_t i = 0; i < n; ++i) box.extend(xyz(pts[i]));
  thread_local std::vector<uint64_t> tlKeys;
  thread_local std::vector<uint32_t> tlIdx;
  thread_local std::vector<float4> tlTmp;
  std::vector<uint64_t>& keys = tlKeys;
  std::vector<uint32_t>& idx = tlIdx;
  std::vector<float4>& tmp = tlTmp;
  keys.resize(n);
  idx.resize(n);
  parallelFor(int64_t(n), 1 << 15, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; ++i) {
      keys[size_t(i)] = mortonCode(xyz(pts[size_t(i)]), box) >> (63 - kOrderKeyBits);
      idx[size_t(i)] = uint32_t(i);
    }
  });
  radixSortPairs(keys, idx, kOrderKeyBits);
  tmp.assign(pts, pts + n);
  parallelFor(int64_t(n), 1 << 15, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; ++i) pts[size_t(i)] = tmp[idx[size_t(i)]];
  });
}

}  // namespace ooc
