// Types shared by every search backend: parameters, results and metrics.
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace ooc {

constexpr uint32_t kNoNeighbor = 0xffffffffu;

struct SearchParams {
  float radius = 0.f;
  // 0: count-only. K > 0: also return up to K neighbor ids per query (the
  // first K found; the count stays exact, so truncation is detectable).
  uint32_t maxNeighbors = 0;
  // RT backends: points per BVH primitive (point bucketing, see device_programs.cu).
  uint32_t pointsPerPrim = 1;
};

struct SearchResult {
  std::vector<uint32_t> counts;     // |N(q)| per query, original query order
  std::vector<uint32_t> neighbors;  // queries * K ids, kNoNeighbor padded (only when K > 0)
  uint32_t K = 0;

  void reset(size_t queries, uint32_t k) {
    K = k;
    counts.assign(queries, 0u);
    neighbors.assign(k ? queries * size_t(k) : 0, kNoNeighbor);
  }
  uint64_t totalNeighbors() const {
    uint64_t s = 0;
    for (uint32_t c : counts) s += c;
    return s;
  }
};

// Flat key/value metrics that the CLI prints and serializes to JSON.
class Metrics {
 public:
  void set(const std::string& key, double v) { upsert(num_, key, v); }
  void set(const std::string& key, const std::string& v) { upsert(str_, key, v); }
  void add(const std::string& key, double v) {
    for (auto& kv : num_) if (kv.first == key) { kv.second += v; return; }
    num_.emplace_back(key, v);
  }
  double get(const std::string& key, double fallback = 0.0) const {
    for (const auto& kv : num_) if (kv.first == key) return kv.second;
    return fallback;
  }
  std::string str(const std::string& key) const {
    for (const auto& kv : str_) if (kv.first == key) return kv.second;
    return {};
  }
  bool has(const std::string& key) const {
    for (const auto& kv : num_) if (kv.first == key) return true;
    return false;
  }
  const std::vector<std::pair<std::string, double>>& numbers() const { return num_; }
  const std::vector<std::pair<std::string, std::string>>& strings() const { return str_; }

  std::string toJson() const {
    std::string out = "{";
    bool first = true;
    auto sep = [&] { if (!first) out += ","; first = false; };
    for (const auto& kv : str_) { sep(); out += "\"" + escape(kv.first) + "\":\"" + escape(kv.second) + "\""; }
    char buf[64];
    for (const auto& kv : num_) {
      sep();
      std::snprintf(buf, sizeof(buf), "%.10g", kv.second);
      out += "\"" + escape(kv.first) + "\":" + buf;
    }
    return out + "}";
  }

 private:
  template <class T>
  static void upsert(std::vector<std::pair<std::string, T>>& v, const std::string& k, const T& x) {
    for (auto& kv : v) if (kv.first == k) { kv.second = x; return; }
    v.emplace_back(k, x);
  }
  static std::string escape(const std::string& s) {
    std::string o;
    for (char c : s) {
      if (c == '"' || c == '\\') { o += '\\'; o += c; }
      else if (c == '\n') o += "\\n";
      else o += c;
    }
    return o;
  }
  std::vector<std::pair<std::string, double>> num_;
  std::vector<std::pair<std::string, std::string>> str_;
};

}  // namespace ooc
