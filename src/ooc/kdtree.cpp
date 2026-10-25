#include "ooc/kdtree.h"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <limits>

#include "common/host_utils.h"

namespace ooc {
namespace {

constexpr int kMaxDepth = 60;

float coordOf(const float4& v, int axis) { return axis == 0 ? v.x : (axis == 1 ? v.y : v.z); }

struct Decision {
  bool split = false;
  int axis = -1;
  float value = 0.f;
  uint32_t mid = 0;
};

Decision decide(std::vector<float4>& pts, uint32_t begin, uint32_t end, uint32_t cap, int depth) {
  Decision d;
  const uint32_t n = end - begin;
  if (n <= cap || depth >= kMaxDepth) return d;
  Aabb box;
  for (uint32_t i = begin; i < end; ++i) box.extend(xyz(pts[i]));
  const float3 e = box.extent();
  int axes[3] = {0, 1, 2};
  const float ext[3] = {e.x, e.y, e.z};
  std::sort(axes, axes + 3, [&](int a, int b) { return ext[a] > ext[b]; });
  float4* first = pts.data() + begin;
  float4* last = pts.data() + end;
  for (int axis : axes) {
    if (!(ext[axis] > 0.f)) break;
    float4* mid = first + n / 2;
    std::nth_element(first, mid, last, [axis](const float4& a, const float4& b) { return coordOf(a, axis) < coordOf(b, axis); });
    float split = coordOf(*mid, axis);
    auto below = [axis, &split](const float4& v) { return coordOf(v, axis) < split; };
    float4* m = std::partition(first, last, below);
    if (m == first) {
      // The median equals the minimum: split just above it instead.
      float next = std::numeric_limits<float>::infinity();
      for (float4* p = first; p != last; ++p) {
        const float c = coordOf(*p, axis);
        if (c > split && c < next) next = c;
      }
      if (std::isinf(next)) continue;  // all coordinates equal along this axis
      split = next;
      m = std::partition(first, last, below);
    }
    d.split = true;
    d.axis = axis;
    d.value = split;
    d.mid = uint32_t(m - pts.data());
    return d;
  }
  return d;  // all points identical: keep an (oversized) leaf
}

// Nodes above this size are split with all threads: parallel bounds, a sampled
// median, and a stable parallel partition through a scratch buffer.
constexpr uint32_t kParallelNode = 1u << 20;

Decision decideLarge(std::vector<float4>& pts, uint32_t begin, uint32_t end, uint32_t cap, int depth,
                     std::vector<float4>& scratch) {
  const uint32_t n = end - begin;
  if (n <= cap || depth >= kMaxDepth) return {};
  constexpr int64_t grain = 1 << 16;
  const int64_t parts = (int64_t(n) + grain - 1) / grain;
  std::vector<Aabb> boxes(static_cast<size_t>(parts));
  float4* base = pts.data() + begin;
  parallelFor(n, grain, [&](int64_t b, int64_t e) {
    Aabb local;
    for (int64_t i = b; i < e; ++i) local.extend(xyz(base[i]));
    boxes[size_t(b / grain)] = local;
  });
  Aabb box;
  for (const auto& b : boxes) box.extend(b);
  const float3 e = box.extent();
  int axes[3] = {0, 1, 2};
  const float ext[3] = {e.x, e.y, e.z};
  std::sort(axes, axes + 3, [&](int a, int b) { return ext[a] > ext[b]; });
  for (int axis : axes) {
    if (!(ext[axis] > 0.f)) break;
    constexpr uint32_t kSamples = 8191;
    std::vector<float> sample(kSamples);
    for (uint32_t k = 0; k < kSamples; ++k) sample[k] = coordOf(base[uint64_t(k) * n / kSamples], axis);
    std::nth_element(sample.begin(), sample.begin() + kSamples / 2, sample.end());
    const float split = sample[kSamples / 2];
    std::vector<uint32_t> below(static_cast<size_t>(parts) + 1, 0u);
    parallelFor(parts, 1, [&](int64_t b, int64_t e2) {
      for (int64_t p = b; p < e2; ++p) {
        uint32_t c = 0;
        for (int64_t i = p * grain, stop = std::min<int64_t>(n, (p + 1) * grain); i < stop; ++i)
          c += coordOf(base[i], axis) < split;
        below[size_t(p)] = c;
      }
    });
    uint32_t left = 0;
    std::vector<uint32_t> leftOff(static_cast<size_t>(parts)), rightOff(static_cast<size_t>(parts));
    for (int64_t p = 0; p < parts; ++p) { leftOff[size_t(p)] = left; left += below[size_t(p)]; }
    if (left == 0 || left == n) break;  // degenerate sample: use the exact serial path
    uint32_t right = left;
    for (int64_t p = 0; p < parts; ++p) {
      rightOff[size_t(p)] = right;
      right += uint32_t(std::min<int64_t>(n, (p + 1) * grain) - p * grain) - below[size_t(p)];
    }
    scratch.resize(n);
    parallelFor(parts, 1, [&](int64_t b, int64_t e2) {
      for (int64_t p = b; p < e2; ++p) {
        uint32_t l = leftOff[size_t(p)], r = rightOff[size_t(p)];
        for (int64_t i = p * grain, stop = std::min<int64_t>(n, (p + 1) * grain); i < stop; ++i)
          scratch[coordOf(base[i], axis) < split ? l++ : r++] = base[i];
      }
    });
    parallelFor(n, grain, [&](int64_t b, int64_t e2) { std::copy(scratch.begin() + b, scratch.begin() + e2, base + b); });
    Decision d;
    d.split = true;
    d.axis = axis;
    d.value = split;
    d.mid = begin + left;
    return d;
  }
  return decide(pts, begin, end, cap, depth);
}

}  // namespace

KdTree KdTree::build(std::vector<float4>& pts, uint32_t leafCapacity) {
  KdTree t;
  const float inf = std::numeric_limits<float>::infinity();
  KdNode root;
  root.cell.lo = make_f3(-inf, -inf, -inf);
  root.cell.hi = make_f3(inf, inf, inf);
  root.begin = 0;
  root.end = uint32_t(pts.size());
  t.nodes_.push_back(root);
  std::vector<int> level{0};
  const uint32_t cap = std::max<uint32_t>(1, leafCapacity);
  std::vector<float4> scratch;
  while (!level.empty()) {
    std::vector<Decision> dec(level.size());
    std::vector<int64_t> small;
    for (size_t i = 0; i < level.size(); ++i) {
      const KdNode& n = t.nodes_[size_t(level[i])];
      if (n.count() > kParallelNode) dec[i] = decideLarge(pts, n.begin, n.end, cap, n.depth, scratch);
      else small.push_back(int64_t(i));
    }
    const int64_t count = int64_t(small.size());
#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t k = 0; k < count; ++k) {
      const size_t i = size_t(small[size_t(k)]);
      const KdNode& n = t.nodes_[size_t(level[i])];
      dec[i] = decide(pts, n.begin, n.end, cap, n.depth);
    }
    std::vector<int> next;
    for (size_t i = 0; i < level.size(); ++i) {
      if (!dec[i].split) continue;
      const int id = level[i];
      KdNode l = t.nodes_[size_t(id)], r = t.nodes_[size_t(id)];
      const int axis = dec[i].axis;
      const float s = dec[i].value;
      l.axis = r.axis = -1;
      l.left = l.right = r.left = r.right = -1;
      l.depth = r.depth = uint8_t(t.nodes_[size_t(id)].depth + 1);
      l.end = dec[i].mid;
      r.begin = dec[i].mid;
      if (axis == 0) { l.cell.hi.x = s; r.cell.lo.x = s; }
      if (axis == 1) { l.cell.hi.y = s; r.cell.lo.y = s; }
      if (axis == 2) { l.cell.hi.z = s; r.cell.lo.z = s; }
      const int li = int(t.nodes_.size());
      t.nodes_.push_back(l);
      t.nodes_.push_back(r);
      KdNode& parent = t.nodes_[size_t(id)];
      parent.axis = int8_t(axis);
      parent.split = s;
      parent.left = li;
      parent.right = li + 1;
      t.maxDepth_ = std::max(t.maxDepth_, int(l.depth));
      next.push_back(li);
      next.push_back(li + 1);
    }
    level.swap(next);
  }
  return t;
}

std::vector<int> KdTree::frontier(int depth) const {
  std::vector<int> out;
  std::vector<int> stack{0};
  while (!stack.empty()) {
    const int id = stack.back();
    stack.pop_back();
    const KdNode& n = nodes_[size_t(id)];
    if (n.leaf() || n.depth >= depth) {
      out.push_back(id);
      continue;
    }
    stack.push_back(n.right);
    stack.push_back(n.left);
  }
  return out;
}

int KdTree::depthForCapacity(uint32_t capacity) const {
  for (int d = 0; d <= maxDepth_; ++d) {
    uint32_t worst = 0;
    for (int id : frontier(d)) worst = std::max(worst, nodes_[size_t(id)].count());
    if (worst <= capacity) return d;
  }
  return maxDepth_;
}

std::vector<int> KdTree::tileMap(const std::vector<int>& frontier) const {
  std::vector<int> map(nodes_.size(), -1);
  for (size_t i = 0; i < frontier.size(); ++i) map[size_t(frontier[i])] = int(i);
  return map;
}

int KdTree::owner(float3 p, const std::vector<int>& tileOfNode) const {
  int id = 0;
  while (tileOfNode[size_t(id)] < 0) {
    const KdNode& n = nodes_[size_t(id)];
    id = comp(p, n.axis) < n.split ? n.left : n.right;
  }
  return tileOfNode[size_t(id)];
}

std::vector<uint32_t> KdTree::nodeCoverage(const std::vector<float3>& sample, double R) const {
  const int threads = omp_get_max_threads();
  std::vector<std::vector<uint32_t>> partial(size_t(threads), std::vector<uint32_t>(nodes_.size(), 0u));
  const double R2 = R * R;
  const int64_t m = int64_t(sample.size());
#pragma omp parallel
  {
    auto& cov = partial[size_t(omp_get_thread_num())];
    std::vector<int> stack;
#pragma omp for schedule(dynamic, 256)
    for (int64_t i = 0; i < m; ++i) {
      const float3 p = sample[size_t(i)];
      stack.assign(1, 0);
      while (!stack.empty()) {
        const int id = stack.back();
        stack.pop_back();
        const KdNode& n = nodes_[size_t(id)];
        if (n.cell.dist2To(p) > R2) continue;
        cov[size_t(id)]++;
        if (!n.leaf()) {
          stack.push_back(n.left);
          stack.push_back(n.right);
        }
      }
    }
  }
  std::vector<uint32_t> total(nodes_.size(), 0u);
  for (const auto& v : partial)
    for (size_t j = 0; j < v.size(); ++j) total[j] += v[j];
  return total;
}

std::vector<uint32_t> KdTree::ownerCounts(const std::vector<float3>& sample) const {
  const int threads = omp_get_max_threads();
  std::vector<std::vector<uint32_t>> partial(size_t(threads), std::vector<uint32_t>(nodes_.size(), 0u));
  const int64_t m = int64_t(sample.size());
#pragma omp parallel
  {
    auto& cnt = partial[size_t(omp_get_thread_num())];
#pragma omp for schedule(static)
    for (int64_t i = 0; i < m; ++i) {
      const float3 p = sample[size_t(i)];
      int id = 0;
      for (;;) {
        cnt[size_t(id)]++;
        const KdNode& n = nodes_[size_t(id)];
        if (n.leaf()) break;
        id = comp(p, n.axis) < n.split ? n.left : n.right;
      }
    }
  }
  std::vector<uint32_t> total(nodes_.size(), 0u);
  for (const auto& v : partial)
    for (size_t j = 0; j < v.size(); ++j) total[j] += v[j];
  return total;
}

}  // namespace ooc
