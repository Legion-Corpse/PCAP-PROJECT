// Host-side micro benchmarks (group "perf": not part of ctest, run with --filter perf).
#include <cstdio>

#include "common/dataset.h"
#include "common/host_utils.h"
#include "common/morton.h"
#include "common/timer.h"
#include "cuda/kernels.h"
#include "ooc/kdtree.h"
#include "test_framework.h"

using namespace ooc;

TEST(perf, host_primitives_40M) {
  Timer t;
  const PointSet P = makeDataset("uniform:40M", 1);
  std::printf("  generate        %8.1f ms\n", t.ms());
  t.reset();
  const Aabb box = computeBounds(P.points);
  std::printf("  bounds          %8.1f ms\n", t.ms());
  t.reset();
  std::vector<uint64_t> keys(P.size());
  parallelFor(int64_t(P.size()), 1 << 16, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; ++i) keys[size_t(i)] = mortonCode(P.points[size_t(i)], box) >> (63 - kOrderKeyBits);
  });
  std::printf("  morton keys     %8.1f ms\n", t.ms());
  t.reset();
  const auto order = sortedOrder(keys, kOrderKeyBits);
  std::printf("  sortedOrder     %8.1f ms\n", t.ms());
  t.reset();
  std::vector<float3> gathered(P.size());
  parallelFor(int64_t(P.size()), 1 << 16, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; ++i) gathered[size_t(i)] = P.points[order[size_t(i)]];
  });
  std::printf("  gather          %8.1f ms\n", t.ms());
  t.reset();
  std::vector<float4> packed(P.size());
  packPoints(P.points.data(), nullptr, P.size(), packed.data());
  std::printf("  pack            %8.1f ms\n", t.ms());
  t.reset();
  const KdTree tree = KdTree::build(packed, 1 << 14);
  std::printf("  kd build        %8.1f ms (%zu nodes)\n", t.ms(), tree.nodes().size());
  CHECK(order.size() == P.size());
}

namespace {
template <class T>
struct NoInit {
  using value_type = T;
  NoInit() = default;
  template <class U> NoInit(const NoInit<U>&) {}
  T* allocate(size_t n) { return std::allocator<T>{}.allocate(n); }
  void deallocate(T* p, size_t n) { std::allocator<T>{}.deallocate(p, n); }
  template <class U> void construct(U* p) noexcept {}
  template <class U, class... A> void construct(U* p, A&&... a) { ::new (static_cast<void*>(p)) U(std::forward<A>(a)...); }
  template <class U> bool operator==(const NoInit<U>&) const { return true; }
  template <class U> bool operator!=(const NoInit<U>&) const { return false; }
};
}  // namespace

TEST(perf, first_touch_160MB) {
  const size_t n = 40'000'000;
  Timer t;
  std::vector<uint32_t> a(n);
  std::printf("  vector(n) zero-fill          %8.1f ms\n", t.ms());
  t.reset();
  std::vector<uint32_t, NoInit<uint32_t>> b(n);
  parallelFor(int64_t(n), 1 << 16, [&](int64_t s, int64_t e) { for (int64_t i = s; i < e; ++i) b[size_t(i)] = uint32_t(i); });
  std::printf("  no-init + parallel touch     %8.1f ms\n", t.ms());
  t.reset();
  parallelFor(int64_t(n), 1 << 16, [&](int64_t s, int64_t e) { for (int64_t i = s; i < e; ++i) a[size_t(i)] = uint32_t(i); });
  std::printf("  rewrite (warm pages)         %8.1f ms\n", t.ms());
  CHECK(a[5] == 5 && b[7] == 7);
}
