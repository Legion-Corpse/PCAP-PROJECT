// Host-only tests: data generation, Morton codes, CPU reference, kd tiling, planner.
#include <algorithm>
#include <cmath>
#include <set>

#include "common/dataset.h"
#include "common/host_utils.h"
#include "common/morton.h"
#include "cpu/cpu_search.h"
#include "cpu/verify.h"
#include "ooc/kdtree.h"
#include "ooc/planner.h"
#include "test_framework.h"
#include "test_util.h"

using namespace ooc;

TEST(cpu, parse_count) {
  CHECK_EQ(parseCount("10"), 10ull);
  CHECK_EQ(parseCount("2K"), 2000ull);
  CHECK_EQ(parseCount("1.5M"), 1500000ull);
  CHECK_EQ(parseCount("1G"), 1000000000ull);
}

TEST(cpu, morton_codes) {
  CHECK_EQ(mortonEncode(1, 0, 0), 1ull);
  CHECK_EQ(mortonEncode(0, 1, 0), 2ull);
  CHECK_EQ(mortonEncode(0, 0, 1), 4ull);
  CHECK_EQ(mortonEncode(3, 3, 3), 63ull);
  CHECK_EQ(mortonEncode(0x1fffff, 0x1fffff, 0x1fffff), 0x7fffffffffffffffull);
  Aabb box;
  box.extend(make_f3(0, 0, 0));
  box.extend(make_f3(1, 1, 1));
  // Monotone along each axis.
  for (int a = 0; a < 3; ++a) {
    uint64_t prev = 0;
    for (int i = 0; i <= 100; ++i) {
      float c[3] = {0.3f, 0.3f, 0.3f};
      c[a] = float(i) / 100.f;
      const uint64_t code = mortonCode(make_f3(c[0], c[1], c[2]), box);
      CHECK(code >= prev);
      prev = code;
    }
  }
}

TEST(cpu, generators_deterministic_and_bounded) {
  for (const char* spec : {"uniform:20K", "clustered:20K:8", "scene:20K"}) {
    const PointSet a = makeDataset(spec, 7), b = makeDataset(spec, 7), c = makeDataset(spec, 8);
    CHECK_EQ(a.size(), size_t(20000));
    CHECK(std::equal(a.points.begin(), a.points.end(), b.points.begin(),
                     [](float3 x, float3 y) { return x.x == y.x && x.y == y.y && x.z == y.z; }));
    CHECK(!(a.points[0].x == c.points[0].x && a.points[0].y == c.points[0].y));
    const Aabb box = a.bounds();
    CHECK(box.lo.x >= 0.f && box.lo.y >= 0.f && box.lo.z >= 0.f);
    CHECK(box.hi.x <= 1.f && box.hi.y <= 1.f && box.hi.z <= 1.f);
  }
}

TEST(cpu, grid_matches_brute_force) {
  for (const char* spec : {"uniform:6K", "clustered:6K:4", "scene:6K"}) {
    const PointSet P = makeDataset(spec, 3);
    const PointSet Q = makeDataset("uniform:2K", 11);  // includes queries far from the data
    for (uint32_t K : {0u, 8u}) {
      const SearchParams sp{radiusForExpectedNeighbors(P.bounds(), P.size(), 12), K};
      SearchResult brute, grid;
      cpuBruteForce(P.points, Q.points, sp, brute);
      cpuGridSearch(P.points, Q.points, sp, grid);
      const VerifyReport r = verifyAgainst(P.points, Q.points, sp, brute, grid);
      CHECK_MSG(r.ok(), spec << " K=" << K << ": " << r.firstError);
    }
  }
}

TEST(cpu, grid_exact_on_lattice_ties) {
  // Integer lattice with r equal to the spacing: many pairs sit exactly at
  // distance r, so any inconsistency in the <= test or the cell filter shows up.
  const PointSet P = test::lattice(12, 0.125f);
  const SearchParams sp{0.125f, 0};
  SearchResult brute, grid;
  cpuBruteForce(P.points, P.points, sp, brute);
  cpuGridSearch(P.points, P.points, sp, grid);
  CHECK(verifyAgainst(P.points, P.points, sp, brute, grid).ok());
  CHECK_EQ(brute.counts[test::latticeIndex(12, 5, 5, 5)], 7u);  // self + 6 axis neighbors
}

TEST(cpu, kdtree_partition_and_routing) {
  const PointSet P = makeDataset("clustered:50K:6", 5);
  std::vector<float4> sorted(P.size());
  packPoints(P.points.data(), nullptr, P.size(), sorted.data());
  const KdTree tree = KdTree::build(sorted, 512);
  CHECK(tree.maxDepth() >= 5);
  // The permutation keeps every original id exactly once.
  std::vector<uint32_t> ids(sorted.size());
  for (size_t i = 0; i < sorted.size(); ++i) ids[i] = test::idOf(sorted[i]);
  std::sort(ids.begin(), ids.end());
  for (size_t i = 0; i < ids.size(); ++i) CHECK_EQ(ids[i], uint32_t(i));

  const double R = 0.02;
  for (int depth : {0, 2, 5, tree.maxDepth()}) {
    const auto frontier = tree.frontier(depth);
    const auto map = tree.tileMap(frontier);
    size_t covered = 0;
    for (size_t t = 0; t < frontier.size(); ++t) {
      const KdNode& n = tree.nodes()[size_t(frontier[t])];
      covered += n.count();
      for (uint32_t i = n.begin; i < n.end; i += 7) {
        const float3 p = xyz(sorted[i]);
        CHECK(n.cell.contains(p));
        CHECK_EQ(tree.owner(p, map), int(t));
      }
    }
    CHECK_EQ(covered, P.size());
    // within() must report exactly the tiles whose cell is within R (brute force over tiles).
    for (size_t i = 0; i < sorted.size(); i += 997) {
      const float3 p = xyz(sorted[i]);
      std::set<int> got;
      tree.within(p, R, map, [&](int t) { got.insert(t); });
      std::set<int> want;
      for (size_t t = 0; t < frontier.size(); ++t)
        if (tree.nodes()[size_t(frontier[t])].cell.dist2To(p) <= R * R) want.insert(int(t));
      CHECK(got == want);
      CHECK(got.count(tree.owner(p, map)) == 1);
    }
  }
}

TEST(cpu, kdtree_degenerate_inputs) {
  // All points identical: no split is possible, a single (oversized) leaf remains.
  std::vector<float4> same(1000, float4{0.5f, 0.5f, 0.5f, 0.f});
  for (size_t i = 0; i < same.size(); ++i) same[i].w = test::idBits(uint32_t(i));
  const KdTree t1 = KdTree::build(same, 16);
  CHECK_EQ(t1.nodes().size(), size_t(1));
  // Many ties on one axis (a plane plus a few points above it).
  std::vector<float4> plane;
  for (uint32_t i = 0; i < 4000; ++i)
    plane.push_back(float4{float(i % 64) / 64.f, float(i / 64) / 64.f, i < 3900 ? 0.f : 0.5f, test::idBits(i)});
  const KdTree t2 = KdTree::build(plane, 64);
  const auto f = t2.frontier(t2.maxDepth());
  size_t covered = 0;
  for (int id : f) covered += t2.nodes()[size_t(id)].count();
  CHECK_EQ(covered, plane.size());
}

namespace {

TileSet fakeTiles(const KdTree& tree, int depth, HaloMode mode, float R) {
  TileSet ts;
  ts.halo = mode;
  ts.depth = depth;
  ts.R = R;
  ts.frontier = tree.frontier(depth);
  ts.tileOfNode = tree.tileMap(ts.frontier);
  ts.tiles.resize(ts.frontier.size());
  for (size_t t = 0; t < ts.tiles.size(); ++t) {
    ts.tiles[t].numPoints = tree.nodes()[size_t(ts.frontier[t])].count();
    ts.tiles[t].blobBytes = 1000 + t;
  }
  return ts;
}

}  // namespace

TEST(cpu, planner_routes_every_query) {
  const PointSet P = makeDataset("uniform:40K", 2);
  const PointSet Q = makeDataset("uniform:30K", 9);
  std::vector<float4> sorted(P.size());
  packPoints(P.points.data(), nullptr, P.size(), sorted.data());
  const KdTree tree = KdTree::build(sorted, 1024);
  const float R = 0.03f;
  for (HaloMode mode : {HaloMode::Points, HaloMode::Queries}) {
    const TileSet ts = fakeTiles(tree, 4, mode, R);
    for (QueryOrder order : {QueryOrder::Original, QueryOrder::Random, QueryOrder::Morton, QueryOrder::Tiled}) {
      PlanConfig pc;
      pc.order = order;
      pc.window = 4096;
      pc.slots = 3;
      const Plan plan = makePlan(Q.points, tree, ts, pc);
      std::vector<uint32_t> seen(Q.size(), 0);
      size_t expectedPairs = 0;
      for (const auto& w : plan.windows) {
        for (const auto& st : w.steps) {
          CHECK(st.slot < 3u);
          for (uint32_t k = 0; k < st.listCount; ++k) {
            const uint32_t slot = w.contiguous ? st.listOffset + k : w.lists[st.listOffset + k];
            const uint32_t q = plan.order[w.first + slot];
            ++seen[q];
            if (mode == HaloMode::Points) CHECK_EQ(tree.owner(Q.points[q], ts.tileOfNode), int(st.tile));
          }
        }
      }
      for (size_t q = 0; q < Q.size(); ++q) {
        size_t routes = 1;
        if (mode == HaloMode::Queries) {
          routes = 0;
          tree.within(Q.points[q], R, ts.tileOfNode, [&](int) { ++routes; });
        }
        expectedPairs += routes;
        CHECK_EQ(size_t(seen[q]), routes);
      }
      CHECK_EQ(size_t(plan.pairs), expectedPairs);
    }
  }
}

TEST(cpu, cache_policies) {
  const PointSet P = makeDataset("uniform:40K", 2);
  std::vector<float4> sorted(P.size());
  packPoints(P.points.data(), nullptr, P.size(), sorted.data());
  const KdTree tree = KdTree::build(sorted, 512);
  const TileSet ts = fakeTiles(tree, 6, HaloMode::Points, 0.01f);
  PlanConfig pc;
  pc.window = 2048;
  for (QueryOrder order : {QueryOrder::Random, QueryOrder::Morton, QueryOrder::Tiled}) {
    pc.order = order;
    for (int slots : {1, 2, 4, 16}) {
      pc.slots = slots;
      pc.eviction = Eviction::LRU;
      const Plan lru = makePlan(P.points, tree, ts, pc);
      pc.eviction = Eviction::Belady;
      const Plan opt = makePlan(P.points, tree, ts, pc);
      CHECK(opt.loads <= lru.loads);   // MIN is optimal
      CHECK(opt.loads >= ts.tiles.size());  // every tile is needed at least once
      CHECK_EQ(opt.loads + opt.hits, opt.steps);
    }
    pc.slots = int(ts.tiles.size());  // everything fits: each tile loaded exactly once
    const Plan all = makePlan(P.points, tree, ts, pc);
    CHECK_EQ(size_t(all.loads), ts.tiles.size());
  }
  // Morton order needs (far) fewer loads than random order with few slots.
  pc.slots = 2;
  pc.eviction = Eviction::Belady;
  pc.order = QueryOrder::Random;
  const Plan rnd = makePlan(P.points, tree, ts, pc);
  pc.order = QueryOrder::Morton;
  const Plan mor = makePlan(P.points, tree, ts, pc);
  CHECK(mor.loads * 4 < rnd.loads);
  // Tile-major order touches every tile in one contiguous run: one load each.
  pc.order = QueryOrder::Tiled;
  const Plan tiled = makePlan(P.points, tree, ts, pc);
  CHECK_EQ(size_t(tiled.loads), ts.tiles.size());
}

TEST(cpu, kdtree_parallel_split_large_nodes) {
  // Nodes above 2^20 points take the parallel path (sampled median + stable
  // parallel partition); the result must still be a valid partition.
  const PointSet P = makeDataset("clustered:2500K:12", 21);
  std::vector<float4> sorted(P.size());
  packPoints(P.points.data(), nullptr, P.size(), sorted.data());
  const KdTree tree = KdTree::build(sorted, 1 << 16);
  std::vector<uint8_t> seen(P.size(), 0);
  for (const float4& v : sorted) seen[test::idOf(v)]++;
  CHECK(std::all_of(seen.begin(), seen.end(), [](uint8_t s) { return s == 1; }));
  const auto frontier = tree.frontier(tree.maxDepth());
  const auto map = tree.tileMap(frontier);
  uint32_t largest = 0;
  for (size_t t = 0; t < frontier.size(); ++t) {
    const KdNode& n = tree.nodes()[size_t(frontier[t])];
    largest = std::max(largest, n.count());
    for (uint32_t i = n.begin; i < n.end; i += 101) CHECK_EQ(tree.owner(xyz(sorted[i]), map), int(t));
  }
  CHECK(largest <= (1u << 16));
}
