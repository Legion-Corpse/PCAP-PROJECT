// Device tests: every GPU backend must agree exactly with the CPU reference.
#include <memory>

#include "common/dataset.h"
#include "cpu/cpu_search.h"
#include "cpu/verify.h"
#include "cuda/grid_search.h"
#include "ooc/ooc_search.h"
#include "rt/incore_search.h"
#include "rt/optix_engine.h"
#include "test_framework.h"
#include "test_util.h"

using namespace ooc;

namespace {

OptixEngine& engine() {
  static std::unique_ptr<OptixEngine> e = std::make_unique<OptixEngine>();
  return *e;
}

struct Case {
  const char* name;
  PointSet P, Q;
  SearchParams sp;
  SearchResult ref;
};

Case makeCase(const char* name, PointSet P, PointSet Q, float r, uint32_t K) {
  Case c{name, std::move(P), std::move(Q), SearchParams{r, K}, {}};
  cpuGridSearch(c.P.points, c.Q.points, c.sp, c.ref);
  return c;
}

std::vector<Case>& cases() {
  static std::vector<Case> cs = [] {
    std::vector<Case> v;
    PointSet u = makeDataset("uniform:60K", 1);
    const float ru = radiusForExpectedNeighbors(u.bounds(), u.size(), 24);
    v.push_back(makeCase("uniform_same", u, makeQueries("same", u, 1), ru, 0));
    v.push_back(makeCase("uniform_list", u, makeQueries("same", u, 1), ru, 16));
    PointSet cl = makeDataset("clustered:60K:5", 2);
    v.push_back(makeCase("clustered_other_queries", cl, makeDataset("clustered:20K:5", 3), 0.01f, 32));
    PointSet sc = makeDataset("scene:60K", 4);
    v.push_back(makeCase("scene", sc, makeQueries("same", sc, 4), 0.004f, 8));
    PointSet lat = test::lattice(24, 0.25f);  // exact-distance ties and split-plane ties
    v.push_back(makeCase("lattice_ties", lat, makeQueries("same", lat, 1), 0.25f, 8));
    PointSet far = makeDataset("uniform:5K", 9);
    for (auto& p : far.points) p = make_f3(p.x * 3 + 5, p.y * 3 - 7, p.z);  // queries away from the data
    v.push_back(makeCase("far_queries", u, far, ru, 4));
    return v;
  }();
  return cs;
}

void expectMatches(const Case& c, const SearchResult& got, const std::string& what) {
  const VerifyReport r = verifyAgainst(c.P.points, c.Q.points, c.sp, c.ref, got);
  CHECK_MSG(r.ok(), c.name << " / " << what << ": " << r.countMismatches << " count, " << r.listErrors
                           << " list errors; " << r.firstError);
}

}  // namespace

TEST(gpu, cuda_grid_matches_cpu) {
  for (const auto& c : cases()) {
    SearchResult out;
    Metrics m;
    cudaGridSearch(c.P.points, c.Q.points, c.sp, out, m, 7000);  // small window: exercises windowing
    expectMatches(c, out, "cuda-grid");
  }
}

TEST(gpu, rt_incore_matches_cpu) {
  for (const auto& c : cases()) {
    for (QueryOrder order : {QueryOrder::Morton, QueryOrder::Original})
      for (uint32_t group : {1u, 3u, 8u}) {  // 3: partial last group exercises NaN padding
        SearchResult out;
        Metrics m;
        InCoreOptions io;
        io.order = order;
        io.windowQueries = 9000;
        SearchParams sp = c.sp;
        sp.pointsPerPrim = group;
        rtInCoreSearch(engine(), c.P.points, c.Q.points, sp, io, out, m);
        expectMatches(c, out, std::string("rt-incore ") + toString(order) + " group=" + std::to_string(group));
      }
  }
}

TEST(gpu, rt_ooc_matches_cpu_all_configurations) {
  for (const auto& c : cases()) {
    for (HaloMode halo : {HaloMode::Points, HaloMode::Queries}) {
      OocOptions oo;
      oo.halo = halo;
      oo.leafPoints = 1024;
      oo.depth = 4;  // 16 tiles
      oo.windowQueries = 5000;
      SearchParams sp = c.sp;
      sp.pointsPerPrim = halo == HaloMode::Points ? 1 : 5;
      Metrics im;
      OocSearch s(engine(), c.P.points, c.Q.points, sp, oo, im);
      for (QueryOrder order : {QueryOrder::Tiled, QueryOrder::Morton, QueryOrder::Random, QueryOrder::Original})
        for (Eviction ev : {Eviction::Belady, Eviction::LRU})
          for (int slots : {1, 2, 3})
            for (bool overlap : {true, false}) {
              if (order != QueryOrder::Tiled && (slots == 3 || !overlap)) continue;  // keep runtime modest
              RunOptions ro;
              ro.order = order;
              ro.eviction = ev;
              ro.slots = slots;
              ro.overlap = overlap;
              for (bool deviceOrder : {true, false}) {
                if (!deviceOrder && (order != QueryOrder::Tiled || slots != 2)) continue;
                ro.deviceOrder = deviceOrder;
                SearchResult out;
                Metrics m = im;
                s.run(c.Q.points, ro, out, m);
                expectMatches(c, out,
                              std::string("ooc halo=") + toString(halo) + " order=" + toString(order) + " ev=" +
                                  toString(ev) + " slots=" + std::to_string(slots) +
                                  (overlap ? " overlap" : " serial") + (deviceOrder ? "" : " host-order"));
              }
            }
    }
  }
}

TEST(gpu, rt_ooc_cost_model_and_budget) {
  const Case& c = cases()[0];
  for (double budgetMb : {8.0, 32.0, 0.0}) {
    OocOptions oo;
    oo.leafPoints = 1024;
    oo.haloAuto = true;
    oo.budgetBytes = size_t(budgetMb * 1048576.0);
    Metrics im;
    OocSearch s(engine(), c.P.points, c.Q.points, c.sp, oo, im);
    CHECK(s.slots() >= 1);
    SearchResult out;
    Metrics m = im;
    s.run(c.Q.points, RunOptions{}, out, m);
    if (budgetMb > 0) CHECK_MSG(m.get("device_bytes_used") <= budgetMb * 1048576.0, m.get("device_bytes_used"));
    expectMatches(c, out, "cost-model budget " + std::to_string(budgetMb));
  }
}

TEST(gpu, rt_tiny_inputs) {
  PointSet one;
  one.points = {make_f3(0.5f, 0.5f, 0.5f)};
  PointSet q;
  q.points = {make_f3(0.5f, 0.5f, 0.5f), make_f3(0.9f, 0.9f, 0.9f)};
  const Case c = makeCase("tiny", one, q, 0.1f, 2);
  SearchResult a, b;
  Metrics m;
  rtInCoreSearch(engine(), c.P.points, c.Q.points, c.sp, InCoreOptions{}, a, m);
  expectMatches(c, a, "rt-incore");
  OocOptions oo;
  OocSearch s(engine(), c.P.points, c.Q.points, c.sp, oo, m);
  s.run(c.Q.points, RunOptions{}, b, m);
  expectMatches(c, b, "rt-ooc");
  CHECK_EQ(b.counts[0], 1u);
  CHECK_EQ(b.counts[1], 0u);
}
