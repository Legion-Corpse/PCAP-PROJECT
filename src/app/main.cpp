// ooc_rtnn: command-line driver for all fixed-radius neighbor search backends.
#include <omp.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>

#include "common/check.h"
#include "common/cli.h"
#include "common/dataset.h"
#include "common/search_types.h"
#include "common/timer.h"
#include "cpu/cpu_search.h"
#include "cpu/verify.h"
#include "cuda/grid_search.h"
#include "ooc/ooc_search.h"
#include "rt/incore_search.h"
#include "rt/optix_engine.h"

using namespace ooc;

namespace {

const char* kUsage = R"(ooc_rtnn - out-of-core RT-core fixed-radius neighbor search

Data
  --points SPEC          uniform:N | clustered:N[:C] | scene:N | file:PATH   (default uniform:1M)
  --queries SPEC         same | <dataset spec>                               (default same)
  --radius R             search radius
  --k-expected K         pick the radius giving ~K neighbors at uniform density (default 32)
  --max-neighbors K      also return up to K neighbor ids per query (default 0: counts only)
  --group G              RT backends: points per BVH primitive (point bucketing, default 1)
  --seed S               RNG seed (default 1)
  --save-points PATH     write the generated points as raw float32 xyz

Methods (--method, comma separated; default rt-ooc)
  brute | cpu-grid | cuda-grid | rt-incore | rt-ooc

Out-of-core index
  --budget-mb MB         device memory the out-of-core search may use (default: all free)
  --tile-points N        max owned points per tile (default: cost model)
  --depth D              explicit kd-tree cut depth (overrides --tile-points)
  --leaf-points N        finest kd-tree leaf (default 16K)
  --halo MODE            points | queries | auto (default points)
  --window N             queries per window (default: from budget)
  --print-estimates      print the cost-model table

Out-of-core runs (comma lists run every combination on the same tiles)
  --order LIST           tiled,morton,random,original (default tiled)
  --eviction LIST        belady,lru (default belady)
  --slots LIST           tile slots, 0 = as many as fit (default 0)
  --overlap LIST         on,off: copy/compute overlap (default on)
  --no-profile           skip per-operation CUDA events
  --host-order           compute the tiled query order on the CPU instead of the GPU

Output
  --verify MODE          none | sample | full (default sample)
  --verify-samples N     queries checked in sample mode (default 20000)
  --repeat N             repeat every run N times (default 1)
  --json PATH            append one JSON object per run
  --validation           enable OptiX validation mode
)";

struct Context {
  const Args& args;
  const PointSet& P;
  const PointSet& Q;
  SearchParams params;
  std::string verifyMode;
  size_t verifySamples;
  uint64_t seed;
  std::unique_ptr<std::ofstream> json;
  std::string device = "cpu";
  std::unique_ptr<SearchResult> reference;
  std::vector<size_t> sampleSubset;    // sampled verification: shared by all runs
  std::vector<uint32_t> sampleCounts;
  int failures = 0;
};

void baseMetrics(Context& c, Metrics& m, const std::string& method) {
  m.set("method", method);
  m.set("points_spec", c.args.str("points", "uniform:1M"));
  m.set("queries_spec", c.args.str("queries", "same"));
  m.set("device", c.device);
  m.set("n_points", double(c.P.size()));
  m.set("n_queries", double(c.Q.size()));
  m.set("radius", c.params.radius);
  m.set("max_neighbors", double(c.params.maxNeighbors));
  m.set("group", double(c.params.pointsPerPrim));
  m.set("omp_threads", double(omp_get_max_threads()));
}

const SearchResult& reference(Context& c) {
  if (!c.reference) {
    std::printf("  computing CPU reference for full verification...\n");
    c.reference = std::make_unique<SearchResult>();
    cpuGridSearch(c.P.points, c.Q.points, c.params, *c.reference);
  }
  return *c.reference;
}

void finish(Context& c, Metrics& m, const SearchResult& out) {
  uint64_t total = 0;
  uint32_t maxCount = 0;
  size_t truncated = 0;
  for (uint32_t v : out.counts) {
    total += v;
    maxCount = std::max(maxCount, v);
    if (out.K && v > out.K) ++truncated;
  }
  m.set("neighbors_total", double(total));
  m.set("neighbors_avg", out.counts.empty() ? 0.0 : double(total) / double(out.counts.size()));
  m.set("neighbors_max", double(maxCount));
  m.set("truncated_queries", double(truncated));
  const double searchMs = m.get("search_ms");
  m.set("mqueries_per_s", searchMs > 0 ? double(c.Q.size()) / (searchMs * 1e3) : 0.0);

  VerifyReport v;
  bool verified = false;
  if (c.verifyMode == "full") {
    v = verifyAgainst(c.P.points, c.Q.points, c.params, reference(c), out);
    verified = true;
  } else if (c.verifyMode == "sample") {
    if (c.sampleSubset.empty()) {
      c.sampleSubset = sampleQueries(c.Q.size(), c.verifySamples, c.seed + 99);
      c.sampleCounts = cpuGridCounts(c.P.points, c.Q.points, c.sampleSubset, c.params.radius);
    }
    v = verifySampled(c.P.points, c.Q.points, c.params, out, c.sampleSubset, c.sampleCounts);
    verified = true;
  }
  m.set("verify", verified ? c.verifyMode : "none");
  m.set("verify_checked", double(v.checked));
  m.set("verify_count_mismatches", double(v.countMismatches));
  m.set("verify_list_errors", double(v.listErrors));
  m.set("status", verified && !v.ok() ? "mismatch" : "ok");
  if (verified && !v.ok()) ++c.failures;

  std::printf("[%-9s] search %9.2f ms  %7.2f Mq/s  index %8.2f ms  avg|N| %6.2f", m.str("method").c_str(), searchMs, m.get("mqueries_per_s"), m.get("index_ms"), m.get("neighbors_avg"));
  if (verified)
    std::printf("  verify %s (%zu)", v.ok() ? "OK" : "FAILED", v.checked);
  std::printf("\n");
  if (verified && !v.ok()) std::printf("    first error: %s\n", v.firstError.c_str());
  if (c.json) *c.json << m.toJson() << "\n" << std::flush;
}

void failRun(Context& c, Metrics& m, const std::string& status, const std::string& what) {
  m.set("status", status);
  m.set("error", what);
  std::printf("  -> %s: %s\n", status.c_str(), what.c_str());
  if (c.json) *c.json << m.toJson() << "\n" << std::flush;
}

void printOocIndex(const OocSearch& s, const Metrics& m) {
  std::printf("  index: %zu tiles (depth %d, halo %s %.2f%%), max tile %.2f MB, store %.1f MB, %d slots, window %zu\n",
              s.tiles().tiles.size(), s.tiles().depth, s.tiles().halo == HaloMode::Points ? "points" : "queries",
              100.0 * m.get("halo_fraction"), double(s.tiles().maxBlobBytes) / 1048576.0,
              double(s.tiles().totalBlobBytes) / 1048576.0, s.slots(), s.window());
  std::printf("  index: partition %.1f ms, calibrate %.1f ms, halo %.1f ms, tile build %.1f ms; H2D %.2f GB/s\n",
              m.get("partition_ms"), m.get("calibrate_ms"), m.get("halo_ms"), m.get("tile_build_ms"),
              m.get("cal_h2d_gbps"));
}

void printEstimates(const OocSearch& s) {
  std::printf("  %5s %7s %7s %10s %10s %6s %6s %12s %6s\n", "depth", "halo", "tiles", "maxTileMB", "totalMB", "dup",
              "slots", "predict_ms", "eq1");
  for (const auto& e : s.estimates())
    std::printf("  %5d %7s %7zu %10.2f %10.1f %6.3f %6d %12.2f %6.2f%s\n", e.depth,
                e.halo == HaloMode::Points ? "points" : "queries", e.tiles, e.maxTileBytes / 1048576.0,
                e.totalBytes / 1048576.0, e.dupFactor, e.slots, e.predictedMs, e.eq1Fraction,
                e.feasible ? "" : "  (infeasible)");
}

void runOoc(Context& c, OptixEngine& engine) {
  const Args& a = c.args;
  OocOptions oo;
  const std::string halo = a.str("halo", "points");
  oo.haloAuto = halo == "auto";
  oo.halo = oo.haloAuto ? HaloMode::Points : parseHaloMode(halo);
  oo.tilePoints = uint32_t(a.integer("tile-points", 0));
  oo.depth = int(a.integer("depth", -1));
  oo.leafPoints = uint32_t(a.integer("leaf-points", 1 << 14));
  oo.budgetBytes = size_t(a.num("budget-mb", 0) * 1048576.0);
  oo.windowQueries = size_t(a.integer("window", 0));
  oo.seed = c.seed;
  const auto orders = a.list("order", "tiled");
  const auto evictions = a.list("eviction", "belady");
  const auto slotsList = a.list("slots", "0");
  const auto overlaps = a.list("overlap", "on");
  const bool profile = !a.has("no-profile");
  const int repeat = int(a.integer("repeat", 1));
  const bool estimates = a.has("print-estimates");

  Metrics im;
  baseMetrics(c, im, "rt-ooc");
  std::unique_ptr<OocSearch> search;
  try {
    search = std::make_unique<OocSearch>(engine, c.P.points, c.Q.points, c.params, oo, im);
  } catch (const DeviceOutOfMemory& e) {
    failRun(c, im, "oom", e.what());
    return;
  } catch (const std::exception& e) {
    failRun(c, im, "error", e.what());
    return;
  }
  printOocIndex(*search, im);
  if (estimates) printEstimates(*search);
  for (const auto& order : orders)
    for (const auto& ev : evictions)
      for (const auto& sl : slotsList)
        for (const auto& ov : overlaps)
          for (int r = 0; r < repeat; ++r) {
            Metrics m = im;
            RunOptions ro;
            ro.order = parseQueryOrder(order);
            ro.eviction = parseEviction(ev);
            ro.slots = std::stoi(sl);
            ro.overlap = ov != "off";
            ro.profile = profile;
            ro.deviceOrder = !a.has("host-order");
            m.set("repeat", double(r));
            SearchResult out;
            try {
              search->run(c.Q.points, ro, out, m);
            } catch (const std::exception& e) {
              failRun(c, m, dynamic_cast<const DeviceOutOfMemory*>(&e) ? "oom" : "error", e.what());
              cudaGetLastError();
              continue;
            }
            m.set("search_ms", m.get("run_ms"));
            std::printf("  run: order=%s eviction=%s slots=%d overlap=%s | loads %.0f (%.2f/tile) hit %.1f%% | "
                        "H2D %.1f MB | traverse %.1f ms, tile-H2D %.1f ms, hidden %.0f%% | predicted %.1f ms\n",
                        order.c_str(), ev.c_str(), int(m.get("slots")), ov.c_str(), m.get("tile_loads"),
                        m.get("loads_per_tile"), 100.0 * m.get("tile_hit_rate"), m.get("h2d_bytes") / 1048576.0,
                        m.get("traverse_ms"), m.get("tile_h2d_ms"), 100.0 * m.get("transfer_hidden_fraction"),
                        m.get("predicted_ms"));
            finish(c, m, out);
          }
}

}  // namespace

int main(int argc, char** argv) {
  // Load every CUDA module at context creation so the first call of a kernel
  // (CUB sort, grid query, ...) is not charged with lazy module loading.
  if (!std::getenv("CUDA_MODULE_LOADING")) {
#ifdef _WIN32
    _putenv_s("CUDA_MODULE_LOADING", "EAGER");
#else
    setenv("CUDA_MODULE_LOADING", "EAGER", 0);
#endif
  }
  try {
    const Args args(argc, argv);
    if (args.has("help")) {
      std::fputs(kUsage, stdout);
      return 0;
    }
    const uint64_t seed = uint64_t(args.integer("seed", 1));
    Timer t;
    const PointSet P = makeDataset(args.str("points", "uniform:1M"), seed);
    const PointSet Q = makeQueries(args.str("queries", "same"), P, seed);
    const double genMs = t.ms();
    if (args.has("save-points")) savePointsBin(P, args.str("save-points"));
    const Aabb box = P.bounds();
    const float radius = args.has("radius") ? float(args.num("radius", 0))
                                            : radiusForExpectedNeighbors(box, P.size(), args.num("k-expected", 32));

    Context c{args, P, Q,
              SearchParams{radius, uint32_t(args.integer("max-neighbors", 0)), uint32_t(args.integer("group", 1))},
              args.str("verify", "sample"), size_t(args.integer("verify-samples", 20000)), seed, nullptr};
    if (args.has("json")) c.json = std::make_unique<std::ofstream>(args.str("json"), std::ios::app);
    const auto methods = args.list("method", "rt-ooc");
    const int repeat = int(args.integer("repeat", 1));
    const bool validation = args.has("validation");

    std::unique_ptr<OptixEngine> engine;
    for (const auto& m : methods)
      if (m == "rt-incore" || m == "rt-ooc" || m == "cuda-grid") {
        engine = std::make_unique<OptixEngine>(validation);
        c.device = engine->deviceName();
        break;
      }
    // Touch remaining options so typos are reported before the (long) runs.
    for (const char* k : {"halo", "tile-points", "depth", "leaf-points", "budget-mb", "window", "order", "eviction",
                          "slots", "overlap", "no-profile", "print-estimates", "host-order"})
      (void)args.str(k);
    for (const auto& k : args.unused()) std::fprintf(stderr, "warning: unknown option --%s\n", k.c_str());

    std::printf("ooc_rtnn | device: %s | points %zu (%s) | queries %zu | r = %.6g | K = %u | generated in %.0f ms\n",
                c.device.c_str(), P.size(), P.name.c_str(), Q.size(), radius, c.params.maxNeighbors, genMs);

    for (const auto& method : methods) {
      std::printf("%s\n", method.c_str());
      if (method == "rt-ooc") {
        runOoc(c, *engine);
        continue;
      }
      for (int r = 0; r < repeat; ++r) {
        Metrics m;
        baseMetrics(c, m, method);
        m.set("repeat", double(r));
        SearchResult out;
        try {
          if (method == "brute") {
            Timer bt;
            cpuBruteForce(P.points, Q.points, c.params, out);
            m.set("search_ms", bt.ms());
          } else if (method == "cpu-grid") {
            cpuGridSearch(P.points, Q.points, c.params, out, &m);
            m.set("index_ms", m.get("build_ms"));
            m.set("search_ms", m.get("query_ms"));
          } else if (method == "cuda-grid") {
            cudaGridSearch(P.points, Q.points, c.params, out, m, size_t(args.integer("window", 0)));
          } else if (method == "rt-incore") {
            InCoreOptions io;
            io.order = parseQueryOrder(args.list("order", "morton").front());
            rtInCoreSearch(*engine, P.points, Q.points, c.params, io, out, m);
          } else {
            throw std::invalid_argument("unknown method: " + method);
          }
        } catch (const DeviceOutOfMemory& e) {
          failRun(c, m, "oom", e.what());
          break;
        } catch (const std::invalid_argument&) {
          throw;
        } catch (const std::exception& e) {
          // e.g. OptiX reporting an out-of-memory build: keep going with the other methods
          const std::string what = e.what();
          failRun(c, m, what.find("MEMORY") != std::string::npos || what.find("memory") != std::string::npos
                            ? "oom" : "error", what);
          cudaGetLastError();
          break;
        }
        finish(c, m, out);
      }
    }
    return c.failures ? 1 : 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
}
