#include "ooc/ooc_search.h"

#include <algorithm>
#include <stdexcept>

#include "common/dataset.h"
#include "common/host_utils.h"
#include "common/timer.h"
#include "rt/optix_engine.h"

namespace ooc {
namespace {

constexpr size_t kReserveBytes = size_t(192) << 20;  // OptiX pipeline, CUDA context, allocator slack

void recordEstimate(Metrics& m, const DepthEstimate& e) {
  m.set("predicted_ms", e.predictedMs);
  m.set("predicted_transfer_ms", e.transferMs);
  m.set("predicted_traverse_ms", e.traverseMs);
  m.set("predicted_eq1_fraction", e.eq1Fraction);
  m.set("predicted_dup_factor", e.dupFactor);
}

}  // namespace

OocSearch::OocSearch(OptixEngine& engine, const std::vector<float3>& points, const std::vector<float3>& queries,
                     const SearchParams& params, const OocOptions& opts, Metrics& m)
    : engine_(engine), params_(params), opts_(opts) {
  NvtxRange range("ooc_index");
  Timer total;
  size_t freeB = 0, totalB = 0;
  CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));
  const size_t avail = freeB > kReserveBytes ? freeB - kReserveBytes : 0;
  budget_ = opts.budgetBytes ? std::min(opts.budgetBytes, avail) : avail;
  m.set("device_budget_bytes", double(budget_));
  m.set("device_free_bytes", double(freeB));

  Timer t;
  sorted_.resize(points.size());
  packPoints(points.data(), nullptr, points.size(), sorted_.data());
  tree_ = KdTree::build(sorted_, opts.leafPoints);
  m.set("partition_ms", t.ms());
  m.set("kd_nodes", double(tree_.nodes().size()));
  m.set("kd_max_depth", double(tree_.maxDepth()));

  const Aabb box = computeBounds(points);
  const float R = conservativeRadius(params.radius, box);
  const uint32_t K = params.maxNeighbors;
  const size_t M = std::max<size_t>(1, queries.size());

  // Window: at most a quarter of the budget for double-buffered queries,
  // routing lists and results, and at most 2M queries - enough to fill the GPU
  // per tile, while keeping pinned buffers small and the host's packing of
  // window w+1 overlapped with the GPU's work on window w.
  const size_t perQuery = 2 * (16 + 4 * 2 + 4 + 4ull * K);
  window_ = opts.windowQueries ? opts.windowQueries
                               : std::clamp<size_t>(budget_ / 4 / perQuery, size_t(4096), size_t(1) << 21);
  window_ = std::min(window_, M);
  const size_t stepsBound = tree_.frontier(tree_.maxDepth()).size();
  const size_t windowBytes = windowBufferBytes(window_, 2 * window_, stepsBound, K);

  t.reset();
  const Calibration cal = CostModel::calibrate(engine, sorted_, params.radius, R, K, params.pointsPerPrim);
  const CostModel model(tree_, points, queries, R, opts.seed);
  m.set("calibrate_ms", t.ms());
  m.set("cal_h2d_gbps", cal.h2dGBps);
  m.set("cal_launch_us", cal.launchUs);
  m.set("cal_reloc_us", cal.relocUs);
  m.set("cal_tau_ns_small", cal.tauNs0);
  m.set("cal_tau_ns_large", cal.tauNs1);
  m.set("cal_gas_bytes_per_prim", cal.gasBytesPerPrim);
  m.set("cal_probe_neighbors", cal.probeNeighbors);

  int depth = 0;
  HaloMode halo = opts.halo;
  if (opts.depth >= 0 || opts.tilePoints) {
    depth = opts.depth >= 0 ? std::min(opts.depth, tree_.maxDepth()) : tree_.depthForCapacity(opts.tilePoints);
    for (int d = 0; d <= tree_.maxDepth(); ++d)
      estimates_.push_back(model.estimate(d, halo, cal, budget_, windowBytes, opts.slots, window_));
    recordEstimate(m, estimates_[size_t(depth)]);
  } else {
    const DepthEstimate best = model.choose(cal, opts.halo, opts.haloAuto, budget_, windowBytes, opts.slots, window_,
                                            &estimates_);
    depth = best.depth;
    halo = best.halo;
    recordEstimate(m, best);
  }
  m.set("halo", toString(halo));
  m.set("window_queries", double(window_));

  tiles_ = buildTiles(engine, tree_, sorted_, depth, halo, R, params.pointsPerPrim, m);
  sorted_.clear();
  sorted_.shrink_to_fit();

  const size_t T = tiles_.tiles.size();
  if (opts.slots > 0) {
    slots_ = int(std::min<size_t>(size_t(opts.slots), T));
  } else {
    const size_t availSlots = budget_ > windowBytes ? (budget_ - windowBytes) / std::max<size_t>(tiles_.maxBlobBytes, 1) : 0;
    slots_ = int(std::min(availSlots, T));
  }
  if (slots_ < 1) throw std::runtime_error("device budget cannot hold a single tile; lower the tile size");
  m.set("index_ms", total.ms());
}

void OocSearch::run(const std::vector<float3>& queries, const RunOptions& run, SearchResult& out, Metrics& m) {
  NvtxRange range("ooc_run");
  Timer total;
  PlanConfig pc;
  pc.order = run.order;
  pc.eviction = run.eviction;
  pc.window = uint32_t(window_);
  pc.slots = run.slots > 0 ? int(std::min<size_t>(size_t(run.slots), tiles_.tiles.size())) : slots_;
  pc.seed = opts_.seed;
  pc.deviceOrder = run.deviceOrder;
  pc.deviceOrderBudget = budget_;  // the tile slots are not allocated yet
  Plan plan = makePlan(queries, tree_, tiles_, pc);

  // The exact window footprint is known now; give up slots if it does not fit.
  const size_t wb = windowBufferBytes(plan.maxWindow, plan.maxPairs, plan.maxSteps, params_.maxNeighbors);
  int slots = pc.slots;
  while (slots > 1 && size_t(slots) * tiles_.maxBlobBytes + wb > budget_ && run.slots <= 0) --slots;
  if (slots != pc.slots) simulateCache(plan, tiles_, slots, pc.eviction);

  m.set("order", toString(run.order));
  m.set("eviction", toString(run.eviction));
  m.set("overlap", run.overlap ? 1.0 : 0.0);
  m.set("plan_order_ms", plan.orderMs);
  m.set("plan_owner_ms", plan.orderPhaseMs[0]);
  m.set("plan_bucket_ms", plan.orderPhaseMs[1]);
  m.set("plan_tilesort_ms", plan.orderPhaseMs[2]);
  m.set("plan_gather_ms", plan.orderPhaseMs[3]);
  m.set("plan_order_device_ms", plan.orderDeviceMs);
  m.set("plan_order_host_ms", plan.orderHostMs);
  m.set("plan_route_ms", plan.routeMs);
  m.set("plan_cache_ms", plan.cacheMs);
  m.set("windows", double(plan.windows.size()));
  m.set("steps", double(plan.steps));
  m.set("tile_loads", double(plan.loads));
  m.set("tile_hits", double(plan.hits));
  m.set("tile_hit_rate", plan.steps ? double(plan.hits) / double(plan.steps) : 0.0);
  m.set("loads_per_tile", tiles_.tiles.empty() ? 0.0 : double(plan.loads) / double(tiles_.tiles.size()));
  m.set("tile_load_bytes", double(plan.loadBytes));
  m.set("pairs", double(plan.pairs));
  m.set("pairs_per_query", queries.empty() ? 0.0 : double(plan.pairs) / double(queries.size()));
  m.set("device_bytes_used", double(size_t(slots) * tiles_.maxBlobBytes + wb));

  ExecConfig ec;
  ec.slots = slots;
  ec.overlap = run.overlap;
  ec.profile = run.profile;
  executePlan(engine_, tiles_, plan, queries, params_, ec, out, m);
  m.set("run_ms", total.ms());
}

}  // namespace ooc
