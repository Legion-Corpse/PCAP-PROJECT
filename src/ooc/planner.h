// Offline schedule for the out-of-core search.
//
// All queries are known before the search starts, so the whole access
// sequence can be planned up front:
//   1. order the queries: original, random, global Morton (Z-order), or
//      tiled = grouped by owner tile (kd DFS order) and Morton-sorted inside
//      each tile, which gives the same locality without a global sort and
//      makes the routing of the single-tile (point-halo) mode free,
//   2. cut the order into windows (bounded device memory for queries/results),
//   3. route each window's queries to tiles (counting sort, order-preserving),
//   4. visit first the touched tiles that are probably still resident (the
//      last `slots` distinct tiles used), then the rest in kd (spatial) order,
//   5. simulate the device tile cache with LRU or Belady's optimal (MIN)
//      replacement; the slot of the previous step is never the victim so that
//      the next transfer can overlap the current traversal.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "common/vec_math.h"
#include "ooc/kdtree.h"
#include "ooc/tile_store.h"

namespace ooc {

enum class QueryOrder { Original, Random, Morton, Tiled };
enum class Eviction { LRU, Belady };

struct PlanStep {
  uint32_t tile = 0;
  uint32_t slot = 0;
  uint32_t listOffset = 0;  // into WindowPlan::lists
  uint32_t listCount = 0;
  bool load = false;
};

struct WindowPlan {
  size_t first = 0;         // into Plan::order
  uint32_t count = 0;
  // Contiguous: step k serves window slots [listOffset, listOffset + listCount)
  // directly (tile-major order with single-tile routing), `lists` is empty.
  bool contiguous = false;
  std::vector<uint32_t> lists;  // window-local query slots, grouped by step
  std::vector<PlanStep> steps;
};

struct Plan {
  std::vector<uint32_t> order;      // original query indices in processing order
  std::vector<uint32_t> tileStart;  // tiled order: tile t serves order[tileStart[t], tileStart[t+1])
  // The queries gathered into processing order. Only materialized when the
  // planner itself needs coordinates (multi-tile routing); otherwise the
  // executor gathers each window while the GPU works on the previous one.
  std::vector<float3> queries;
  std::vector<WindowPlan> windows;
  uint64_t steps = 0, loads = 0, hits = 0, pairs = 0, loadBytes = 0;
  size_t maxSteps = 0, maxPairs = 0, maxWindow = 0;
  double orderMs = 0, routeMs = 0, cacheMs = 0;
  double orderPhaseMs[4] = {0, 0, 0, 0};  // host tiled order: owner, bucket, in-tile sort, gather
  double orderDeviceMs = 0, orderHostMs = 0; // device tiled order: chunks on the GPU, host concatenation
};

struct PlanConfig {
  QueryOrder order = QueryOrder::Tiled;
  Eviction eviction = Eviction::Belady;
  uint32_t window = 1u << 20;
  int slots = 2;
  uint64_t seed = 1;
  bool deviceOrder = false;       // compute the tiled order on the GPU (gpu_order.h)
  size_t deviceOrderBudget = 0;   // device bytes the GPU ordering may use
};

std::vector<uint32_t> queryOrder(const std::vector<float3>& queries, QueryOrder order, uint64_t seed);

Plan makePlan(const std::vector<float3>& queries, const KdTree& tree, const TileSet& tiles, const PlanConfig& cfg);

// Re-runs only the cache simulation (used to compare policies on one routing).
void simulateCache(Plan& plan, const TileSet& tiles, int slots, Eviction eviction);

QueryOrder parseQueryOrder(const std::string& s);
Eviction parseEviction(const std::string& s);
HaloMode parseHaloMode(const std::string& s);
const char* toString(QueryOrder o);
const char* toString(Eviction e);
const char* toString(HaloMode h);

}  // namespace ooc
