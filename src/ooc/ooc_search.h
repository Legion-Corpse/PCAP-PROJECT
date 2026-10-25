// Out-of-core RT-core fixed-radius neighbor search: the full pipeline.
//
//   points -> kd-tree -> cost model picks tile depth -> tiles (compacted,
//   relocatable GAS + points in pinned host memory) -> per run: plan (query
//   order, windows, routing, cache policy) -> pipelined execution.
#pragma once

#include <memory>
#include <vector>

#include "common/search_types.h"
#include "ooc/cost_model.h"
#include "ooc/executor.h"
#include "ooc/kdtree.h"
#include "ooc/planner.h"
#include "ooc/tile_store.h"

namespace ooc {

class OptixEngine;

struct OocOptions {
  // Index (tiling) options
  HaloMode halo = HaloMode::Points;
  bool haloAuto = false;
  uint32_t tilePoints = 0;  // max owned points per tile; 0 = chosen by the cost model
  int depth = -1;           // explicit kd depth (overrides tilePoints)
  uint32_t leafPoints = 1u << 14;
  size_t budgetBytes = 0;   // device memory the search may use; 0 = free memory - reserve
  size_t windowQueries = 0; // 0 = automatic
  int slots = 0;            // 0 = as many as fit in the budget
  uint64_t seed = 1;
};

struct RunOptions {
  QueryOrder order = QueryOrder::Tiled;
  Eviction eviction = Eviction::Belady;
  bool overlap = true;
  bool profile = true;
  bool deviceOrder = true;  // tiled order computed on the GPU (otherwise on the host)
  int slots = 0;            // 0 = index default
};

class OocSearch {
 public:
  // Builds the tile store. `queries` are only used to calibrate the cost model.
  OocSearch(OptixEngine& engine, const std::vector<float3>& points, const std::vector<float3>& queries,
            const SearchParams& params, const OocOptions& opts, Metrics& metrics);

  void run(const std::vector<float3>& queries, const RunOptions& run, SearchResult& out, Metrics& metrics);

  const TileSet& tiles() const { return tiles_; }
  const KdTree& tree() const { return tree_; }
  const std::vector<DepthEstimate>& estimates() const { return estimates_; }
  int slots() const { return slots_; }
  size_t window() const { return window_; }

 private:
  OptixEngine& engine_;
  SearchParams params_;
  OocOptions opts_;
  std::vector<float4> sorted_;
  KdTree tree_;
  TileSet tiles_;
  std::vector<DepthEstimate> estimates_;
  size_t budget_ = 0, window_ = 0;
  int slots_ = 0;
};

}  // namespace ooc
