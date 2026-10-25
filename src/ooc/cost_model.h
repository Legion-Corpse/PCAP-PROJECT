// Transfer/traversal cost model used to pick the tile size (tree depth).
//
// For tile t with compacted size S_t and |Q_t| routed queries, double
// buffering hides the transfer of the next tile when (synopsis Eq. 1)
//
//     |Q_t| * tau(n_t)  >=  S_t / B_h2d
//
// and the pipelined run time is approximately
//
//     T(d) = x_0 + sum_t max(y_t, x_{t+1}),
//     x_t = S_t / B_h2d + c_reloc,   y_t = c_launch * steps_t + |Q_t| * tau(n_t),
//
// where tau(n) (traversal time per query in a tile of n points) is calibrated
// on the device with probe tiles of two sizes and interpolated in log2(n).
// Smaller tiles mean more halo (or query) duplication and more launches;
// larger tiles mean fewer cache slots and a longer un-hidden first transfer.
#pragma once

#include <string>
#include <vector>

#include "common/search_types.h"
#include "ooc/kdtree.h"
#include "ooc/tile_store.h"

namespace ooc {

class OptixEngine;

struct Calibration {
  double h2dGBps = 10.0;
  double launchUs = 10.0;
  double relocUs = 5.0;
  double tauNs0 = 50.0, tauNs1 = 60.0;  // per-query traversal time at n0 and n1 points
  double n0 = 1 << 16, n1 = 1 << 19;
  double gasBytesPerPrim = 64.0;
  double probeNeighbors = 0.0;  // mean |N(q)| seen by the probes

  double tauNs(double n) const;
};

struct DepthEstimate {
  int depth = 0;
  HaloMode halo = HaloMode::Points;
  size_t tiles = 0;
  double maxTileBytes = 0, totalBytes = 0;
  double dupFactor = 1;   // stored points / N (Points) or routed pairs / M (Queries)
  int slots = 0;          // tile slots that fit the budget
  double transferMs = 0, traverseMs = 0, predictedMs = 0;
  double eq1Fraction = 0; // tiles whose traversal hides the next transfer
  bool feasible = false;
};

class CostModel {
 public:
  // `sortedPoints` is the kd-permuted point array; probes use contiguous runs of it.
  static Calibration calibrate(OptixEngine& engine, const std::vector<float4>& sortedPoints, float radius, float R,
                               uint32_t K, uint32_t group);

  CostModel(const KdTree& tree, const std::vector<float3>& points, const std::vector<float3>& queries, float R,
            uint64_t seed);

  DepthEstimate estimate(int depth, HaloMode halo, const Calibration& cal, size_t budgetBytes, size_t windowBytes,
                         int slotsOverride, size_t windowQueries) const;

  // Best feasible estimate over depths (and halo modes when `haloAuto`).
  DepthEstimate choose(const Calibration& cal, HaloMode halo, bool haloAuto, size_t budgetBytes, size_t windowBytes,
                       int slotsOverride, size_t windowQueries, std::vector<DepthEstimate>* all = nullptr) const;

 private:
  const KdTree& tree_;
  double N_, M_;
  double pointScale_, queryScale_;
  std::vector<uint32_t> pointCover_, queryOwner_, queryCover_;
};

}  // namespace ooc
