// Point-set generation and I/O.
//
// Dataset specs (case-insensitive kinds, sizes accept K/M/G suffixes):
//   uniform:N            N points uniform in the unit cube
//   clustered:N[:C]      Gaussian mixture of C clusters (default 64) + 5% background noise
//   scene:N              scanned-scene surrogate: ground plane, boxes, spheres (2-manifold data)
//   file:PATH            .bin (float32 xyz), .kitti (float32 xyzi), .ply, .xyz/.txt/.csv
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "common/vec_math.h"

namespace ooc {

struct PointSet {
  std::string name;
  std::vector<float3> points;

  size_t size() const { return points.size(); }
  Aabb bounds() const;
};

Aabb computeBounds(const std::vector<float3>& points);  // OpenMP reduction
uint64_t parseCount(const std::string& text);           // "10M" -> 10000000
PointSet makeDataset(const std::string& spec, uint64_t seed);
PointSet loadPointFile(const std::string& path);
void savePointsBin(const PointSet& ps, const std::string& path);

// Query sets: "same" (queries = points) or any dataset spec.
PointSet makeQueries(const std::string& spec, const PointSet& points, uint64_t seed);

// Radius at which a uniform cloud of n points filling `box` has `k` expected neighbors.
float radiusForExpectedNeighbors(const Aabb& box, size_t n, double k);

// Pads a search radius so that every pair accepted by the float distance test
// is guaranteed to be found by a candidate filter (grid cell, AABB, kd halo)
// evaluated in rounded arithmetic. See docs/DESIGN.md, "Exactness".
float conservativeRadius(float r, const Aabb& box);

}  // namespace ooc
