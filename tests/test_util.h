// Shared helpers for the test executables.
#pragma once

#include <cstring>

#include "common/dataset.h"

namespace test {

inline float idBits(uint32_t id) {
  float f;
  std::memcpy(&f, &id, sizeof(f));
  return f;
}

inline uint32_t idOf(const float4& v) {
  uint32_t id;
  std::memcpy(&id, &v.w, sizeof(id));
  return id;
}

inline size_t latticeIndex(int n, int x, int y, int z) { return (size_t(z) * n + y) * n + x; }

// n^3 points on a cubic lattice with the given spacing.
inline ooc::PointSet lattice(int n, float spacing) {
  ooc::PointSet ps;
  ps.name = "lattice";
  for (int z = 0; z < n; ++z)
    for (int y = 0; y < n; ++y)
      for (int x = 0; x < n; ++x) ps.points.push_back(ooc::make_f3(x * spacing, y * spacing, z * spacing));
  return ps;
}

}  // namespace test
