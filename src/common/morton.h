// 63-bit 3D Morton (Z-order) codes, 21 bits per axis.
#pragma once

#include "common/vec_math.h"

namespace ooc {

// Spreads the low 21 bits of v so that bit i moves to bit 3i.
OOC_HD uint64_t mortonSpread21(uint64_t v) {
  v &= 0x1fffffULL;
  v = (v | (v << 32)) & 0x1f00000000ffffULL;
  v = (v | (v << 16)) & 0x1f0000ff0000ffULL;
  v = (v | (v << 8)) & 0x100f00f00f00f00fULL;
  v = (v | (v << 4)) & 0x10c30c30c30c30c3ULL;
  v = (v | (v << 2)) & 0x1249249249249249ULL;
  return v;
}

OOC_HD uint64_t mortonEncode(uint32_t x, uint32_t y, uint32_t z) {
  return mortonSpread21(x) | (mortonSpread21(y) << 1) | (mortonSpread21(z) << 2);
}

// Quantizes p inside `box` to a 2^21 lattice and returns its Morton code.
OOC_HD uint64_t mortonCode(float3 p, const Aabb& box) {
  const float3 e = box.extent();
  const float scale = float((1u << 21) - 1);
  auto q = [&](float v, float lo, float ext) -> uint32_t {
    if (!(ext > 0.f)) return 0u;
    float t = (v - lo) / ext;
    t = t < 0.f ? 0.f : (t > 1.f ? 1.f : t);
    return uint32_t(t * scale);
  };
  return mortonEncode(q(p.x, box.lo.x, e.x), q(p.y, box.lo.y, e.y), q(p.z, box.lo.z, e.z));
}

}  // namespace ooc
