// Small vector / box helpers shared by host code, CUDA kernels and OptiX programs.
#pragma once

#include <vector_types.h>
#include <cfloat>
#include <cstdint>

#if defined(__CUDACC__)
#define OOC_HD __host__ __device__ __forceinline__
#else
#define OOC_HD inline
#endif

namespace ooc {

OOC_HD float3 make_f3(float x, float y, float z) { return float3{x, y, z}; }
OOC_HD float3 operator+(float3 a, float3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
OOC_HD float3 operator-(float3 a, float3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
OOC_HD float3 operator*(float3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
OOC_HD float3 fminf3(float3 a, float3 b) { return {a.x < b.x ? a.x : b.x, a.y < b.y ? a.y : b.y, a.z < b.z ? a.z : b.z}; }
OOC_HD float3 fmaxf3(float3 a, float3 b) { return {a.x > b.x ? a.x : b.x, a.y > b.y ? a.y : b.y, a.z > b.z ? a.z : b.z}; }
OOC_HD float comp(float3 v, int axis) { return axis == 0 ? v.x : (axis == 1 ? v.y : v.z); }
OOC_HD float3 xyz(float4 v) { return {v.x, v.y, v.z}; }

// Squared Euclidean distance with every operation rounded separately.
//
// nvcc contracts a*b+c into a fused multiply-add by default, while MSVC with
// /fp:precise does not. A pair at distance ~r could then be a neighbor on one
// backend and not on another. The *_rn intrinsics are never contracted, so the
// CPU grid, the CUDA grid and the RT-core programs all agree bit-for-bit and
// can be cross-checked with exact equality.
OOC_HD float dist2(float3 a, float3 b) {
#if defined(__CUDA_ARCH__)
  const float dx = __fsub_rn(a.x, b.x), dy = __fsub_rn(a.y, b.y), dz = __fsub_rn(a.z, b.z);
  return __fadd_rn(__fadd_rn(__fmul_rn(dx, dx), __fmul_rn(dy, dy)), __fmul_rn(dz, dz));
#else
  const float dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
  const float xx = dx * dx, yy = dy * dy, zz = dz * dz;
  return (xx + yy) + zz;
#endif
}

struct Aabb {
  float3 lo{FLT_MAX, FLT_MAX, FLT_MAX};
  float3 hi{-FLT_MAX, -FLT_MAX, -FLT_MAX};

  OOC_HD void extend(float3 p) { lo = fminf3(lo, p); hi = fmaxf3(hi, p); }
  OOC_HD void extend(const Aabb& b) { lo = fminf3(lo, b.lo); hi = fmaxf3(hi, b.hi); }
  OOC_HD bool valid() const { return lo.x <= hi.x && lo.y <= hi.y && lo.z <= hi.z; }
  OOC_HD float3 extent() const { return hi - lo; }
  OOC_HD bool contains(float3 p) const {
    return p.x >= lo.x && p.x <= hi.x && p.y >= lo.y && p.y <= hi.y && p.z >= lo.z && p.z <= hi.z;
  }
  // Squared distance from p to the box (0 inside). Infinite faces are allowed.
  OOC_HD double dist2To(float3 p) const {
    double d = 0.0;
    const float pc[3] = {p.x, p.y, p.z};
    const float lc[3] = {lo.x, lo.y, lo.z};
    const float hc[3] = {hi.x, hi.y, hi.z};
    for (int a = 0; a < 3; ++a) {
      double g = 0.0;
      if (pc[a] < lc[a]) g = double(lc[a]) - double(pc[a]);
      else if (pc[a] > hc[a]) g = double(pc[a]) - double(hc[a]);
      d += g * g;
    }
    return d;
  }
  OOC_HD int longestAxis() const {
    const float3 e = extent();
    return (e.x >= e.y && e.x >= e.z) ? 0 : (e.y >= e.z ? 1 : 2);
  }
};

}  // namespace ooc
