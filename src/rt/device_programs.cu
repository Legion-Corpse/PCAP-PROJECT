// OptiX programs for fixed-radius neighbor search on RT cores.
//
// Mapping (RTNN-style): every point p is a custom primitive whose AABB is the
// cube of half-width r around p. A query q becomes an (almost) zero-length ray
// starting at q, so BVH traversal on the RT core reports exactly the cubes that
// contain q. The intersection program does the exact sphere test and never
// reports a hit, so traversal continues and visits every candidate.
//
// Point bucketing: with params.group = G > 1 a primitive is a run of G
// Morton-adjacent points and its AABB is their bounding box grown by r. Each
// RT-core -> SM hand-off then tests G points instead of one, trading cheap
// distance tests for expensive intersection-program invocations.
#include <optix.h>

#include "common/vec_math.h"
#include "rt/launch_params.h"

using namespace ooc;

extern "C" {
__constant__ LaunchParams params;
}

extern "C" __global__ void __raygen__frnn() {
  const uint32_t i = optixGetLaunchIndex().x;
  const uint32_t slot = params.queryList ? params.queryList[i] : i;
  const float4 q = params.queries[slot];
  uint32_t p0 = slot;
  uint32_t p1 = params.counts[slot];  // continue counting where the previous tile stopped
  const OptixTraversableHandle handle = params.handleSlot ? *params.handleSlot : params.handle;
  optixTrace(handle, make_float3(q.x, q.y, q.z), make_float3(1.f, 0.f, 0.f), 0.f, 1e-16f, 0.f,
             OptixVisibilityMask(255), OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, 0, 1, 0, p0,
             p1);
  params.counts[slot] = p1;
}

extern "C" __global__ void __intersection__frnn() {
  const uint32_t G = params.group;
  const float4* p = params.points + size_t(optixGetPrimitiveIndex()) * G;
  const float3 o = optixGetWorldRayOrigin();
  const float r2 = params.radius2;
  const uint32_t c0 = optixGetPayload_1();
  uint32_t c = c0;
  for (uint32_t j = 0; j < G; ++j) {
    const float4 v = p[j];
    if (dist2(o, xyz(v)) <= r2) {  // false for NaN padding
      if (c < params.maxNeighbors)
        params.neighbors[size_t(optixGetPayload_0()) * params.maxNeighbors + c] = __float_as_uint(v.w);
      ++c;
    }
  }
  if (c != c0) optixSetPayload_1(c);
}

extern "C" __global__ void __miss__frnn() {}
