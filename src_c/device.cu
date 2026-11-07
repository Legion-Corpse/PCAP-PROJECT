#include <optix.h>
#include "rt_params.h"
extern "C" {
__constant__ RtParams params;
}

extern "C" __global__ void __raygen__frnn(void) {
    const uint32_t i = optixGetLaunchIndex().x;
    const uint32_t slot = params.query_list ? params.query_list[i] : i;
    const Point q = params.queries[slot];
    uint32_t p0 = slot, p1 = params.counts[slot];
    optixTrace(params.handle, make_float3(q.x, q.y, q.z), make_float3(1.f, 0.f, 0.f), 0.f, 1e-16f,
               0.f, 255, OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, 0, 1, 0,
               p0, p1);
    params.counts[slot] = p1;
}
extern "C" __global__ void __intersection__frnn(void) {
    const Point *p = params.points + size_t(optixGetPrimitiveIndex()) * params.group;
    const float3 q = optixGetWorldRayOrigin();
    const uint32_t old_count = optixGetPayload_1();
    uint32_t count = old_count;
    for (uint32_t j = 0; j < params.group; ++j) {
        const Point v = p[j];
        // explicit rounding matches the host predicate in common.c
        const float dx = __fsub_rn(q.x, v.x), dy = __fsub_rn(q.y, v.y), dz = __fsub_rn(q.z, v.z);
        const float d2 =
            __fadd_rn(__fadd_rn(__fmul_rn(dx, dx), __fmul_rn(dy, dy)), __fmul_rn(dz, dz));
        if (d2 <= params.radius2) {
            if (count < params.max_neighbors)
                params.neighbors[size_t(optixGetPayload_0()) * params.max_neighbors + count] = v.id;
            ++count;
        }
    }
    if (count != old_count)
        optixSetPayload_1(count);
}
extern "C" __global__ void __miss__frnn(void) {}
