#include <cuda_runtime.h>
#include "grid_kernels.h"

__device__ static int coordinate(float value, float lo, double cell) {
    return int(floor((double(value) - double(lo)) / cell));
}
__global__ static void build_cells(const Point *points, uint32_t n, uint32_t *heads, uint32_t *next,
                                   DeviceGridDesc g) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const Point p = points[i];
    const int x = min(max(coordinate(p.x, g.lo[0], g.cell), 0), g.dims[0] - 1);
    const int y = min(max(coordinate(p.y, g.lo[1], g.cell), 0), g.dims[1] - 1);
    const int z = min(max(coordinate(p.z, g.lo[2], g.cell), 0), g.dims[2] - 1);
    const size_t cell = (size_t(z) * g.dims[1] + y) * g.dims[0] + x;
    next[i] = atomicExch(heads + cell, i);
}
__global__ static void search_cells(const Point *points, const uint32_t *heads,
                                    const uint32_t *next, const Point *queries, uint32_t n,
                                    DeviceGridDesc g, float r2, uint32_t k, uint32_t *counts,
                                    uint32_t *neighbors) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const Point q = queries[i];
    const double fx = floor((double(q.x) - g.lo[0]) / g.cell);
    const double fy = floor((double(q.y) - g.lo[1]) / g.cell);
    const double fz = floor((double(q.z) - g.lo[2]) / g.cell);
    if (fx < -1 || fy < -1 || fz < -1 || fx > g.dims[0] || fy > g.dims[1] || fz > g.dims[2]) {
        counts[i] = 0;
        return;
    }
    const int cx = int(fx), cy = int(fy), cz = int(fz);
    uint32_t count = 0;
    for (int z = max(cz - 1, 0); z <= min(cz + 1, g.dims[2] - 1); ++z)
        for (int y = max(cy - 1, 0); y <= min(cy + 1, g.dims[1] - 1); ++y)
            for (int x = max(cx - 1, 0); x <= min(cx + 1, g.dims[0] - 1); ++x) {
                uint32_t j = heads[(size_t(z) * g.dims[1] + y) * g.dims[0] + x];
                while (j != UINT32_MAX) {
                    const Point p = points[j];
                    const float dx = __fsub_rn(q.x, p.x), dy = __fsub_rn(q.y, p.y),
                                dz = __fsub_rn(q.z, p.z);
                    const float d2 = __fadd_rn(__fadd_rn(__fmul_rn(dx, dx), __fmul_rn(dy, dy)),
                                               __fmul_rn(dz, dz));
                    if (d2 <= r2) {
                        if (count < k)
                            neighbors[size_t(i) * k + count] = p.id;
                        ++count;
                    }
                    j = next[j];
                }
            }
    counts[i] = count;
}
extern "C" cudaError_t grid_build_device(const Point *points, uint32_t n, uint32_t *heads,
                                         uint32_t *next, DeviceGridDesc g, cudaStream_t stream) {
    if (n)
        build_cells<<<(n + 255ull) / 256, 256, 0, stream>>>(points, n, heads, next, g);
    return cudaGetLastError();
}
extern "C" cudaError_t grid_query_device(const Point *points, const uint32_t *heads,
                                         const uint32_t *next, const Point *queries, uint32_t n,
                                         DeviceGridDesc g, float r2, uint32_t k, uint32_t *counts,
                                         uint32_t *neighbors, cudaStream_t stream) {
    if (n)
        search_cells<<<(n + 255ull) / 256, 256, 0, stream>>>(points, heads, next, queries, n, g, r2,
                                                             k, counts, neighbors);
    return cudaGetLastError();
}
