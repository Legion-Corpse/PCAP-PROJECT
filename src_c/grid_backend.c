#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include "grid_backend.h"
#include "grid_kernels.h"
#include "search.h"

struct CudaGrid {
    Point *points;
    uint32_t *heads, *next;
    uint32_t count;
    DeviceGridDesc grid;
    float radius2;
    cudaStream_t stream;
    CudaGridStats stats;
};
static int grid_error(char *error, size_t size, const char *what, const char *message) {
    if (error && size)
        snprintf(error, size, "%s: %s", what, message);
    return -1;
}
#define CUDA_TRY(call)                                                                             \
    do {                                                                                           \
        cudaError_t e_ = (call);                                                                   \
        if (e_ != cudaSuccess) {                                                                   \
            grid_error(error, error_size, #call, cudaGetErrorString(e_));                          \
            goto fail;                                                                             \
        }                                                                                          \
    } while (0)

void cuda_grid_free(CudaGrid *index) {
    if (!index)
        return;
    if (index->points)
        cudaFree(index->points);
    if (index->heads)
        cudaFree(index->heads);
    if (index->next)
        cudaFree(index->next);
    if (index->stream)
        cudaStreamDestroy(index->stream);
    free(index);
}
int cuda_grid_build(const PointSet *points, float radius, CudaGrid **out, CudaGridStats *stats,
                    char *error, size_t error_size) {
    const double start = clock_ms();
    CudaGrid *index = NULL;
    Bounds bounds;
    double cell, max_abs = 0, extent[3];
    size_t cells;
    if (!points || !points->count || points->count > UINT32_MAX || !out || *out ||
        !isfinite(radius) || radius <= 0 || !isfinite(radius * radius) || radius * radius < FLT_MIN)
        return grid_error(error, error_size, "cuda_grid_build",
                          "invalid points, radius, or output");
    if (pointset_bounds(points, &bounds, error, error_size))
        return -1;
    index = calloc(1, sizeof(*index));
    if (!index)
        return grid_error(error, error_size, "cuda_grid_build", "out of host memory");
    for (int a = 0; a < 3; ++a) {
        max_abs = fmax(max_abs, fmax(fabs(bounds.lo[a]), fabs(bounds.hi[a])));
        extent[a] = (double)bounds.hi[a] - bounds.lo[a];
        index->grid.lo[a] = bounds.lo[a];
    }
    cell = (double)(radius * (1.f + 1e-5f) + 4.f * FLT_EPSILON * (float)max_abs);
    /* Cap the directory at 2^24 cells while keeping cells at least r wide. */
    for (;;) {
        cells = 1;
        int valid = 1;
        for (int a = 0; a < 3; ++a) {
            const double d = floor(extent[a] / cell) + 1;
            if (d > 16777216 || d < 1 || !isfinite(d)) {
                valid = 0;
                break;
            }
            index->grid.dims[a] = (int)d;
            cells *= (size_t)index->grid.dims[a];
            if (cells > 16777216) {
                valid = 0;
                break;
            }
        }
        if (valid)
            break;
        cell *= 1.25;
    }
    index->grid.cell = cell;
    index->count = (uint32_t)points->count;
    index->radius2 = radius * radius;
    CUDA_TRY(cudaStreamCreateWithFlags(&index->stream, cudaStreamNonBlocking));
    CUDA_TRY(cudaMalloc((void **)&index->points, points->count * sizeof(Point)));
    CUDA_TRY(cudaMalloc((void **)&index->next, points->count * sizeof(uint32_t)));
    CUDA_TRY(cudaMalloc((void **)&index->heads, cells * sizeof(uint32_t)));
    CUDA_TRY(cudaMemcpyAsync(index->points, points->data, points->count * sizeof(Point),
                             cudaMemcpyHostToDevice, index->stream));
    CUDA_TRY(cudaMemsetAsync(index->heads, 0xff, cells * sizeof(uint32_t), index->stream));
    CUDA_TRY(grid_build_device(index->points, index->count, index->heads, index->next, index->grid,
                               index->stream));
    CUDA_TRY(cudaStreamSynchronize(index->stream));
    index->stats.build_ms = clock_ms() - start;
    index->stats.device_bytes =
        points->count * (sizeof(Point) + sizeof(uint32_t)) + cells * sizeof(uint32_t);
    if (stats)
        *stats = index->stats;
    *out = index;
    return 0;
fail:
    cuda_grid_free(index);
    return -1;
}
int cuda_grid_query(CudaGrid *index, const PointSet *queries, unsigned k, SearchResult *out,
                    CudaGridStats *stats, char *error, size_t error_size) {
    const double start = clock_ms();
    Point *d_queries = NULL;
    uint32_t *d_counts = NULL, *d_neighbors = NULL;
    const size_t window = queries && queries->count < 65536 ? queries->count : 65536;
    if (!index || !queries || !out || queries->count > UINT32_MAX)
        return grid_error(error, error_size, "cuda_grid_query",
                          "invalid index, queries, or output");
    if (search_result_alloc(out, queries->count, k, error, error_size))
        return -1;
    if (!queries->count) {
        if (stats) {
            *stats = index->stats;
            stats->search_ms = clock_ms() - start;
        }
        return 0;
    }
    if (k && window > SIZE_MAX / (sizeof(uint32_t) * (size_t)k)) {
        grid_error(error, error_size, "cuda_grid_query", "neighbor allocation overflow");
        goto fail;
    }
    CUDA_TRY(cudaMalloc((void **)&d_queries, window * sizeof(Point)));
    CUDA_TRY(cudaMalloc((void **)&d_counts, window * sizeof(uint32_t)));
    if (k)
        CUDA_TRY(cudaMalloc((void **)&d_neighbors, window * (size_t)k * sizeof(uint32_t)));
    for (size_t begin = 0; begin < queries->count; begin += window) {
        uint32_t n = (uint32_t)(queries->count - begin < window ? queries->count - begin : window);
        CUDA_TRY(cudaMemcpyAsync(d_queries, queries->data + begin, (size_t)n * sizeof(Point),
                                 cudaMemcpyHostToDevice, index->stream));
        if (k)
            CUDA_TRY(cudaMemsetAsync(d_neighbors, 0xff, (size_t)n * k * sizeof(uint32_t),
                                     index->stream));
        CUDA_TRY(grid_query_device(index->points, index->heads, index->next, d_queries, n,
                                   index->grid, index->radius2, k, d_counts, d_neighbors,
                                   index->stream));
        CUDA_TRY(cudaMemcpyAsync(out->counts + begin, d_counts, (size_t)n * sizeof(uint32_t),
                                 cudaMemcpyDeviceToHost, index->stream));
        if (k)
            CUDA_TRY(cudaMemcpyAsync(out->neighbors + begin * k, d_neighbors,
                                     (size_t)n * k * sizeof(uint32_t), cudaMemcpyDeviceToHost,
                                     index->stream));
        CUDA_TRY(cudaStreamSynchronize(index->stream));
    }
    cudaFree(d_queries);
    cudaFree(d_counts);
    if (d_neighbors)
        cudaFree(d_neighbors);
    if (stats) {
        *stats = index->stats;
        stats->search_ms = clock_ms() - start;
        stats->device_bytes += window * (sizeof(Point) + sizeof(uint32_t) * (1 + (size_t)k));
    }
    return 0;
fail:
    if (d_queries)
        cudaFree(d_queries);
    if (d_counts)
        cudaFree(d_counts);
    if (d_neighbors)
        cudaFree(d_neighbors);
    search_result_free(out);
    return -1;
}
int rt_cuda_grid(const PointSet *points, const PointSet *queries, float radius, unsigned k,
                 SearchResult *out, CudaGridStats *stats, char *error, size_t error_size) {
    CudaGrid *index = NULL;
    if (cuda_grid_build(points, radius, &index, stats, error, error_size))
        return -1;
    const int result = cuda_grid_query(index, queries, k, out, stats, error, error_size);
    cuda_grid_free(index);
    return result;
}
