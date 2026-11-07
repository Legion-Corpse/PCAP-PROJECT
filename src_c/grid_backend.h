#ifndef PCAP_GRID_BACKEND_H
#define PCAP_GRID_BACKEND_H
#include "common.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct CudaGrid CudaGrid;
typedef struct CudaGridStats {
    double build_ms, search_ms;
    size_t device_bytes;
} CudaGridStats;
int cuda_grid_build(const PointSet *points, float radius, CudaGrid **out, CudaGridStats *stats,
                    char *error, size_t error_size);
int cuda_grid_query(CudaGrid *index, const PointSet *queries, unsigned max_neighbors,
                    SearchResult *out, CudaGridStats *stats, char *error, size_t error_size);
void cuda_grid_free(CudaGrid *index);
int rt_cuda_grid(const PointSet *points, const PointSet *queries, float radius,
                 unsigned max_neighbors, SearchResult *out, CudaGridStats *stats, char *error,
                 size_t error_size);
#ifdef __cplusplus
}
#endif
#endif
