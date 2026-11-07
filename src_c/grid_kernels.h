#ifndef PCAP_GRID_KERNELS_H
#define PCAP_GRID_KERNELS_H
#include <cuda_runtime_api.h>
#include "common.h"
typedef struct DeviceGridDesc {
    double cell;
    float lo[3];
    int dims[3];
} DeviceGridDesc;
#ifdef __cplusplus
extern "C" {
#endif
cudaError_t grid_build_device(const Point *points, uint32_t n, uint32_t *heads, uint32_t *next,
                              DeviceGridDesc grid, cudaStream_t stream);
cudaError_t grid_query_device(const Point *points, const uint32_t *heads, const uint32_t *next,
                              const Point *queries, uint32_t n, DeviceGridDesc grid, float radius2,
                              uint32_t k, uint32_t *counts, uint32_t *neighbors,
                              cudaStream_t stream);
#ifdef __cplusplus
}
#endif
#endif
