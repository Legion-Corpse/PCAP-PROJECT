#ifndef PCAP_RT_BACKEND_H
#define PCAP_RT_BACKEND_H
#include <cuda_runtime_api.h>
#include <optix_types.h>
#include "common.h"
#include "rt_params.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct RtContext RtContext;
typedef struct RtTile {
    void *host; /* cudaHostAlloc pinned, points followed by aligned GAS */
    size_t bytes, gas_offset, gas_bytes;
    uint32_t n_points, group;
    OptixRelocationInfo relocation;
} RtTile;
typedef struct RtSlot {
    void *device;
    size_t capacity;
    OptixTraversableHandle handle;
    void *device_params;
    RtParams *host_params;
    cudaEvent_t params_uploaded;
    int params_pending;
} RtSlot;

/* All outputs must be zero initialized. All functions return 0/-1. */
int rt_context_create(int validation, RtContext **out, char *error, size_t error_size);
void rt_context_destroy(RtContext *context);
int rt_tile_build(RtContext *context, const Point *points, uint32_t count, float radius,
                  uint32_t group, RtTile *out, char *error, size_t error_size);
void rt_tile_free(RtTile *tile);
size_t rt_slot_overhead_bytes(void);
int rt_slot_alloc(RtSlot *slot, size_t bytes, char *error, size_t error_size);
void rt_slot_free(RtSlot *slot);
/* Upload + relocation are enqueued on stream. Caller supplies event dependencies
   when a slot is shared by transfer and compute streams. */
int rt_slot_upload(RtContext *context, const RtTile *tile, RtSlot *slot, cudaStream_t stream,
                   char *error, size_t error_size);
/* Device pointers only. Existing counts/list entries accumulate across tiles.
   Callers zero counts and initialize unused list entries to UINT32_MAX once per
   query window. query_list may be NULL. No synchronization after the launch. */
int rt_launch(RtContext *context, RtSlot *slot, uint32_t group, const Point *queries,
              const uint32_t *query_list, uint32_t count, uint32_t *counts, uint32_t *neighbors,
              uint32_t max_neighbors, float radius2, cudaStream_t stream, char *error,
              size_t error_size);
#ifdef __cplusplus
}
#endif
#endif
