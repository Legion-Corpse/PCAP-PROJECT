#ifndef PCAP_RT_PARAMS_H
#define PCAP_RT_PARAMS_H
#include <stdint.h>
#include "common.h"
/* shared by host and device code; keep every field naturally aligned */
typedef struct RtParams {
    uint64_t handle;
    const Point *points;
    const Point *queries;
    const uint32_t *query_list;
    uint32_t *counts;
    uint32_t *neighbors;
    uint32_t group;
    uint32_t num_queries;
    uint32_t max_neighbors;
    float radius2;
} RtParams;
#endif
