#ifndef PCAP_PLANNER_H
#define PCAP_PLANNER_H
#include "rt_backend.h"
#define PLAN_CANDIDATE_COUNT 4

typedef struct PlanCandidate {
    size_t tile_points, tile_count, window_queries, estimated_tile_bytes, estimated_executor_bytes;
    unsigned slots;
    int feasible;
    double routed_queries, tile_loads;
    double transfer_ms, launch_ms, traversal_ms, predicted_ms;
} PlanCandidate;
typedef struct PlanStats {
    double calibration_ms, probe_build_ms, h2d_gbps;
    double launch_ms, relocation_ms, traversal_ns_per_query, bytes_per_point;
    size_t probe_points, query_samples, budget_bytes, selected_tile_points;
    double selected_predicted_ms;
    PlanCandidate candidates[PLAN_CANDIDATE_COUNT];
} PlanStats;

/* Point data must already be Morton sorted and stays unchanged. The prediction
   models Morton-sorted query windows equal to that point set, count-only output, two executor slots
   where at least two chunks exist, and windows up to 262144 queries. It is a
   calibrated candidate heuristic, not an optimality or memory-fit guarantee;
   window bounds give a conservative tile-load estimate. The executor must still
   check actual compacted tile sizes. Calibration uses
   temporary GPU allocations outside the executor budget and is reported apart.
   budget_bytes==0 uses 80% of currently free device memory. */
int plan_tile_points(RtContext *context, PointSet *sorted, float radius, unsigned group,
                     size_t budget_bytes, size_t *tile_points, PlanStats *stats, char *error,
                     size_t error_size);
#endif
