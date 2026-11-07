#include "planner.h"
#include "search.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdint.h>

static int plan_error(char *error, size_t size, const char *what, const char *message) {
    if (error && size)
        snprintf(error, size, "%s: %s", what, message);
    return -1;
}
#define CU(call)                                                                                   \
    do {                                                                                           \
        cudaError_t e_ = (call);                                                                   \
        if (e_ != cudaSuccess) {                                                                   \
            plan_error(error, error_size, #call, cudaGetErrorString(e_));                          \
            goto cleanup;                                                                          \
        }                                                                                          \
    } while (0)

static size_t smaller(size_t a, size_t b) { return a < b ? a : b; }
static size_t spread_index(size_t i, size_t n, size_t samples) {
    /* count <= UINT32_MAX, so the product fits the 64-bit Windows size_t. */
    return i * n / samples;
}
static void chunk_bounds(const Point *p, size_t count, Bounds *bounds) {
    bounds->lo[0] = bounds->hi[0] = p[0].x;
    bounds->lo[1] = bounds->hi[1] = p[0].y;
    bounds->lo[2] = bounds->hi[2] = p[0].z;
    for (size_t i = 1; i < count; ++i) {
        const float v[3] = {p[i].x, p[i].y, p[i].z};
        for (int a = 0; a < 3; ++a) {
            bounds->lo[a] = fminf(bounds->lo[a], v[a]);
            bounds->hi[a] = fmaxf(bounds->hi[a], v[a]);
        }
    }
}
static int covered(Point point, const Bounds *bounds, float radius) {
    const float magnitude = fmaxf(fabsf(point.x), fmaxf(fabsf(point.y), fabsf(point.z)));
    const float r = radius * (1.f + 1e-5f) + 8.f * FLT_EPSILON * magnitude;
    return point.x >= bounds->lo[0] - r && point.x <= bounds->hi[0] + r &&
           point.y >= bounds->lo[1] - r && point.y <= bounds->hi[1] + r &&
           point.z >= bounds->lo[2] - r && point.z <= bounds->hi[2] + r;
}
/* Window/tile box overlap is a conservative load estimate: a query routed to
   a tile implies overlap, but two boxes may overlap without any routed query. */
static int windows_overlap(const Bounds *query, const Bounds *tile, float radius) {
    float magnitude = 0;
    for (int a = 0; a < 3; ++a)
        magnitude = fmaxf(magnitude, fmaxf(fabsf(query->lo[a]), fabsf(query->hi[a])));
    const float r = radius * (1.f + 1e-5f) + 8.f * FLT_EPSILON * magnitude;
    for (int a = 0; a < 3; ++a)
        if (query->hi[a] < tile->lo[a] - r || query->lo[a] > tile->hi[a] + r)
            return 0;
    return 1;
}
static double median3(double a, double b, double c) {
    if (a > b) {
        double t = a;
        a = b;
        b = t;
    }
    if (b > c) {
        double t = b;
        b = c;
        c = t;
    }
    if (a > b)
        b = a;
    return b;
}

int plan_tile_points(RtContext *context, PointSet *sorted, float radius, unsigned group,
                     size_t budget_bytes, size_t *tile_points, PlanStats *stats, char *error,
                     size_t error_size) {
    const size_t sizes[PLAN_CANDIDATE_COUNT] = {16384, 65536, 262144, 1048576};
    const double started = clock_ms();
    const size_t sample_limit = 4096, query_reserve = 65536;
    RtTile tile = {0};
    RtSlot slot = {0};
    cudaStream_t stream = 0;
    cudaEvent_t begin = 0, end = 0;
    Point *host_queries = NULL, *device_queries = NULL;
    Bounds *window_bounds = NULL;
    uint32_t *device_counts = NULL;
    PlanStats result;
    size_t probe_count, probe_begin, samples;
    double transfer_samples[3], upload_samples[3], launch_samples[3], traverse_samples[3];
    double raw_transfer_ms, traverse_ms, bytes_per_ms, launch_ms, per_query_ms;
    int status = -1, best = -1;
    memset(&result, 0, sizeof(result));
    if (!context || !sorted || !sorted->data || !sorted->count || sorted->count > UINT32_MAX ||
        !group || !tile_points || !stats || !isfinite(radius) || radius <= 0 ||
        !isfinite(radius * radius))
        return plan_error(error, error_size, "plan_tile_points", "invalid calibration input");
    if (!budget_bytes) {
        size_t free_bytes = 0, total_bytes = 0;
        CU(cudaMemGetInfo(&free_bytes, &total_bytes));
        budget_bytes = free_bytes - free_bytes / 5;
    }
    result.budget_bytes = budget_bytes;
    probe_count = smaller(sorted->count, 65536);
    /* A contiguous middle Morton chunk, aligned like a real 64K candidate. */
    probe_begin = ((sorted->count - probe_count) / 2 / probe_count) * probe_count;
    samples = smaller(probe_count, sample_limit);
    result.probe_points = probe_count;
    result.query_samples = smaller(sorted->count, sample_limit);
    {
        const double start = clock_ms();
        if (rt_tile_build(context, sorted->data + probe_begin, (uint32_t)probe_count, radius, group,
                          &tile, error, error_size))
            goto cleanup;
        result.probe_build_ms = clock_ms() - start;
    }
    result.bytes_per_point = (double)tile.bytes / (double)probe_count;
    if (rt_slot_alloc(&slot, tile.bytes, error, error_size))
        goto cleanup;
    CU(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CU(cudaEventCreate(&begin));
    CU(cudaEventCreate(&end));
    CU(cudaHostAlloc((void **)&host_queries, samples * sizeof(Point), cudaHostAllocDefault));
    for (size_t i = 0; i < samples; ++i)
        host_queries[i] = sorted->data[probe_begin + spread_index(i, probe_count, samples)];
    CU(cudaMalloc((void **)&device_queries, samples * sizeof(Point)));
    CU(cudaMalloc((void **)&device_counts, samples * sizeof(uint32_t)));
    CU(cudaMemcpyAsync(device_queries, host_queries, samples * sizeof(Point),
                       cudaMemcpyHostToDevice, stream));
    if (rt_slot_upload(context, &tile, &slot, stream, error, error_size))
        goto cleanup;
    CU(cudaStreamSynchronize(stream));
    for (int trial = 0; trial < 3; ++trial) {
        float elapsed = 0;
        CU(cudaEventRecord(begin, stream));
        CU(cudaMemcpyAsync(slot.device, tile.host, tile.bytes, cudaMemcpyHostToDevice, stream));
        CU(cudaEventRecord(end, stream));
        CU(cudaEventSynchronize(end));
        CU(cudaEventElapsedTime(&elapsed, begin, end));
        transfer_samples[trial] = elapsed;
    }
    /* Measure upload plus relocation separately. Each raw upload replaces the
       previously relocated bytes, so relocate again before any traversal. */
    for (int trial = 0; trial < 3; ++trial) {
        float elapsed = 0;
        CU(cudaEventRecord(begin, stream));
        if (rt_slot_upload(context, &tile, &slot, stream, error, error_size))
            goto cleanup;
        CU(cudaEventRecord(end, stream));
        CU(cudaEventSynchronize(end));
        CU(cudaEventElapsedTime(&elapsed, begin, end));
        upload_samples[trial] = elapsed;
    }
    CU(cudaMemsetAsync(device_counts, 0, samples * sizeof(uint32_t), stream));
    if (rt_launch(context, &slot, group, device_queries, NULL, (uint32_t)samples, device_counts,
                  NULL, 0, radius * radius, stream, error, error_size))
        goto cleanup;
    CU(cudaStreamSynchronize(stream));
    for (int trial = 0; trial < 3; ++trial) {
        float elapsed = 0;
        CU(cudaMemsetAsync(device_counts, 0, samples * sizeof(uint32_t), stream));
        CU(cudaEventRecord(begin, stream));
        if (rt_launch(context, &slot, group, device_queries, NULL, 1, device_counts, NULL, 0,
                      radius * radius, stream, error, error_size))
            goto cleanup;
        CU(cudaEventRecord(end, stream));
        CU(cudaEventSynchronize(end));
        CU(cudaEventElapsedTime(&elapsed, begin, end));
        launch_samples[trial] = elapsed;
        CU(cudaMemsetAsync(device_counts, 0, samples * sizeof(uint32_t), stream));
        CU(cudaEventRecord(begin, stream));
        if (rt_launch(context, &slot, group, device_queries, NULL, (uint32_t)samples, device_counts,
                      NULL, 0, radius * radius, stream, error, error_size))
            goto cleanup;
        CU(cudaEventRecord(end, stream));
        CU(cudaEventSynchronize(end));
        CU(cudaEventElapsedTime(&elapsed, begin, end));
        traverse_samples[trial] = elapsed;
    }
    raw_transfer_ms = median3(transfer_samples[0], transfer_samples[1], transfer_samples[2]);
    launch_ms = median3(launch_samples[0], launch_samples[1], launch_samples[2]);
    traverse_ms = median3(traverse_samples[0], traverse_samples[1], traverse_samples[2]);
    /* Nonzero floors make predictions finite even at event timer resolution. */
    raw_transfer_ms = fmax(raw_transfer_ms, 0.001);
    launch_ms = fmax(launch_ms, 0.001);
    per_query_ms = fmax(traverse_ms - launch_ms, traverse_ms * 0.25) / (double)samples;
    bytes_per_ms = (double)tile.bytes / raw_transfer_ms;
    result.h2d_gbps = bytes_per_ms / 1e6;
    result.launch_ms = launch_ms;
    result.relocation_ms = fmax(
        0.0, median3(upload_samples[0], upload_samples[1], upload_samples[2]) - raw_transfer_ms);
    result.traversal_ns_per_query = per_query_ms * 1e6;
    /* A single probe cannot certify the largest compressed GAS. Reserve 25%
       over its byte/point slope plus alignment/header room, then let the real
       executor enforce its measured footprint after construction. */
    for (int candidate = 0; candidate < PLAN_CANDIDATE_COUNT; ++candidate) {
        PlanCandidate *p = &result.candidates[candidate];
        double estimate;
        size_t slot_bytes, remaining;
        p->tile_points = smaller(sizes[candidate], sorted->count);
        p->tile_count = (sorted->count + p->tile_points - 1) / p->tile_points;
        p->slots = p->tile_count > 1 ? 2 : 1;
        estimate = ceil(result.bytes_per_point * (double)p->tile_points * 1.25) + 4096;
        if (!isfinite(estimate) || estimate >= (double)SIZE_MAX)
            continue;
        p->estimated_tile_bytes = (size_t)estimate;
        if (p->estimated_tile_bytes > SIZE_MAX - rt_slot_overhead_bytes())
            continue;
        slot_bytes = p->estimated_tile_bytes + rt_slot_overhead_bytes();
        if (slot_bytes > budget_bytes / p->slots)
            continue;
        remaining = budget_bytes - slot_bytes * p->slots;
        if (remaining < query_reserve)
            continue;
        p->window_queries = remaining / (sizeof(Point) + sizeof(uint32_t) * (1 + p->slots));
        p->window_queries = smaller(p->window_queries, 262144);
        p->window_queries = smaller(p->window_queries, sorted->count);
        if (!p->window_queries)
            continue;
        p->estimated_executor_bytes =
            slot_bytes * p->slots +
            p->window_queries * (sizeof(Point) + sizeof(uint32_t) * (1 + p->slots));
        p->feasible = 1;
        const size_t window_count = (sorted->count + p->window_queries - 1) / p->window_queries;
        window_bounds = malloc(window_count * sizeof(*window_bounds));
        if (!window_bounds) {
            plan_error(error, error_size, "plan_tile_points",
                       "out of memory for query window bounds");
            goto cleanup;
        }
        for (size_t window = 0; window < window_count; ++window) {
            const size_t start = window * p->window_queries;
            chunk_bounds(sorted->data + start, smaller(p->window_queries, sorted->count - start),
                         &window_bounds[window]);
        }
        for (size_t start = 0; start < sorted->count; start += p->tile_points) {
            const size_t count = smaller(p->tile_points, sorted->count - start);
            Bounds bounds;
            size_t extra_hits = 0;
            double coverage;
            chunk_bounds(sorted->data + start, count, &bounds);
            for (size_t i = 0; i < result.query_samples; ++i) {
                const size_t index = spread_index(i, sorted->count, result.query_samples);
                if (index >= start && index < start + count)
                    continue;
                if (covered(sorted->data[index], &bounds, radius))
                    ++extra_hits;
            }
            /* Owned queries are counted exactly. Only extra routing through
               other chunks is sampled, avoiding empty-sample tile bias. */
            coverage = fmin(1.0, (double)count / (double)sorted->count +
                                     (double)extra_hits / (double)result.query_samples);
            p->routed_queries += (double)sorted->count * coverage;
            for (size_t window = 0; window < window_count; ++window)
                if (windows_overlap(&window_bounds[window], &bounds, radius))
                    p->tile_loads += 1.0;
        }
        free(window_bounds);
        window_bounds = NULL;
        p->transfer_ms = p->tile_loads * (double)p->estimated_tile_bytes / bytes_per_ms;
        p->launch_ms = p->tile_loads * (launch_ms + result.relocation_ms);
        {
            const double probe_log = log2(fmax(2.0, (double)probe_count / group));
            const double candidate_log = log2(fmax(2.0, (double)p->tile_points / group));
            const double scale = fmax(0.5, candidate_log / probe_log);
            p->traversal_ms = p->routed_queries * per_query_ms * scale;
        }
        /* Serial sum is deliberate: no unmeasured overlap credit. */
        p->predicted_ms = p->transfer_ms + p->launch_ms + p->traversal_ms;
        if (!isfinite(p->predicted_ms)) {
            p->feasible = 0;
            continue;
        }
        if (best < 0 || p->predicted_ms < result.candidates[best].predicted_ms)
            best = candidate;
    }
    result.calibration_ms = clock_ms() - started;
    *stats = result;
    if (best < 0) {
        plan_error(error, error_size, "plan_tile_points",
                   "no candidate fits two slots plus query reserve; raise budget or set a smaller "
                   "tile manually");
        goto cleanup;
    }
    result.selected_tile_points = result.candidates[best].tile_points;
    result.selected_predicted_ms = result.candidates[best].predicted_ms;
    *tile_points = result.selected_tile_points;
    *stats = result;
    status = 0;
cleanup:
    free(window_bounds);
    if (stream)
        cudaStreamSynchronize(stream);
    if (begin)
        cudaEventDestroy(begin);
    if (end)
        cudaEventDestroy(end);
    if (device_queries)
        cudaFree(device_queries);
    if (device_counts)
        cudaFree(device_counts);
    if (host_queries)
        cudaFreeHost(host_queries);
    rt_slot_free(&slot);
    rt_tile_free(&tile);
    if (stream)
        cudaStreamDestroy(stream);
    result.calibration_ms = clock_ms() - started;
    *stats = result;
    return status;
}
