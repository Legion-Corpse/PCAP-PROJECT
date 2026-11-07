#include "search.h"
#include "rt_backend.h"
#include "planner.h"
#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#ifdef _WIN32
#include <windows.h>
#endif

struct StreamIndex {
    RtContext *rt;
    RtTile *tiles;
    Bounds *bounds;
    size_t count;
    StreamOptions options;
    float radius;
    RtSlot resident;
};
double clock_ms(void) {
    LARGE_INTEGER n, f;
    QueryPerformanceCounter(&n);
    QueryPerformanceFrequency(&f);
    return 1000.0 * (double)n.QuadPart / (double)f.QuadPart;
}
static int fail(char *e, size_t n, const char *s) {
    if (e && n)
        snprintf(e, n, "%s", s);
    return -1;
}
static int hit(Point q, Bounds b, float r) {
    float a = fmaxf(fmaxf(fabsf(q.x), fabsf(q.y)), fabsf(q.z));
    r = r * (1 + 1e-5f) + 8 * 1.192092896e-7f * a;
    return q.x >= b.lo[0] - r && q.x <= b.hi[0] + r && q.y >= b.lo[1] - r && q.y <= b.hi[1] + r &&
           q.z >= b.lo[2] - r && q.z <= b.hi[2] + r;
}
static int overlap_bounds(Bounds a, Bounds b, float r) {
    int k;
    float mag = 0;
    for (k = 0; k < 3; k++)
        mag = fmaxf(mag, fmaxf(fabsf(a.lo[k]), fabsf(a.hi[k])));
    r = r * (1 + 1e-5f) + 8 * FLT_EPSILON * mag;
    for (k = 0; k < 3; k++)
        if (a.hi[k] < b.lo[k] - r || a.lo[k] > b.hi[k] + r)
            return 0;
    return 1;
}
size_t stream_tile_count(const StreamIndex *s) { return s ? s->count : 0; }
int stream_tile_view(const StreamIndex *s, size_t tile, StreamTileView *view) {
    if (!s || !view || tile >= s->count)
        return -1;
    view->points = (const Point *)s->tiles[tile].host;
    view->count = s->tiles[tile].n_points;
    view->bytes = s->tiles[tile].bytes;
    view->bounds = s->bounds[tile];
    return 0;
}
int stream_tile_routes_query(const StreamIndex *s, size_t tile, Point query) {
    if (!s || tile >= s->count || !isfinite(query.x) || !isfinite(query.y) || !isfinite(query.z))
        return -1;
    return hit(query, s->bounds[tile], s->radius);
}
void stream_free(StreamIndex *s) {
    size_t i;
    if (!s)
        return;
    rt_slot_free(&s->resident);
    if (s->tiles)
        for (i = 0; i < s->count; i++)
            rt_tile_free(&s->tiles[i]);
    free(s->tiles);
    free(s->bounds);
    rt_context_destroy(s->rt);
    free(s);
}
int stream_build(const PointSet *points, float radius, const StreamOptions *o, StreamIndex **out,
                 StreamStats *stats, char *error, size_t es) {
    double start = clock_ms(), t;
    size_t i, tile = o->tile_points ? o->tile_points : 65536;
    PointSet ordered = {0};
    StreamIndex *s = NULL;
    if (!out || !points || !points->count || !o->group || !tile || !isfinite(radius) ||
        radius <= 0 || !isfinite(radius * radius) || radius * radius < FLT_MIN)
        return fail(error, es, "invalid index configuration or nonnormal radius square");
    s = (StreamIndex *)calloc(1, sizeof(*s));
    if (!s)
        return fail(error, es, "index allocation failed");
    s->options = *o;
    s->radius = radius;
    if (pointset_alloc(&ordered, points->count, error, es))
        goto error;
    memcpy(ordered.data, points->data, points->count * sizeof(Point));
    t = clock_ms();
    if (pointset_morton_sort(&ordered, error, es))
        goto error;
    stats->sort_ms = clock_ms() - t;
    if (rt_context_create(0, &s->rt, error, es))
        goto error;
    if (!o->tile_points) {
        PlanStats plan = {0};
        if (plan_tile_points(s->rt, &ordered, radius, o->group, o->budget_bytes, &tile, &plan,
                             error, es))
            goto error;
        stats->calibration_ms = plan.calibration_ms;
        stats->predicted_ms = plan.selected_predicted_ms;
    }
    stats->selected_tile_points = tile;
    s->count = (points->count + tile - 1) / tile;
    s->tiles = (RtTile *)calloc(s->count, sizeof(RtTile));
    s->bounds = (Bounds *)calloc(s->count, sizeof(Bounds));
    if (!s->tiles || !s->bounds)
        goto oom;
    for (i = 0; i < s->count; i++) {
        PointSet part = {ordered.data + i * tile, points->count - i * tile};
        if (part.count > tile)
            part.count = tile;
        if (pointset_bounds(&part, &s->bounds[i], error, es) ||
            rt_tile_build(s->rt, part.data, (uint32_t)part.count, radius, o->group, &s->tiles[i],
                          error, es))
            goto error;
        stats->host_bytes += s->tiles[i].bytes;
    }
    if (o->resident) {
        if (s->count != 1) {
            fail(error, es, "resident index must use one tile");
            goto error;
        }
        if (rt_slot_alloc(&s->resident, s->tiles[0].bytes, error, es) ||
            rt_slot_upload(s->rt, &s->tiles[0], &s->resident, 0, error, es) ||
            cudaDeviceSynchronize() != cudaSuccess)
            goto error;
    }
    pointset_free(&ordered);
    stats->tiles = s->count;
    stats->build_ms = clock_ms() - start;
    *out = s;
    return 0;
oom:
    fail(error, es, "tile table allocation failed");
error:
    pointset_free(&ordered);
    stream_free(s);
    return -1;
}
static int stream_query_ordered(StreamIndex *s, const PointSet *q, unsigned K, SearchResult *out,
                                StreamStats *stats, char *error, size_t es) {
    size_t maxblob = 0, i, w, startq, W,
           budget = s->options.budget_bytes ? s->options.budget_bytes : (size_t)1 << 30;
    unsigned slots = s->options.slots ? s->options.slots : 2, j;
    RtSlot slot[2];
    cudaEvent_t ready[2] = {0}, done[2] = {0};
    cudaStream_t copy = 0, compute = 0;
    Point *dq = NULL;
    uint32_t *dc = NULL, *dn = NULL, *dl[2] = {0}, *hl[2] = {0};
    int used[2] = {0}, result = -1;
    double start = clock_ms();
    size_t query_bytes;
    memset(slot, 0, sizeof(slot));
    if (slots > 2)
        slots = 2;
    if (slots > s->count)
        slots = (unsigned)s->count;
    for (i = 0; i < s->count; i++)
        if (s->tiles[i].bytes > maxblob)
            maxblob = s->tiles[i].bytes;
    query_bytes = sizeof(Point) + sizeof(uint32_t) * (1 + K + slots);
    if (maxblob > budget / slots)
        return fail(error, es, "budget cannot hold tile slots; lower tile-points");
    if (rt_slot_overhead_bytes() * slots > budget - maxblob * slots)
        return fail(error, es, "budget cannot hold slot parameters");
    W = (budget - maxblob * slots - rt_slot_overhead_bytes() * slots) / query_bytes;
    if (W > q->count)
        W = q->count;
    if (W > 262144)
        W = 262144;
    if (s->options.window_queries && W > s->options.window_queries)
        W = s->options.window_queries;
    if (!W)
        return fail(error, es, "budget cannot hold query buffers");
    stats->executor_bytes = (maxblob + rt_slot_overhead_bytes()) * slots + W * query_bytes;
    if (search_result_alloc(out, q->count, K, error, es))
        return -1;
#define CU(call)                                                                                   \
    do {                                                                                           \
        cudaError_t c = (call);                                                                    \
        if (c != cudaSuccess) {                                                                    \
            fail(error, es, cudaGetErrorString(c));                                                \
            goto cleanup;                                                                          \
        }                                                                                          \
    } while (0)
    CU(cudaStreamCreate(&copy));
    CU(cudaStreamCreate(&compute));
    CU(cudaMalloc((void **)&dq, W * sizeof(Point)));
    CU(cudaMalloc((void **)&dc, W * 4));
    if (K)
        CU(cudaMalloc((void **)&dn, W * 4 * K));
    for (j = 0; j < slots; j++) {
        if (s->options.resident)
            slot[j] = s->resident;
        else if (rt_slot_alloc(&slot[j], maxblob, error, es))
            goto cleanup;
        CU(cudaMalloc((void **)&dl[j], W * 4));
        CU(cudaHostAlloc((void **)&hl[j], W * 4, 0));
        CU(cudaEventCreateWithFlags(&ready[j], cudaEventDisableTiming));
        CU(cudaEventCreateWithFlags(&done[j], cudaEventDisableTiming));
        CU(cudaEventRecord(done[j], compute));
    }
    for (startq = 0; startq < q->count; startq += W) {
        size_t count = q->count - startq, step = 0;
        Bounds query_bounds;
        PointSet part;
        if (count > W)
            count = W;
        stats->windows++;
        part.data = q->data + startq;
        part.count = count;
        if (pointset_bounds(&part, &query_bounds, error, es))
            goto cleanup;
        CU(cudaMemcpy(dq, q->data + startq, count * sizeof(Point), cudaMemcpyHostToDevice));
        CU(cudaMemset(dc, 0, count * 4));
        if (K)
            CU(cudaMemset(dn, 255, count * 4 * K));
        for (i = 0; i < s->count; i++) {
            if (!overlap_bounds(query_bounds, s->bounds[i], s->radius))
                continue;
            size_t n = 0;
            j = (unsigned)(step % slots);
            if (used[j])
                CU(cudaEventSynchronize(ready[j]));
            for (w = 0; w < count; w++)
                if (hit(q->data[startq + w], s->bounds[i], s->radius))
                    hl[j][n++] = (uint32_t)w;
            if (!n)
                continue;
            CU(cudaStreamWaitEvent(copy, done[j], 0));
            if (!s->options.resident &&
                rt_slot_upload(s->rt, &s->tiles[i], &slot[j], copy, error, es))
                goto cleanup;
            CU(cudaMemcpyAsync(dl[j], hl[j], n * 4, cudaMemcpyHostToDevice, copy));
            CU(cudaEventRecord(ready[j], copy));
            CU(cudaStreamWaitEvent(compute, ready[j], 0));
            if (rt_launch(s->rt, &slot[j], s->options.group, dq, dl[j], (uint32_t)n, dc, dn, K,
                          s->radius * s->radius, compute, error, es))
                goto cleanup;
            CU(cudaEventRecord(done[j], compute));
            used[j] = 1;
            step++;
            if (!s->options.resident) {
                stats->loads++;
                stats->h2d_bytes += (double)s->tiles[i].bytes;
            }
            if (!s->options.overlap)
                CU(cudaStreamSynchronize(compute));
        }
        CU(cudaStreamSynchronize(compute));
        CU(cudaStreamSynchronize(copy));
        CU(cudaMemcpy(out->counts + startq, dc, count * 4, cudaMemcpyDeviceToHost));
        if (K)
            CU(cudaMemcpy(out->neighbors + startq * K, dn, count * 4 * K, cudaMemcpyDeviceToHost));
    }
    stats->search_ms = clock_ms() - start;
    result = 0;
cleanup:
    if (compute)
        cudaStreamSynchronize(compute);
    if (copy)
        cudaStreamSynchronize(copy);
    for (j = 0; j < slots; j++) {
        if (ready[j])
            cudaEventDestroy(ready[j]);
        if (done[j])
            cudaEventDestroy(done[j]);
        if (dl[j])
            cudaFree(dl[j]);
        if (hl[j])
            cudaFreeHost(hl[j]);
        if (s->options.resident) {
            if (slot[j].device)
                s->resident = slot[j];
        } else
            rt_slot_free(&slot[j]);
    }
    if (dq)
        cudaFree(dq);
    if (dc)
        cudaFree(dc);
    if (dn)
        cudaFree(dn);
    if (copy)
        cudaStreamDestroy(copy);
    if (compute)
        cudaStreamDestroy(compute);
    if (result)
        search_result_free(out);
    return result;
#undef CU
}
int stream_query(StreamIndex *s, const PointSet *q, unsigned K, SearchResult *out,
                 StreamStats *stats, char *error, size_t es) {
    PointSet ordered = {0};
    SearchResult sorted = {0};
    size_t i;
    double start = clock_ms();
    int result = -1;
    if (pointset_alloc(&ordered, q->count, error, es))
        return -1;
    memcpy(ordered.data, q->data, q->count * sizeof(Point));
    for (i = 0; i < ordered.count; i++)
        ordered.data[i].id = (uint32_t)i;
    if (pointset_morton_sort(&ordered, error, es) ||
        stream_query_ordered(s, &ordered, K, &sorted, stats, error, es) ||
        search_result_alloc(out, q->count, K, error, es))
        goto cleanup;
    for (i = 0; i < ordered.count; i++) {
        size_t original = ordered.data[i].id;
        out->counts[original] = sorted.counts[i];
        if (K)
            memcpy(out->neighbors + original * K, sorted.neighbors + i * K, K * sizeof(uint32_t));
    }
    stats->search_ms = clock_ms() - start;
    result = 0;
cleanup:
    pointset_free(&ordered);
    search_result_free(&sorted);
    return result;
}
