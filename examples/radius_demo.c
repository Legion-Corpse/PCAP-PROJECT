#include "common.h"
#include "grid_backend.h"
#include "search.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int compare_id(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

static int show_tiles(const StreamIndex *index, const PointSet *queries) {
    size_t tile, i;
    puts("Tile structure (actual Morton-ordered point ownership; tile numbers start at 0):");
    for (tile = 0; tile < stream_tile_count(index); ++tile) {
        StreamTileView view;
        if (stream_tile_view(index, tile, &view))
            return -1;
        printf("  Tile %zu: IDs=[", tile);
        for (i = 0; i < view.count; ++i)
            printf("%s%u", i ? ", " : "", view.points[i].id);
        printf("] points=%zu blob_bytes=%zu\n", view.count, view.bytes);
        printf("    bounds: (%g, %g, %g) -> (%g, %g, %g)\n", view.bounds.lo[0], view.bounds.lo[1],
               view.bounds.lo[2], view.bounds.hi[0], view.bounds.hi[1], view.bounds.hi[2]);
    }
    puts("Query routing (same conservative radius-expanded bounds test as execution):");
    for (i = 0; i < queries->count; ++i) {
        int first = 1;
        printf("  Q%zu candidate tiles=[", i);
        for (tile = 0; tile < stream_tile_count(index); ++tile) {
            int routed = stream_tile_routes_query(index, tile, queries->data[i]);
            if (routed < 0)
                return -1;
            if (routed) {
                printf("%s%zu", first ? "" : ", ", tile);
                first = 0;
            }
        }
        puts("]");
    }
    puts("Each point belongs to one tile. Queries may visit several tiles; the distance test "
         "decides matches. This view is not a GPU execution trace.");
    return 0;
}

int main(int argc, char **argv) {
    const char *method = argc > 1 ? argv[1] : "rt-ooc";
    const char *point_path = argc > 2 ? argv[2] : "examples/points.xyz";
    const char *query_path = argc > 3 ? argv[3] : "examples/queries.xyz";
    PointSet points = {0}, queries = {0};
    SearchResult result = {0}, reference = {0};
    CpuGrid *cpu = NULL;
    CudaGrid *grid = NULL;
    StreamIndex *rt = NULL;
    StreamStats stats = {0};
    CudaGridStats grid_stats = {0};
    StreamOptions options = {2, 1048576, 0, 3, 2, 1, 0};
    size_t k = 8, i, j, h;
    uint32_t largest_count = 0;
    double started, build_ms = 0, search_ms = 0, verification_ms;
    char error[512] = {0};
    int status = 1;
    if (argc > 6 || (argc > 1 && !strcmp(method, "--help"))) {
        puts("Usage: radius_demo [cpu-grid|cuda-grid|rt-incore|rt-ooc] "
             "[points.xyz] [queries.xyz] [max_ids:0..1024] [tile_points:1..4294967295]\n"
             "Radius is fixed at 1. Run from the repository root when using default paths.");
        return argc > 6 ? 2 : 0;
    }
    if (strcmp(method, "cpu-grid") && strcmp(method, "cuda-grid") && strcmp(method, "rt-incore") &&
        strcmp(method, "rt-ooc")) {
        fprintf(stderr, "Unknown method: %s\n", method);
        return 2;
    }
    if (argc > 4 && (parse_count(argv[4], &k, error, sizeof(error)) || k > 1024)) {
        fprintf(stderr, "max_ids must be an integer from 0 to 1024\n");
        return 2;
    }
    if (argc > 5 && (parse_count(argv[5], &options.tile_points, error, sizeof(error)) ||
                     !options.tile_points || options.tile_points > UINT32_MAX)) {
        fprintf(stderr, "tile_points must be an integer from 1 to 4294967295\n");
        return 2;
    }
    if (pointset_load(point_path, &points, error, sizeof(error)) ||
        pointset_load(query_path, &queries, error, sizeof(error)))
        goto cleanup;
    if (!points.count || !queries.count) {
        snprintf(error, sizeof(error), "point and query files must be nonempty");
        goto cleanup;
    }
    printf("Method: %s | radius: 1 | max stored IDs: %zu\n", method, k);
    puts("Stored points (ID: x y z):");
    for (i = 0; i < points.count; ++i)
        printf("  %u: %g %g %g\n", points.data[i].id, points.data[i].x, points.data[i].y,
               points.data[i].z);
    started = clock_ms();
    if (!strcmp(method, "cpu-grid")) {
        if (cpu_grid_build(&points, 1.f, &cpu, error, sizeof(error)))
            goto cleanup;
        build_ms = clock_ms() - started;
        started = clock_ms();
        if (cpu_grid_query(cpu, &queries, (unsigned)k, &result, error, sizeof(error)))
            goto cleanup;
        search_ms = clock_ms() - started;
    } else if (!strcmp(method, "cuda-grid")) {
        if (cuda_grid_build(&points, 1.f, &grid, &grid_stats, error, sizeof(error)))
            goto cleanup;
        build_ms = clock_ms() - started;
        started = clock_ms();
        if (cuda_grid_query(grid, &queries, (unsigned)k, &result, &grid_stats, error,
                            sizeof(error)))
            goto cleanup;
        search_ms = clock_ms() - started;
    } else {
        if (!strcmp(method, "rt-incore")) {
            options.tile_points = points.count;
            options.slots = 1;
            options.resident = 1;
        }
        if (stream_build(&points, 1.f, &options, &rt, &stats, error, sizeof(error)))
            goto cleanup;
        build_ms = clock_ms() - started;
        started = clock_ms();
        if (stream_query(rt, &queries, (unsigned)k, &result, &stats, error, sizeof(error)))
            goto cleanup;
        search_ms = clock_ms() - started;
    }
    started = clock_ms();
    if (cpu_brute(&points, &queries, 1.f, 0, &reference, error, sizeof(error)))
        goto cleanup;
    for (i = 0; i < queries.count; ++i) {
        size_t returned = result.counts[i] < k ? result.counts[i] : k;
        uint32_t *ids = k ? result.neighbors + i * k : NULL;
        if (result.counts[i] != reference.counts[i]) {
            snprintf(error, sizeof(error), "count mismatch for query %zu", i);
            goto cleanup;
        }
        if (result.counts[i] > largest_count)
            largest_count = result.counts[i];
        for (j = 0; j < returned; ++j) {
            Point p, q = queries.data[i];
            float dx, dy, dz, distance2;
            if (ids[j] >= points.count) {
                snprintf(error, sizeof(error), "invalid result ID");
                goto cleanup;
            }
            p = points.data[ids[j]];
            dx = q.x - p.x;
            dy = q.y - p.y;
            dz = q.z - p.z;
            distance2 = (dx * dx + dy * dy) + dz * dz;
            if (!(distance2 <= 1.f)) {
                snprintf(error, sizeof(error), "result ID outside radius");
                goto cleanup;
            }
            for (h = 0; h < j; ++h)
                if (ids[h] == ids[j]) {
                    snprintf(error, sizeof(error), "duplicate result ID");
                    goto cleanup;
                }
        }
    }
    verification_ms = clock_ms() - started;
    printf("Timing (ms): build=%.6f search=%.6f build+search=%.6f verification=%.6f\n", build_ms,
           search_ms, build_ms + search_ms, verification_ms);
    puts("Timing excludes input loading, console printing, and cleanup; verification is separate.");
    printf("Memory (bytes): device=%zu pinned_host_tiles=%zu\n",
           grid ? grid_stats.device_bytes
           : rt ? stats.executor_bytes
                : 0,
           rt ? stats.host_bytes : 0);
    puts("Memory covers explicit device search buffers and pinned tile storage; excludes context, "
         "construction, other host memory, and verification allocations.");
    if (rt)
        printf("RT tiles: %zu | tile loads: %zu | executor bytes: %zu\n", stats.tiles, stats.loads,
               stats.executor_bytes);
    if (rt && show_tiles(rt, &queries)) {
        snprintf(error, sizeof(error), "cannot inspect tile structure");
        goto cleanup;
    }
    puts("Query results (IDs sorted for display; truncated subsets may differ):");
    for (i = 0; i < queries.count; ++i) {
        size_t returned = result.counts[i] < k ? result.counts[i] : k;
        uint32_t *ids = k ? result.neighbors + i * k : NULL;
        if (returned > 1)
            qsort(ids, returned, sizeof(*ids), compare_id);
        printf("  Q%zu (%g, %g, %g): count=%u IDs=[", i, queries.data[i].x, queries.data[i].y,
               queries.data[i].z, result.counts[i]);
        for (j = 0; j < returned; ++j)
            printf("%s%u", j ? ", " : "", ids[j]);
        printf("] truncated=%s\n", result.counts[i] > k ? "yes" : "no");
    }
    printf("PASS: all %zu query counts and returned IDs verified against brute force.\n",
           queries.count);
    if (largest_count > k)
        printf("To see all IDs, rerun with max_ids at least %u (demo limit: 1024).\n",
               largest_count);
    status = 0;
cleanup:
    if (status)
        fprintf(stderr, "Demo failed: %s\n", error);
    search_result_free(&reference);
    search_result_free(&result);
    stream_free(rt);
    cuda_grid_free(grid);
    cpu_grid_free(cpu);
    pointset_free(&queries);
    pointset_free(&points);
    return status;
}
