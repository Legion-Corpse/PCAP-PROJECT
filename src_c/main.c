#include "common.h"
#include "search.h"
#include "grid_backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <ctype.h>
#include <errno.h>
#include <float.h>

static const char *option(int argc, char **argv, const char *key, const char *fallback) {
    int i;
    const char *value = fallback;
    for (i = 1; i < argc; i += 2)
        if (!strcmp(argv[i], key))
            value = i + 1 < argc ? argv[i + 1] : NULL;
    return value;
}
static int number(const char *s, double *out) {
    char *end;
    const char *p;
    if (!s || !*s)
        return -1;
    for (p = s; *p; ++p)
        if (isspace((unsigned char)*p))
            return -1;
    errno = 0;
    *out = strtod(s, &end);
    return end == s || *end || errno || !isfinite(*out) ? -1 : 0;
}
static int checked_size(int argc, char **argv, const char *key, const char *value, size_t *out,
                        char *e) {
    const char *s = option(argc, argv, key, value);
    char reason[400];
    if (parse_count(s ? s : "", out, reason, sizeof(reason))) {
        snprintf(e, 512, "%s: %s", key, reason);
        return -1;
    }
    return 0;
}
static int radius_ok(double radius) {
    float r;
    if (!isfinite(radius) || radius <= 0 || radius > FLT_MAX)
        return 0;
    r = (float)radius;
    return r > 0 && isfinite(r * r) && r * r >= FLT_MIN;
}
static int methods_ok(const char *s, char *error, size_t size) {
    const char *p = s;
    size_t length;
    if (!s || !*s) {
        snprintf(error, size, "empty method list");
        return 0;
    }
    while (*p) {
        const char *end = strchr(p, ',');
        length = end ? (size_t)(end - p) : strlen(p);
        if (!((length == 5 && !strncmp(p, "brute", length)) ||
              (length == 8 && !strncmp(p, "cpu-grid", length)) ||
              (length == 9 && !strncmp(p, "cuda-grid", length)) ||
              (length == 9 && !strncmp(p, "rt-incore", length)) ||
              (length == 6 && !strncmp(p, "rt-ooc", length)))) {
            snprintf(error, size, "unknown method in list: %s", s);
            return 0;
        }
        if (!end)
            return 1;
        p = end + 1;
        if (!*p) {
            snprintf(error, size, "empty method in list: %s", s);
            return 0;
        }
    }
    return 1;
}
static int verify(const PointSet *p, const PointSet *q, float r, unsigned K,
                  const SearchResult *got, const char *mode, size_t limit, char *err, size_t es,
                  size_t *checked) {
    PointSet sample = {0};
    SearchResult ref = {0};
    size_t i, j, k, n = !strcmp(mode, "full") ? q->count : (q->count < limit ? q->count : limit);
    int rc = -1;
    *checked = 0;
    if (!strcmp(mode, "none"))
        return 0;
    if (pointset_alloc(&sample, n, err, es))
        return -1;
    for (i = 0; i < n; i++)
        sample.data[i] = q->data[!strcmp(mode, "full") ? i : (i * q->count / n)];
    if (cpu_grid_search(p, &sample, r, 0, &ref, err, es))
        goto done;
    for (i = 0; i < n; i++) {
        size_t qi = !strcmp(mode, "full") ? i : (i * q->count / n);
        if (got->counts[qi] != ref.counts[i]) {
            snprintf(err, es, "count mismatch at query %zu", qi);
            goto done;
        }
        for (j = 0; j < K && j < got->counts[qi]; j++) {
            unsigned id = got->neighbors[qi * K + j];
            float dx, dy, dz, d2;
            if (id >= p->count) {
                snprintf(err, es, "neighbor identifier out of range");
                goto done;
            }
            dx = p->data[id].x - q->data[qi].x;
            dy = p->data[id].y - q->data[qi].y;
            dz = p->data[id].z - q->data[qi].z;
            d2 = (dx * dx + dy * dy) + dz * dz;
            if (d2 > r * r) {
                snprintf(err, es, "invalid neighbor membership");
                goto done;
            }
            for (k = 0; k < j; k++)
                if (got->neighbors[qi * K + k] == id) {
                    snprintf(err, es, "duplicate neighbor id");
                    goto done;
                }
        }
    }
    *checked = n;
    rc = 0;
done:
    search_result_free(&ref);
    pointset_free(&sample);
    return rc;
}
int main(int argc, char **argv) {
    const char *ps = option(argc, argv, "--points", "uniform:100K"),
               *qs = option(argc, argv, "--queries", "same"),
               *modes = option(argc, argv, "--method", "rt-ooc"),
               *vm = option(argc, argv, "--verify", "sample"),
               *json = option(argc, argv, "--json", "");
    PointSet p = {0}, q = {0};
    Bounds bounds;
    size_t repeat = 1, K = 0, group = 8, tile = 0, budget = 64, window = 0, samples = 512, seed = 1,
           slots = 2, i;
    char error[512] = {0}, methods[256], *token;
    double radius = 0, expected = 32;
    int result = 1, alias = 0, explicit_radius = 0;
    FILE *file = NULL;
    StreamIndex *index = NULL;
    CpuGrid *cpu = NULL;
    CudaGrid *grid = NULL;
    SearchResult output = {0};
    if (argc == 2 && !strcmp(argv[1], "--help")) {
        puts("C17 RT-core fixed-radius search\n--points "
             "uniform:100K|clustered:100K:5|scene:100K|file:path\n--queries same|dataset --method "
             "brute,cpu-grid,cuda-grid,rt-incore,rt-ooc\n--radius R (R*R must be finite normal "
             "float) | --k-expected 32 --group 8 --tile-points 0 (automatic)\n--budget-mb 64 "
             "(positive integer MiB, executor cap only) --slots 1|2 --overlap on|off\n--window N "
             "--max-neighbors K --repeat N (index reused)\n--verify full|sample|none "
             "--verify-samples 512 --seed 1 --json PATH\nUnknown options are errors; repeated "
             "options use their last value.");
        return 0;
    }
    if (!ps || !qs || !modes || !vm || !json) {
        fprintf(stderr, "missing option value\n");
        return 2;
    }
    {
        const char *keys[] = {"--points",     "--queries", "--method",         "--radius",
                              "--k-expected", "--group",   "--tile-points",    "--budget-mb",
                              "--slots",      "--overlap", "--window",         "--max-neighbors",
                              "--repeat",     "--verify",  "--verify-samples", "--seed",
                              "--json"};
        int a;
        for (a = 1; a < argc; a += 2) {
            size_t j;
            int known = 0;
            for (j = 0; j < sizeof(keys) / sizeof(keys[0]); j++)
                if (!strcmp(argv[a], keys[j]))
                    known = 1;
            if (!known || a + 1 >= argc || !strncmp(argv[a + 1], "--", 2)) {
                fprintf(stderr, "error: unknown option or missing value: %s\n", argv[a]);
                return 2;
            }
        }
    }
#define INTOPT(k, d, v)                                                                            \
    if (checked_size(argc, argv, k, d, &v, error))                                                 \
    goto fail
    INTOPT("--repeat", "1", repeat);
    INTOPT("--max-neighbors", "0", K);
    INTOPT("--group", "8", group);
    INTOPT("--tile-points", "0", tile);
    INTOPT("--budget-mb", "64", budget);
    INTOPT("--window", "0", window);
    INTOPT("--verify-samples", "512", samples);
    INTOPT("--seed", "1", seed);
    INTOPT("--slots", "2", slots);
    if (!repeat || repeat > 1000 || !group || group > 64 || tile > UINT_MAX || K > 65536 ||
        !budget || budget > SIZE_MAX / 1048576 || slots < 1 || slots > 2 || !samples) {
        snprintf(error, sizeof(error), "invalid numeric option range");
        goto fail;
    }
    if (strcmp(vm, "full") && strcmp(vm, "sample") && strcmp(vm, "none")) {
        snprintf(error, sizeof(error), "unknown verification mode");
        goto fail;
    }
    if (strcmp(option(argc, argv, "--overlap", "on"), "on") &&
        strcmp(option(argc, argv, "--overlap", "on"), "off")) {
        snprintf(error, sizeof(error), "invalid overlap mode");
        goto fail;
    }
    if (number(option(argc, argv, "--k-expected", "32"), &expected) || expected <= 0) {
        snprintf(error, sizeof(error), "invalid expected neighbor count");
        goto fail;
    }
    explicit_radius = option(argc, argv, "--radius", NULL) != NULL;
    if (explicit_radius &&
        (number(option(argc, argv, "--radius", NULL), &radius) || !radius_ok(radius))) {
        snprintf(error, sizeof(error), "radius must be positive with a finite normal float square");
        goto fail;
    }
    if (strlen(modes) >= sizeof(methods)) {
        snprintf(error, sizeof(error), "method list too long");
        goto fail;
    }
    if (!methods_ok(modes, error, sizeof(error)))
        goto fail;
    if (*json) {
        file = fopen(json, "a");
        if (!file) {
            snprintf(error, sizeof(error), "cannot open JSON output");
            goto fail;
        }
    }
    if (dataset_make(ps, seed, &p, error, sizeof(error)))
        goto fail;
    if (!strcmp(qs, "same")) {
        q = p;
        alias = 1;
    } else if (dataset_make(qs, seed + 1, &q, error, sizeof(error)))
        goto fail;
    if (!p.count || !q.count || p.count > UINT_MAX || q.count > UINT_MAX) {
        snprintf(error, sizeof(error), "empty or oversized point set");
        goto fail;
    }
    if (pointset_bounds(&p, &bounds, error, sizeof(error)))
        goto fail;
    if (!explicit_radius)
        radius = radius_for_expected(&bounds, p.count, expected);
    if (!radius_ok(radius)) {
        snprintf(error, sizeof(error), "radius must be positive with a finite normal float square");
        goto fail;
    }
    strcpy(methods, modes);
    token = strtok(methods, ",");
    if (!token) {
        snprintf(error, sizeof(error), "empty method list");
        goto fail;
    }
    while (token) {
        StreamStats build = {0};
        CudaGridStats gs = {0};
        StreamOptions config = {tile,
                                budget * 1048576,
                                window,
                                (unsigned)group,
                                (unsigned)slots,
                                strcmp(option(argc, argv, "--overlap", "on"), "off") != 0,
                                0};
        double bstart = clock_ms(), buildms = 0;
        int rt = !strcmp(token, "rt-ooc") || !strcmp(token, "rt-incore");
        if (rt) {
            if (!strcmp(token, "rt-incore")) {
                config.tile_points = p.count;
                config.budget_bytes = (size_t)6 << 30;
                config.slots = 1;
                config.resident = 1;
            }
            if (stream_build(&p, (float)radius, &config, &index, &build, error, sizeof(error)))
                goto fail;
            buildms = build.build_ms;
        } else if (!strcmp(token, "cpu-grid")) {
            if (cpu_grid_build(&p, (float)radius, &cpu, error, sizeof(error)))
                goto fail;
            buildms = clock_ms() - bstart;
        } else if (!strcmp(token, "cuda-grid")) {
            if (cuda_grid_build(&p, (float)radius, &grid, &gs, error, sizeof(error)))
                goto fail;
            buildms = gs.build_ms;
        } else if (strcmp(token, "brute") && strcmp(token, "cuda-grid")) {
            snprintf(error, sizeof(error), "unknown method: %s", token);
            goto fail;
        }
        for (i = 0; i < repeat; i++) {
            StreamStats stats = build;
            double start = clock_ms(), searchms;
            size_t checked = 0, j;
            unsigned long long total = 0;
            int rc;
            if (rt)
                rc = stream_query(index, &q, (unsigned)K, &output, &stats, error, sizeof(error));
            else if (cpu)
                rc = cpu_grid_query(cpu, &q, (unsigned)K, &output, error, sizeof(error));
            else if (!strcmp(token, "brute"))
                rc = cpu_brute(&p, &q, (float)radius, (unsigned)K, &output, error, sizeof(error));
            else
                rc = cuda_grid_query(grid, &q, (unsigned)K, &output, &gs, error, sizeof(error));
            searchms = clock_ms() - start;
            if (grid)
                stats.executor_bytes = gs.device_bytes;
            double first_answer_ms = clock_ms() - bstart;
            char cold_json[64];
            if (i == 0)
                snprintf(cold_json, sizeof(cold_json), "%.6f", first_answer_ms);
            else
                strcpy(cold_json, "null");
            if (rc || verify(&p, &q, (float)radius, (unsigned)K, &output, vm, samples, error,
                             sizeof(error), &checked))
                goto fail;
            for (j = 0; j < q.count; j++)
                total += output.counts[j];
            printf("[%s] repeat=%zu build=%.3f ms search=%.3f ms tiles=%zu loads=%zu "
                   "executor=%.3f MiB pinned_tiles=%.3f MiB verify=%s checked=%zu\n",
                   token, i, buildms, searchms, build.tiles, stats.loads,
                   stats.executor_bytes / 1048576.0, build.host_bytes / 1048576.0, vm, checked);
            if (file) {
                fprintf(
                    file,
                    "{\"implementation\":\"c17\",\"method\":\"%s\",\"n_points\":%zu,\"n_queries\":%"
                    "zu,\"repeat\":%zu,\"radius\":%.9g,\"group\":%zu,\"max_neighbors\":%zu,\"index_"
                    "ms\":%.6f,\"search_ms\":%.6f,\"phase_total_ms\":%.6f,\"first_answer_wall_ms\":"
                    "%s,\"calibration_ms\":%.6f,\"predicted_ms\":%.6f,\"selected_tile_points\":%zu,"
                    "\"tiles\":%zu,\"loads\":%zu,\"executor_bytes\":%zu,\"host_tile_bytes\":%zu,"
                    "\"neighbors_total\":%llu,\"verify\":\"%s\",\"verify_checked\":%zu,\"verify_"
                    "count_mismatches\":0,\"verify_list_errors\":0,\"status\":\"ok\"}\n",
                    token, p.count, q.count, i, radius, group, K, buildms, searchms,
                    buildms + searchms, cold_json, build.calibration_ms, build.predicted_ms,
                    build.selected_tile_points, build.tiles, stats.loads, stats.executor_bytes,
                    build.host_bytes, total, vm, checked);
                if (fflush(file) || ferror(file)) {
                    snprintf(error, sizeof(error), "JSON write failed");
                    goto fail;
                }
            }
            search_result_free(&output);
        }
        stream_free(index);
        index = NULL;
        cpu_grid_free(cpu);
        cpu = NULL;
        cuda_grid_free(grid);
        grid = NULL;
        token = strtok(NULL, ",");
    }
    result = 0;
fail:
    search_result_free(&output);
    stream_free(index);
    cpu_grid_free(cpu);
    cuda_grid_free(grid);
    if (result)
        fprintf(stderr, "error: %s\n", error);
    if (file && fclose(file)) {
        fprintf(stderr, "error: JSON close failed\n");
        result = 1;
    }
    if (!alias)
        pointset_free(&q);
    pointset_free(&p);
    return result;
}
