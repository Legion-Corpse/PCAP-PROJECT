#include "common.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char error[512];
typedef struct ReferenceMorton {
    uint64_t key;
    size_t position;
    Point point;
} ReferenceMorton;

static int reference_morton_compare(const void *a, const void *b) {
    const ReferenceMorton *x = (const ReferenceMorton *)a, *y = (const ReferenceMorton *)b;
    if (x->key != y->key)
        return x->key < y->key ? -1 : 1;
    if (x->point.id != y->point.id)
        return x->point.id < y->point.id ? -1 : 1;
    return x->position < y->position ? -1 : x->position > y->position;
}

/* Independent bit-by-bit reference, retaining original position as the final
   tie-break so duplicate IDs cannot conceal a changed or missing point. */
static void check_morton_reference(const Point *input, size_t count) {
    PointSet sorted = {0};
    ReferenceMorton *reference;
    double lo[3] = {DBL_MAX, DBL_MAX, DBL_MAX}, hi[3] = {-DBL_MAX, -DBL_MAX, -DBL_MAX};
    size_t i;
    int axis;
    unsigned bit;
    assert(pointset_alloc(&sorted, count, error, sizeof error) == 0);
    if (!count) {
        assert(pointset_morton_sort(&sorted, error, sizeof error) == 0);
        return;
    }
    memcpy(sorted.data, input, count * sizeof(Point));
    reference = (ReferenceMorton *)malloc(count * sizeof(*reference));
    assert(reference);
    for (i = 0; i < count; ++i) {
        const double value[3] = {input[i].x, input[i].y, input[i].z};
        for (axis = 0; axis < 3; ++axis) {
            if (value[axis] < lo[axis])
                lo[axis] = value[axis];
            if (value[axis] > hi[axis])
                hi[axis] = value[axis];
        }
    }
    for (i = 0; i < count; ++i) {
        const double value[3] = {input[i].x, input[i].y, input[i].z};
        uint32_t cell[3];
        uint64_t key = 0;
        for (axis = 0; axis < 3; ++axis)
            cell[axis] =
                hi[axis] == lo[axis]
                    ? 0
                    : (uint32_t)(((value[axis] - lo[axis]) / (hi[axis] - lo[axis])) * 2097151.0);
        for (bit = 0; bit < 21; ++bit)
            for (axis = 0; axis < 3; ++axis)
                key |= (uint64_t)((cell[axis] >> bit) & 1u) << (3 * bit + axis);
        reference[i].key = key;
        reference[i].position = i;
        reference[i].point = input[i];
    }
    qsort(reference, count, sizeof(*reference), reference_morton_compare);
    assert(pointset_morton_sort(&sorted, error, sizeof error) == 0);
    for (i = 0; i < count; ++i)
        if (memcmp(&sorted.data[i], &reference[i].point, sizeof(Point))) {
            fprintf(stderr, "Morton reference mismatch at sorted index %zu of %zu\n", i, count);
            exit(EXIT_FAILURE);
        }
    assert(pointset_morton_sort(&sorted, error, sizeof error) == 0);
    for (i = 0; i < count; ++i)
        assert(memcmp(&sorted.data[i], &reference[i].point, sizeof(Point)) == 0);
    free(reference);
    pointset_free(&sorted);
}

static void test_morton_sort_order_and_permutation(void) {
    Point ties[4097],
        extreme[] = {{FLT_MAX, 0, 0, 8}, {-FLT_MAX, 1, 0, 7}, {0, 0, 0, 6}, {0, -1, 0, 5}};
    const char *specs[] = {"uniform:4097", "clustered:4097:3", "scene:4097"};
    size_t i, j;
    check_morton_reference(NULL, 0);
    check_morton_reference(extreme, 1);
    check_morton_reference(extreme, 4);
    /* Thousands of distinct records share one quantized key and repeated IDs. */
    for (i = 0; i < 4097; ++i) {
        ties[i].x = 0;
        ties[i].y = (float)i * 1e-12f;
        ties[i].z = 0;
        ties[i].id = (uint32_t)((i * 13) % 17);
    }
    ties[4096].x = 1;
    ties[4096].y = 1;
    ties[4096].z = 1;
    check_morton_reference(ties, 4097);
    for (i = 0; i < 3; ++i) {
        PointSet points = {0};
        assert(dataset_make(specs[i], 81, &points, error, sizeof error) == 0);
        for (j = 0; j < points.count; ++j)
            points.data[j].id = (uint32_t)((points.count - j) % 97);
        check_morton_reference(points.data, points.count);
        pointset_free(&points);
    }
}

static void compare(const PointSet *p, const PointSet *q, float radius) {
    SearchResult brute = {0}, grid = {0};
    size_t i, j;
    assert(cpu_brute(p, q, radius, 2, &brute, error, sizeof error) == 0);
    assert(cpu_grid_search(p, q, radius, 2, &grid, error, sizeof error) == 0);
    assert(grid.count == q->count);
    for (i = 0; i < q->count; ++i) {
        assert(grid.counts[i] == brute.counts[i]);
        for (j = 0; j < grid.counts[i] && j < 2; ++j) {
            size_t k;
            for (k = 0; k < p->count && p->data[k].id != grid.neighbors[i * 2 + j]; ++k) {
            }
            assert(k < p->count);
            {
                float dx = p->data[k].x - q->data[i].x, dy = p->data[k].y - q->data[i].y,
                      dz = p->data[k].z - q->data[i].z;
                float xy = dx * dx + dy * dy;
                assert(xy + dz * dz <= radius * radius);
            }
            if (j)
                assert(grid.neighbors[i * 2] != grid.neighbors[i * 2 + j]);
        }
    }
    search_result_free(&brute);
    search_result_free(&grid);
}
static void fixture(const char *path, const char *text) {
    FILE *f = fopen(path, "wb");
    assert(f);
    assert(fwrite(text, 1, strlen(text), f) == strlen(text));
    assert(fclose(f) == 0);
}
int main(int argc, char **argv) {
    Point raw[] = {{0, 0, 0, 11}, {0, 0, 0, 12}, {1, 0, 0, 13}, {0, 1, 0, 14}, {3, 3, 3, 15}};
    Point query[] = {{0, 0, 0, 0}, {-1, 0, 0, 1}, {100, 100, 100, 2}};
    PointSet p = {raw, 5}, q = {query, 3}, a = {0}, b = {0};
    SearchResult result = {0};
    CpuGrid *grid = NULL;
    size_t n, i;
    char path[1024];
    const char *directory = argc > 1 ? argv[1] : ".";
    const char *kinds[] = {"uniform:200", "clustered:200:4", "scene:200"};
    test_morton_sort_order_and_permutation();
    compare(&p, &q, 1.0f);
    {
        Point edge[] = {{nextafterf(1.0f, 0.0f), 0, 0, 0},
                        {1, 0, 0, 1},
                        {nextafterf(1.0f, 2.0f), 0, 0, 2},
                        {FLT_MAX, 0, 0, 3},
                        {-FLT_MAX, 0, 0, 4}};
        Point tiny[] = {{0, 0, 0, 0}, {1e-23f, 0, 0, 1}};
        PointSet e = {edge, 5}, t = {tiny, 2};
        compare(&e, &q, 1.0f);
        assert(cpu_brute(&t, &t, nextafterf(0.0f, 1.0f), 0, &result, error, sizeof error) != 0);
        assert(cpu_brute(&t, &t, 1e-20f, 0, &result, error, sizeof error) != 0);
        assert(cpu_grid_build(&t, 1e-30f, &grid, error, sizeof error) != 0);
        compare(&t, &t, sqrtf(FLT_MIN));
    }
    assert(cpu_brute(&p, &q, 1.0f, 1, &result, error, sizeof error) == 0);
    assert(result.counts[0] == 4 && result.counts[1] == 2 && result.counts[2] == 0);
    assert(result.neighbors[0] == 11);
    search_result_free(&result);
    assert(cpu_grid_build(&p, 1.0f, &grid, error, sizeof error) == 0);
    assert(cpu_grid_query(grid, &q, 0, &result, error, sizeof error) == 0);
    assert(result.neighbors == NULL && result.counts[0] == 4);
    search_result_free(&result);
    cpu_grid_free(grid);
    assert(cpu_brute(&p, &q, NAN, 0, &result, error, sizeof error) != 0);
    assert(pointset_alloc(&a, SIZE_MAX, error, sizeof error) != 0);
    assert(search_result_alloc(&result, SIZE_MAX, UINT32_MAX, error, sizeof error) != 0);
    assert(parse_count("0.2K", &n, error, sizeof error) == 0 && n == 200);
    assert(parse_count("2oops", &n, error, sizeof error) != 0);
    assert(parse_count("-1", &n, error, sizeof error) != 0);
    assert(parse_count("nan", &n, error, sizeof error) != 0);
    assert(parse_count("18446744073709551616", &n, error, sizeof error) != 0);
    for (i = 0; i < 3; ++i) {
        assert(dataset_make(kinds[i], 123, &a, error, sizeof error) == 0);
        assert(dataset_make(kinds[i], 123, &b, error, sizeof error) == 0);
        assert(a.count == 200 && memcmp(a.data, b.data, 200 * sizeof(Point)) == 0);
        compare(&a, &b, 0.15f);
        assert(pointset_morton_sort(&a, error, sizeof error) == 0);
        {
            size_t j;
            for (j = 0; j < a.count; ++j) {
                assert(a.data[j].id < b.count);
                assert(memcmp(&a.data[j], &b.data[a.data[j].id], sizeof(Point)) == 0);
            }
        }
        compare(&a, &b, 0.15f);
        pointset_free(&a);
        pointset_free(&b);
    }
    snprintf(path, sizeof path, "%s/core_points.xyz", directory);
    fixture(path, "# example\n0,0,0\n1 0 0\n");
    assert(pointset_load(path, &a, error, sizeof error) == 0 && a.count == 2 && a.data[1].id == 1);
    pointset_free(&a);
    fixture(path, "0 0 nan\n");
    assert(pointset_load(path, &a, error, sizeof error) != 0);
    remove(path);
    snprintf(path, sizeof path, "%s/core_points.ply", directory);
    fixture(path, "ply\nformat ascii 1.0\nelement vertex 2\nproperty uchar red\nproperty float "
                  "z\nproperty double x\nproperty float y\nend_header\n255 3 1 2\n0 6 4 5\n");
    assert(pointset_load(path, &a, error, sizeof error) == 0 && a.count == 2);
    assert(a.data[0].x == 1 && a.data[0].y == 2 && a.data[0].z == 3);
    pointset_free(&a);
    {
        FILE *f = fopen(path, "wb");
        const unsigned char bytes[] = {255, 0, 0, 64, 64, 1, 0, 2, 0};
        assert(f);
        fputs("ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty uchar "
              "red\nproperty float z\nproperty short x\nproperty ushort y\nend_header\n",
              f);
        assert(fwrite(bytes, 1, sizeof bytes, f) == sizeof bytes);
        fclose(f);
        assert(pointset_load(path, &a, error, sizeof error) == 0 && a.count == 1);
        assert(a.data[0].x == 1 && a.data[0].y == 2 && a.data[0].z == 3);
        pointset_free(&a);
    }
    fixture(path, "ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float "
                  "y\nproperty float z\nend_header\n1 2 3\n");
    assert(pointset_load(path, &a, error, sizeof error) != 0);
    remove(path);
    snprintf(path, sizeof path, "%s/core_points.bin", directory);
    assert(pointset_save_bin(path, &p, error, sizeof error) == 0);
    assert(pointset_load(path, &a, error, sizeof error) == 0 && a.count == 5 && a.data[2].x == 1);
    pointset_free(&a);
    fixture(path, "bad");
    assert(pointset_load(path, &a, error, sizeof error) != 0);
    remove(path);
    puts("C17 core: dataset I/O, deterministic generation, Morton IDs, exact brute/grid and "
         "overflow checks passed");
    return 0;
}
