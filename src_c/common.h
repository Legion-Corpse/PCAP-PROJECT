#ifndef PCAP_COMMON_H
#define PCAP_COMMON_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* All owning outputs must be zero-initialized; free with the matching function.
   Functions return 0 on success, -1 with a diagnostic in error on failure.
   Search radii must be positive and their float square finite and normal;
   smaller squares make radius candidate filters incompatible with float
   distance underflow. Counts include the query point itself when present. */
typedef struct Point {
    float x, y, z;
    uint32_t id;
} Point;
typedef struct PointSet {
    Point *data;
    size_t count;
} PointSet;
typedef struct Bounds {
    float lo[3], hi[3];
} Bounds;
typedef struct SearchResult {
    uint32_t *counts;
    uint32_t *neighbors; /* query-major, at most max_neighbors IDs per query */
    size_t count;
    uint32_t max_neighbors;
} SearchResult;
typedef struct CpuGrid CpuGrid;

int pointset_alloc(PointSet *out, size_t count, char *error, size_t error_size);
void pointset_free(PointSet *set);
int pointset_load(const char *path, PointSet *out, char *error, size_t error_size);
int pointset_save_bin(const char *path, const PointSet *set, char *error, size_t error_size);
int dataset_make(const char *spec, uint64_t seed, PointSet *out, char *error, size_t error_size);
int parse_count(const char *text, size_t *out, char *error, size_t error_size);
int pointset_bounds(const PointSet *set, Bounds *bounds, char *error, size_t error_size);
float radius_for_expected(const Bounds *bounds, size_t count, double expected);
int pointset_morton_sort(PointSet *set, char *error, size_t error_size);
int search_result_alloc(SearchResult *out, size_t count, uint32_t max_neighbors, char *error,
                        size_t error_size);
void search_result_free(SearchResult *out);
int cpu_brute(const PointSet *points, const PointSet *queries, float radius, uint32_t max_neighbors,
              SearchResult *out, char *error, size_t error_size);
int cpu_grid_build(const PointSet *points, float radius, CpuGrid **out, char *error,
                   size_t error_size);
int cpu_grid_query(const CpuGrid *grid, const PointSet *queries, uint32_t max_neighbors,
                   SearchResult *out, char *error, size_t error_size);
void cpu_grid_free(CpuGrid *grid);
int cpu_grid_search(const PointSet *points, const PointSet *queries, float radius,
                    uint32_t max_neighbors, SearchResult *out, char *error, size_t error_size);

#ifdef __cplusplus
}
#endif
#endif
