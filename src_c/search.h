#ifndef PCAP_SEARCH_H
#define PCAP_SEARCH_H
#include "common.h"
typedef struct StreamIndex StreamIndex;
/* Read-only view into an index's actual pinned tile. Valid until stream_free;
   excludes padded bucket entries. The caller must not modify or free points. */
typedef struct StreamTileView {
    const Point *points;
    size_t count, bytes;
    Bounds bounds;
} StreamTileView;
typedef struct StreamOptions {
    size_t tile_points, budget_bytes, window_queries;
    unsigned group, slots;
    int overlap, resident;
} StreamOptions;
typedef struct StreamStats {
    double build_ms, search_ms, sort_ms, h2d_bytes, calibration_ms, predicted_ms;
    size_t tiles, loads, executor_bytes, host_bytes, windows, selected_tile_points;
} StreamStats;
double clock_ms(void);
int stream_build(const PointSet *, float, const StreamOptions *, StreamIndex **, StreamStats *,
                 char *, size_t);
int stream_query(StreamIndex *, const PointSet *, unsigned, SearchResult *, StreamStats *, char *,
                 size_t);
size_t stream_tile_count(const StreamIndex *);
int stream_tile_view(const StreamIndex *, size_t tile, StreamTileView *);
/* Same conservative routing test as execution: 0/1, or -1 for an invalid tile.
   A candidate tile need not contain an accepted point for this query. */
int stream_tile_routes_query(const StreamIndex *, size_t tile, Point query);
void stream_free(StreamIndex *);
#endif
