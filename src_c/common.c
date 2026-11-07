#include "common.h"

#include <ctype.h>
#include <errno.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail(char *error, size_t size, const char *fmt, ...) {
    va_list args;
    if (error && size) {
        va_start(args, fmt);
        vsnprintf(error, size, fmt, args);
        va_end(args);
    }
    return -1;
}
static int finite_point(Point p) { return isfinite(p.x) && isfinite(p.y) && isfinite(p.z); }
static int valid_set(const PointSet *s, char *error, size_t size) {
    size_t i;
    if (!s || (s->count && !s->data) || s->count > UINT32_MAX)
        return fail(error, size, "invalid point set or point count exceeds uint32 IDs");
    for (i = 0; i < s->count; ++i)
        if (!finite_point(s->data[i]))
            return fail(error, size, "point coordinates must be finite");
    return 0;
}
int pointset_alloc(PointSet *out, size_t count, char *error, size_t size) {
    if (!out || out->data || count > UINT32_MAX || count > SIZE_MAX / sizeof(Point))
        return fail(error, size, "point allocation overflow or nonempty output");
    out->data = count ? (Point *)calloc(count, sizeof(Point)) : NULL;
    if (count && !out->data)
        return fail(error, size, "point allocation failed");
    out->count = count;
    return 0;
}
void pointset_free(PointSet *s) {
    if (s) {
        free(s->data);
        memset(s, 0, sizeof *s);
    }
}
void search_result_free(SearchResult *s) {
    if (s) {
        free(s->counts);
        free(s->neighbors);
        memset(s, 0, sizeof *s);
    }
}
int search_result_alloc(SearchResult *out, size_t n, uint32_t k, char *error, size_t size) {
    if (!out || out->counts || out->neighbors || n > SIZE_MAX / sizeof(uint32_t) ||
        (k && n > SIZE_MAX / sizeof(uint32_t) / k))
        return fail(error, size, "search result allocation overflow or nonempty output");
    out->counts = n ? (uint32_t *)calloc(n, sizeof(uint32_t)) : NULL;
    out->neighbors = n && k ? (uint32_t *)malloc(n * k * sizeof(uint32_t)) : NULL;
    if ((n && !out->counts) || (n && k && !out->neighbors)) {
        search_result_free(out);
        return fail(error, size, "search result allocation failed");
    }
    out->count = n;
    out->max_neighbors = k;
    return 0;
}
int parse_count(const char *text, size_t *out, char *error, size_t size) {
    char *end;
    double v, mult = 1;
    const char *p;
    if (!text || !*text || !out || *text == '-')
        return fail(error, size, "invalid count");
    for (p = text; *p; ++p)
        if (isspace((unsigned char)*p))
            return fail(error, size, "invalid count: %s", text);
    errno = 0;
    v = strtod(text, &end);
    if (end == text || errno || !isfinite(v) || v < 0)
        return fail(error, size, "invalid count: %s", text);
    if (*end) {
        if (end[1])
            return fail(error, size, "invalid count: %s", text);
        switch (toupper((unsigned char)*end)) {
        case 'K':
            mult = 1e3;
            break;
        case 'M':
            mult = 1e6;
            break;
        case 'G':
            mult = 1e9;
            break;
        default:
            return fail(error, size, "invalid count: %s", text);
        }
    }
    v *= mult;
    if (!isfinite(v) || v >= ldexp(1.0, (int)(sizeof(size_t) * CHAR_BIT)) || floor(v) != v)
        return fail(error, size, "count overflow or fractional count: %s", text);
    *out = (size_t)v;
    return 0;
}
int pointset_bounds(const PointSet *s, Bounds *b, char *error, size_t size) {
    size_t i;
    int j;
    if (!b)
        return fail(error, size, "missing bounds output");
    if (valid_set(s, error, size))
        return -1;
    if (!s->count)
        return fail(error, size, "empty point set has no bounds");
    for (j = 0; j < 3; ++j) {
        b->lo[j] = FLT_MAX;
        b->hi[j] = -FLT_MAX;
    }
    for (i = 0; i < s->count; ++i) {
        const float v[3] = {s->data[i].x, s->data[i].y, s->data[i].z};
        for (j = 0; j < 3; ++j) {
            if (v[j] < b->lo[j])
                b->lo[j] = v[j];
            if (v[j] > b->hi[j])
                b->hi[j] = v[j];
        }
    }
    return 0;
}
float radius_for_expected(const Bounds *b, size_t n, double expected) {
    double volume;
    if (!b || !n || !isfinite(expected) || expected <= 0)
        return NAN;
    volume = fmax((double)b->hi[0] - b->lo[0], 1e-6) * fmax((double)b->hi[1] - b->lo[1], 1e-6) *
             fmax((double)b->hi[2] - b->lo[2], 1e-6);
    return (float)cbrt(3.0 * expected * volume / (4.0 * 3.14159265358979323846 * (double)n));
}
static uint64_t mix(uint64_t v) {
    v = (v ^ (v >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    v = (v ^ (v >> 27)) * UINT64_C(0x94d049bb133111eb);
    return v ^ (v >> 31);
}
static uint64_t random_next(uint64_t *s) {
    *s += UINT64_C(0x9e3779b97f4a7c15);
    return mix(*s);
}
static double uniform(uint64_t *s) {
    return (double)(random_next(s) >> 11) * (1.0 / 9007199254740992.0);
}
static double normal(uint64_t *s) {
    double u = fmax(uniform(s), 1e-300);
    return sqrt(-2 * log(u)) * cos(6.283185307179586 * uniform(s));
}
static float clamp(double x) { return (float)fmin(1.0, fmax(0.0, x)); }
int dataset_make(const char *spec, uint64_t seed, PointSet *out, char *error, size_t size) {
    char kind[32], number[96];
    const char *colon, *extra;
    size_t n, i, clusters = 64, len;
    uint64_t rng = mix(seed);
    Point *centers = NULL;
    if (!spec || !out)
        return fail(error, size, "missing dataset spec");
    if (strncmp(spec, "file:", 5) == 0)
        return pointset_load(spec + 5, out, error, size);
    colon = strchr(spec, ':');
    if (!colon || (len = (size_t)(colon - spec)) >= sizeof kind)
        return fail(error, size, "invalid dataset spec: %s", spec);
    for (i = 0; i < len; ++i)
        kind[i] = (char)tolower((unsigned char)spec[i]);
    kind[len] = 0;
    if (strcmp(kind, "uniform") && strcmp(kind, "clustered") && strcmp(kind, "scene"))
        return fail(error, size, "unknown dataset: %s", kind);
    extra = strchr(colon + 1, ':');
    len = extra ? (size_t)(extra - colon - 1) : strlen(colon + 1);
    if (len >= sizeof number)
        return fail(error, size, "invalid dataset count");
    memcpy(number, colon + 1, len);
    number[len] = 0;
    if (parse_count(number, &n, error, size))
        return -1;
    if (extra && (strcmp(kind, "clustered") || parse_count(extra + 1, &clusters, error, size)))
        return fail(error, size, "invalid cluster count");
    if (!clusters || clusters > UINT32_MAX || clusters > SIZE_MAX / sizeof(Point))
        return fail(error, size, "invalid cluster count");
    if (strcmp(kind, "clustered") == 0) {
        centers = (Point *)calloc(clusters, sizeof(Point));
        if (!centers)
            return fail(error, size, "cluster allocation failed");
        for (i = 0; i < clusters; ++i) {
            centers[i].x = (float)(0.1 + 0.8 * uniform(&rng));
            centers[i].y = (float)(0.1 + 0.8 * uniform(&rng));
            centers[i].z = (float)(0.1 + 0.8 * uniform(&rng));
        }
    }
    if (pointset_alloc(out, n, error, size)) {
        free(centers);
        return -1;
    }
    for (i = 0; i < n; ++i) {
        Point p;
        p.x = (float)uniform(&rng);
        p.y = (float)uniform(&rng);
        p.z = (float)uniform(&rng);
        p.id = (uint32_t)i;
        if (centers && uniform(&rng) > 0.05) {
            Point c = centers[random_next(&rng) % clusters];
            p.x = clamp(c.x + 0.025 * normal(&rng));
            p.y = clamp(c.y + 0.025 * normal(&rng));
            p.z = clamp(c.z + 0.025 * normal(&rng));
        } else if (strcmp(kind, "scene") == 0) {
            double t = uniform(&rng);
            if (t < 0.4)
                p.z = clamp(0.0002 * normal(&rng));
            else if (t < 0.8) {
                if (t < 0.6)
                    p.x = (float)(floor(p.x * 8) / 8);
                else
                    p.y = (float)(floor(p.y * 8) / 8);
                p.z *= 0.4f;
            } else {
                double z = 2 * uniform(&rng) - 1, a = 6.283185307179586 * uniform(&rng),
                       r = sqrt(fmax(0, 1 - z * z));
                p.x = (float)(0.5 + 0.2 * r * cos(a));
                p.y = (float)(0.5 + 0.2 * r * sin(a));
                p.z = (float)(0.25 + 0.2 * z);
            }
        }
        out->data[i] = p;
    }
    free(centers);
    return 0;
}

typedef struct MortonEntry {
    uint64_t key;
    uint32_t position, id;
} MortonEntry;

static uint64_t spread_morton_bits(uint32_t coordinate) {
    uint64_t v = coordinate & UINT32_C(0x1fffff);
    v = (v | (v << 32)) & UINT64_C(0x001f00000000ffff);
    v = (v | (v << 16)) & UINT64_C(0x001f0000ff0000ff);
    v = (v | (v << 8)) & UINT64_C(0x100f00f00f00f00f);
    v = (v | (v << 4)) & UINT64_C(0x10c30c30c30c30c3);
    return (v | (v << 2)) & UINT64_C(0x1249249249249249);
}

static uint64_t morton_key(uint32_t x, uint32_t y, uint32_t z) {
    return spread_morton_bits(x) | (spread_morton_bits(y) << 1) | (spread_morton_bits(z) << 2);
}

static int morton_tie_compare(const void *a, const void *b) {
    const MortonEntry *x = (const MortonEntry *)a, *y = (const MortonEntry *)b;
    if (x->id != y->id)
        return x->id < y->id ? -1 : 1;
    return x->position < y->position ? -1 : x->position > y->position;
}

int pointset_morton_sort(PointSet *s, char *error, size_t size) {
    Bounds b;
    MortonEntry *source, *target, *swap;
    size_t i, begin, end;
    unsigned pass;
    int axis;
    if (valid_set(s, error, size))
        return -1;
    if (s->count < 2)
        return 0;
    if (pointset_bounds(s, &b, error, size))
        return -1;
    if (s->count > SIZE_MAX / sizeof(MortonEntry))
        return fail(error, size, "Morton allocation overflow");
    source = (MortonEntry *)malloc(s->count * sizeof(MortonEntry));
    target = (MortonEntry *)malloc(s->count * sizeof(MortonEntry));
    if (!source || !target) {
        free(source);
        free(target);
        return fail(error, size, "Morton allocation failed");
    }
    for (i = 0; i < s->count; ++i) {
        uint32_t c[3];
        const float v[3] = {s->data[i].x, s->data[i].y, s->data[i].z};
        for (axis = 0; axis < 3; ++axis) {
            double extent = (double)b.hi[axis] - b.lo[axis];
            double unit = extent ? ((double)v[axis] - b.lo[axis]) / extent : 0;
            c[axis] = (uint32_t)(fmin(1, fmax(0, unit)) * 2097151);
        }
        source[i].key = morton_key(c[0], c[1], c[2]);
        source[i].position = (uint32_t)i;
        source[i].id = s->data[i].id;
    }
    /* LSD radix passes are stable; records hold indices instead of Point
       copies so both buffers together use 32 bytes per input point. */
    for (pass = 0; pass < 8; ++pass) {
        size_t positions[256] = {0}, offset = 0;
        unsigned bucket, shift = pass * 8;
        for (i = 0; i < s->count; ++i)
            ++positions[(source[i].key >> shift) & 255u];
        for (bucket = 0; bucket < 256; ++bucket) {
            size_t count = positions[bucket];
            positions[bucket] = offset;
            offset += count;
        }
        for (i = 0; i < s->count; ++i)
            target[positions[(source[i].key >> shift) & 255u]++] = source[i];
        swap = source;
        source = target;
        target = swap;
    }
    free(target);
    /* Generated point IDs already follow input order. Sorting only descending
       equal-key runs retains that cheap path and defines arbitrary/duplicate
       ID behavior using original position as the final tie-break. */
    for (begin = 0; begin < s->count; begin = end) {
        int ordered = 1;
        for (end = begin + 1; end < s->count && source[end].key == source[begin].key; ++end)
            if (source[end - 1].id > source[end].id)
                ordered = 0;
        if (!ordered)
            qsort(source + begin, end - begin, sizeof(MortonEntry), morton_tie_compare);
    }
    /* The sorted indices map each destination to its original source. Follow
       disjoint cycles in place, releasing the spare radix buffer first. */
    for (i = 0; i < s->count; ++i) {
        Point saved;
        size_t position = i;
        if (source[i].position == UINT32_MAX)
            continue;
        saved = s->data[i];
        while (source[position].position != i) {
            size_t next = source[position].position;
            s->data[position] = s->data[next];
            source[position].position = UINT32_MAX;
            position = next;
        }
        s->data[position] = saved;
        source[position].position = UINT32_MAX;
    }
    free(source);
    return 0;
}

static uint64_t little_uint(const unsigned char *p, unsigned width) {
    uint64_t v = 0;
    unsigned i;
    for (i = 0; i < width; ++i)
        v |= (uint64_t)p[i] << (i * 8);
    return v;
}
static float little_float(const unsigned char *p) {
    uint32_t bits = (uint32_t)little_uint(p, 4);
    float v;
    memcpy(&v, &bits, 4);
    return v;
}
static int append_point(PointSet *s, size_t *capacity, Point p, char *error, size_t size) {
    if (!finite_point(p))
        return fail(error, size, "point coordinates must be finite");
    if (s->count == UINT32_MAX)
        return fail(error, size, "point count exceeds uint32 IDs");
    if (s->count == *capacity) {
        size_t n = *capacity ? *capacity * 2 : 1024;
        Point *data;
        if (n < *capacity || n > UINT32_MAX)
            n = UINT32_MAX;
        if (n > SIZE_MAX / sizeof(Point))
            return fail(error, size, "point allocation overflow");
        data = (Point *)realloc(s->data, n * sizeof(Point));
        if (!data)
            return fail(error, size, "point allocation failed");
        s->data = data;
        *capacity = n;
    }
    p.id = (uint32_t)s->count;
    s->data[s->count++] = p;
    return 0;
}
static int load_xyz(FILE *f, PointSet *s, char *error, size_t size) {
    char line[8192];
    size_t capacity = 0, line_no = 0;
    while (fgets(line, sizeof line, f)) {
        char *p = line, *end;
        double value[3];
        int i;
        Point point;
        ++line_no;
        if (!strchr(line, '\n') && !feof(f))
            return fail(error, size, "XYZ line %zu is too long", line_no);
        while (isspace((unsigned char)*p))
            ++p;
        if (!*p || *p == '#')
            continue;
        for (i = 0; i < 3; ++i) {
            while (isspace((unsigned char)*p) || *p == ',' || *p == ';')
                ++p;
            errno = 0;
            value[i] = strtod(p, &end);
            if (end == p || errno || !isfinite(value[i]) || fabs(value[i]) > FLT_MAX)
                return fail(error, size, "invalid or nonfinite XYZ coordinate at line %zu",
                            line_no);
            p = end;
        }
        if (*p && !isspace((unsigned char)*p) && *p != ',' && *p != ';' && *p != '#')
            return fail(error, size, "invalid XYZ row at line %zu", line_no);
        point.x = (float)value[0];
        point.y = (float)value[1];
        point.z = (float)value[2];
        point.id = 0;
        if (append_point(s, &capacity, point, error, size))
            return -1;
    }
    return ferror(f) ? fail(error, size, "XYZ read failed") : 0;
}
typedef struct PlyProperty {
    unsigned width;
    int kind, axis;
} PlyProperty;
static int ply_type(const char *type, PlyProperty *p) {
    const char *names[] = {"char",   "int8",    "uchar",  "uint8",  "short", "int16",
                           "ushort", "uint16",  "int",    "int32",  "uint",  "uint32",
                           "float",  "float32", "double", "float64"};
    const unsigned widths[] = {1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 8, 8};
    size_t i;
    for (i = 0; i < 16; ++i)
        if (!strcmp(type, names[i])) {
            p->width = widths[i];
            p->kind = i >= 12 ? 2 : (i / 2) % 2 == 0 ? 1 : 0;
            return 0;
        }
    return -1;
}
static double ply_scalar(const unsigned char *bytes, const PlyProperty *p) {
    uint64_t u = little_uint(bytes, p->width);
    if (p->kind == 2) {
        if (p->width == 4)
            return little_float(bytes);
        else {
            double d;
            memcpy(&d, &u, 8);
            return d;
        }
    }
    if (p->kind == 1) {
        if (p->width == 1)
            return (int8_t)u;
        if (p->width == 2)
            return (int16_t)u;
        return (int32_t)u;
    }
    return (double)u;
}
static int load_ply(FILE *f, PointSet *s, char *error, size_t size) {
    char line[1024], format[64] = "", element[64] = "";
    PlyProperty props[64];
    size_t n = 0, np = 0, i, j;
    int vertex = 0, ended = 0, axes = 0;
    if (!fgets(line, sizeof line, f) || (strcmp(line, "ply\n") && strcmp(line, "ply\r\n")))
        return fail(error, size, "invalid PLY header");
    while (fgets(line, sizeof line, f)) {
        char type[64], name[64];
        if (sscanf(line, "format %63s", format) == 1)
            continue;
        if (sscanf(line, "element %63s", element) == 1) {
            char number[96];
            size_t count;
            if (sscanf(line, "element %63s %95s", element, number) != 2 ||
                parse_count(number, &count, error, size))
                return fail(error, size, "invalid PLY element count");
            vertex = !strcmp(element, "vertex");
            if (vertex)
                n = count;
            else if (!np && count)
                return fail(error, size, "PLY vertices must precede other nonempty elements");
            continue;
        }
        if (vertex && sscanf(line, "property %63s %63s", type, name) == 2) {
            if (np == 64 || ply_type(type, &props[np]))
                return fail(error, size, "unsupported PLY vertex property: %s", type);
            props[np].axis = !strcmp(name, "x")   ? 0
                             : !strcmp(name, "y") ? 1
                             : !strcmp(name, "z") ? 2
                                                  : -1;
            if (props[np].axis >= 0)
                axes |= 1 << props[np].axis;
            ++np;
            continue;
        }
        if (!strncmp(line, "end_header", 10)) {
            ended = 1;
            break;
        }
    }
    if (!ended || axes != 7 || (strcmp(format, "ascii") && strcmp(format, "binary_little_endian")))
        return fail(error, size, "PLY needs complete ASCII or binary little-endian x/y/z header");
    if (pointset_alloc(s, n, error, size))
        return -1;
    for (i = 0; i < n; ++i) {
        float xyz[3] = {0, 0, 0};
        for (j = 0; j < np; ++j) {
            double v;
            if (!strcmp(format, "ascii")) {
                if (fscanf(f, "%lf", &v) != 1)
                    return fail(error, size, "truncated or malformed PLY vertex %zu", i);
            } else {
                unsigned char bytes[8];
                if (fread(bytes, 1, props[j].width, f) != props[j].width)
                    return fail(error, size, "truncated PLY vertex %zu", i);
                v = ply_scalar(bytes, &props[j]);
            }
            if (props[j].axis >= 0) {
                if (!isfinite(v) || fabs(v) > FLT_MAX)
                    return fail(error, size, "PLY point coordinates must be finite");
                xyz[props[j].axis] = (float)v;
            }
        }
        s->data[i].x = xyz[0];
        s->data[i].y = xyz[1];
        s->data[i].z = xyz[2];
        s->data[i].id = (uint32_t)i;
    }
    return 0;
}
int pointset_load(const char *path, PointSet *out, char *error, size_t size) {
    FILE *f;
    const char *dot;
    int result;
    PointSet temp = {0};
    if (!path || !out || out->data)
        return fail(error, size, "invalid point input or nonempty output");
    f = fopen(path, "rb");
    if (!f)
        return fail(error, size, "cannot open dataset: %s", path);
    dot = strrchr(path, '.');
    if (dot && (!strcmp(dot, ".bin") || !strcmp(dot, ".kitti"))) {
        unsigned char bytes[16];
        size_t got, capacity = 0, stride = !strcmp(dot, ".kitti") ? 16 : 12;
        result = 0;
        while ((got = fread(bytes, 1, stride, f)) != 0) {
            Point p;
            if (got != stride) {
                result = fail(error, size, "binary dataset has incomplete point record");
                break;
            }
            p.x = little_float(bytes);
            p.y = little_float(bytes + 4);
            p.z = little_float(bytes + 8);
            p.id = 0;
            if (append_point(&temp, &capacity, p, error, size)) {
                result = -1;
                break;
            }
        }
        if (ferror(f))
            result = fail(error, size, "binary dataset read failed");
    } else if (dot && !strcmp(dot, ".ply"))
        result = load_ply(f, &temp, error, size);
    else
        result = load_xyz(f, &temp, error, size);
    if (fclose(f) != 0 && !result)
        result = fail(error, size, "dataset close failed");
    if (result) {
        pointset_free(&temp);
        return -1;
    }
    *out = temp;
    return 0;
}
int pointset_save_bin(const char *path, const PointSet *s, char *error, size_t size) {
    FILE *f;
    size_t i;
    int result = 0;
    if (!path)
        return fail(error, size, "missing point output path");
    if (valid_set(s, error, size))
        return -1;
    f = fopen(path, "wb");
    if (!f)
        return fail(error, size, "cannot open point output: %s", path);
    for (i = 0; i < s->count; ++i) {
        const float values[3] = {s->data[i].x, s->data[i].y, s->data[i].z};
        unsigned char bytes[12];
        int j, k;
        for (j = 0; j < 3; ++j) {
            uint32_t bits;
            memcpy(&bits, &values[j], 4);
            for (k = 0; k < 4; ++k)
                bytes[j * 4 + k] = (unsigned char)(bits >> (8 * k));
        }
        if (fwrite(bytes, 1, 12, f) != 12) {
            result = fail(error, size, "point output write failed: %s", path);
            break;
        }
    }
    if (fclose(f) != 0)
        result = fail(error, size, "point output close failed: %s", path);
    return result;
}

/* Compile with contraction disabled (/fp:strict or -ffp-contract=off). Each
   float operation matches the GPU's explicitly rounded distance predicate. */
static int within(Point a, Point b, float r2) {
    float dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
    float x = dx * dx, y = dy * dy, z = dz * dz;
    float xy = x + y;
    return xy + z <= r2;
}
static int radius_valid(float r, char *error, size_t size) {
    return (!isfinite(r) || r <= 0 || !isfinite(r * r) || r * r < FLT_MIN)
               ? fail(error, size, "radius must be positive with a finite normal float square")
               : 0;
}
int cpu_brute(const PointSet *p, const PointSet *q, float radius, uint32_t k, SearchResult *out,
              char *error, size_t size) {
    size_t i, j;
    float r2 = radius * radius;
    if (valid_set(p, error, size) || valid_set(q, error, size) ||
        radius_valid(radius, error, size) || search_result_alloc(out, q->count, k, error, size))
        return -1;
    for (i = 0; i < q->count; ++i) {
        uint32_t count = 0;
        for (j = 0; j < p->count; ++j)
            if (within(q->data[i], p->data[j], r2)) {
                if (count < k)
                    out->neighbors[i * k + count] = p->data[j].id;
                ++count;
            }
        out->counts[i] = count;
    }
    return 0;
}
struct CpuGrid {
    Point *points;
    size_t *start;
    size_t dims[3], cells;
    Bounds bounds;
    double cell;
    float radius;
};
void cpu_grid_free(CpuGrid *g) {
    if (g) {
        free(g->points);
        free(g->start);
        free(g);
    }
}
static size_t grid_axis(double x, int axis, const CpuGrid *g) {
    double c = floor((x - g->bounds.lo[axis]) / g->cell);
    if (c <= 0)
        return 0;
    if (c >= (double)g->dims[axis])
        return g->dims[axis] - 1;
    return (size_t)c;
}
static size_t grid_index(size_t x, size_t y, size_t z, const CpuGrid *g) {
    return (z * g->dims[1] + y) * g->dims[0] + x;
}
static size_t point_cell(Point p, const CpuGrid *g) {
    return grid_index(grid_axis(p.x, 0, g), grid_axis(p.y, 1, g), grid_axis(p.z, 2, g), g);
}
int cpu_grid_build(const PointSet *p, float radius, CpuGrid **out, char *error, size_t size) {
    CpuGrid *g;
    size_t i, *cursor;
    double longest = 0, magnitude = 1;
    int j;
    if (!out || *out)
        return fail(error, size, "nonempty grid output");
    if (valid_set(p, error, size) || radius_valid(radius, error, size))
        return -1;
    g = (CpuGrid *)calloc(1, sizeof *g);
    if (!g)
        return fail(error, size, "grid allocation failed");
    g->radius = radius;
    if (p->count && pointset_bounds(p, &g->bounds, error, size)) {
        cpu_grid_free(g);
        return -1;
    }
    for (j = 0; j < 3; ++j) {
        longest = fmax(longest, (double)g->bounds.hi[j] - g->bounds.lo[j]);
        magnitude = fmax(magnitude, fmax(fabs(g->bounds.lo[j]), fabs(g->bounds.hi[j])));
    }
    g->cell = fmax((double)radius + 8 * FLT_EPSILON * magnitude, longest / 99.0);
    g->cells = 1;
    for (j = 0; j < 3; ++j) {
        g->dims[j] = (size_t)floor(((double)g->bounds.hi[j] - g->bounds.lo[j]) / g->cell) + 1;
        g->cells *= g->dims[j];
    }
    g->start = (size_t *)calloc(g->cells + 1, sizeof(size_t));
    cursor = (size_t *)malloc(g->cells * sizeof(size_t));
    g->points = p->count ? (Point *)malloc(p->count * sizeof(Point)) : NULL;
    if (!g->start || !cursor || (p->count && !g->points)) {
        free(cursor);
        cpu_grid_free(g);
        return fail(error, size, "grid storage allocation failed");
    }
    for (i = 0; i < p->count; ++i)
        ++g->start[point_cell(p->data[i], g) + 1];
    for (i = 1; i <= g->cells; ++i)
        g->start[i] += g->start[i - 1];
    memcpy(cursor, g->start, g->cells * sizeof(size_t));
    for (i = 0; i < p->count; ++i)
        g->points[cursor[point_cell(p->data[i], g)]++] = p->data[i];
    free(cursor);
    *out = g;
    return 0;
}
int cpu_grid_query(const CpuGrid *g, const PointSet *q, uint32_t k, SearchResult *out, char *error,
                   size_t size) {
    size_t i;
    float r2;
    if (!g)
        return fail(error, size, "missing grid");
    if (valid_set(q, error, size) || search_result_alloc(out, q->count, k, error, size))
        return -1;
    r2 = g->radius * g->radius;
    for (i = 0; i < q->count; ++i) {
        const float v[3] = {q->data[i].x, q->data[i].y, q->data[i].z};
        size_t low[3], high[3], x, y, z, j;
        int axis, outside = 0;
        uint32_t count = 0;
        for (axis = 0; axis < 3; ++axis) {
            double magnitude =
                fmax(fabs(v[axis]), fmax(fabs(g->bounds.lo[axis]), fabs(g->bounds.hi[axis])));
            double margin = (double)g->radius + 8 * FLT_EPSILON * fmax(1, magnitude);
            double lo = (double)v[axis] - margin, hi = (double)v[axis] + margin;
            if (hi < g->bounds.lo[axis] || lo > g->bounds.hi[axis])
                outside = 1;
            low[axis] = grid_axis(lo, axis, g);
            high[axis] = grid_axis(hi, axis, g);
        }
        if (!outside)
            for (z = low[2]; z <= high[2]; ++z)
                for (y = low[1]; y <= high[1]; ++y)
                    for (x = low[0]; x <= high[0]; ++x) {
                        size_t cell = grid_index(x, y, z, g);
                        for (j = g->start[cell]; j < g->start[cell + 1]; ++j)
                            if (within(q->data[i], g->points[j], r2)) {
                                if (count < k)
                                    out->neighbors[i * k + count] = g->points[j].id;
                                ++count;
                            }
                    }
        out->counts[i] = count;
    }
    return 0;
}
int cpu_grid_search(const PointSet *p, const PointSet *q, float radius, uint32_t k,
                    SearchResult *out, char *error, size_t size) {
    CpuGrid *g = NULL;
    int result;
    if (cpu_grid_build(p, radius, &g, error, size))
        return -1;
    result = cpu_grid_query(g, q, k, out, error, size);
    cpu_grid_free(g);
    return result;
}
