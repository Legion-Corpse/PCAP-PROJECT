#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#define OPTIXAPI static
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include "rt_backend.h"

extern const unsigned char g_devicePrograms[];
extern const size_t g_deviceProgramsSize;

struct RtContext {
    OptixDeviceContext context;
    OptixModule module;
    OptixProgramGroup groups[3];
    OptixPipeline pipeline;
    OptixShaderBindingTable sbt;
    void *sbt_records;
};

_Static_assert(sizeof(Point) == 16, "Point device ABI must be 16 bytes");
_Static_assert(sizeof(RtParams) == 64, "RtParams device ABI must be 64 bytes");

static int diagnostic(char *error, size_t size, const char *operation, const char *message) {
    if (error && size)
        snprintf(error, size, "%s: %s", operation, message);
    return -1;
}
#define CUDA_TRY(call)                                                                             \
    do {                                                                                           \
        cudaError_t e_ = (call);                                                                   \
        if (e_ != cudaSuccess) {                                                                   \
            diagnostic(error, error_size, #call, cudaGetErrorString(e_));                          \
            goto fail;                                                                             \
        }                                                                                          \
    } while (0)
#define OPTIX_TRY(call)                                                                            \
    do {                                                                                           \
        OptixResult e_ = (call);                                                                   \
        if (e_ != OPTIX_SUCCESS) {                                                                 \
            diagnostic(error, error_size, #call, optixGetErrorName(e_));                           \
            goto fail;                                                                             \
        }                                                                                          \
    } while (0)

static void optix_log(unsigned int level, const char *tag, const char *message, void *data) {
    (void)data;
    if (level <= 2)
        fprintf(stderr, "[optix][%u][%s] %s\n", level, tag, message);
}

void rt_context_destroy(RtContext *ctx) {
    if (!ctx)
        return;
    if (ctx->pipeline)
        optixPipelineDestroy(ctx->pipeline);
    for (int i = 0; i < 3; ++i)
        if (ctx->groups[i])
            optixProgramGroupDestroy(ctx->groups[i]);
    if (ctx->module)
        optixModuleDestroy(ctx->module);
    if (ctx->sbt_records)
        cudaFree(ctx->sbt_records);
    if (ctx->context)
        optixDeviceContextDestroy(ctx->context);
    free(ctx);
}

int rt_context_create(int validation, RtContext **out, char *error, size_t error_size) {
    RtContext *ctx = NULL;
    OptixDeviceContextOptions options = {0};
    OptixModuleCompileOptions module_options = {0};
    OptixPipelineCompileOptions pipeline_options = {0};
    OptixPipelineLinkOptions link_options = {0};
    OptixProgramGroupDesc descriptions[3] = {0};
    OptixProgramGroupOptions group_options = {0};
    _Alignas(16) unsigned char headers[3][OPTIX_SBT_RECORD_HEADER_SIZE];
    char log[4096] = {0};
    size_t log_size = sizeof(log);
    OptixResult result;
    if (!out || *out)
        return diagnostic(error, error_size, "rt_context_create", "output must be empty");
    ctx = calloc(1, sizeof(*ctx));
    if (!ctx)
        return diagnostic(error, error_size, "rt_context_create", "out of host memory");
    CUDA_TRY(cudaFree(NULL));
    OPTIX_TRY(optixInit());
    options.logCallbackFunction = optix_log;
    options.logCallbackLevel = 3;
    options.validationMode = validation ? OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL
                                        : OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
    OPTIX_TRY(optixDeviceContextCreate(NULL, &options, &ctx->context));
    module_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    pipeline_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_options.numPayloadValues = 2;
    pipeline_options.pipelineLaunchParamsVariableName = "params";
    pipeline_options.pipelineLaunchParamsSizeInBytes = sizeof(RtParams);
    pipeline_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;
    result = optixModuleCreate(ctx->context, &module_options, &pipeline_options,
                               (const char *)g_devicePrograms, g_deviceProgramsSize, log, &log_size,
                               &ctx->module);
    if (result != OPTIX_SUCCESS) {
        diagnostic(error, error_size, "optixModuleCreate", log);
        goto fail;
    }
    descriptions[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    descriptions[0].raygen.module = ctx->module;
    descriptions[0].raygen.entryFunctionName = "__raygen__frnn";
    descriptions[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    descriptions[1].miss.module = ctx->module;
    descriptions[1].miss.entryFunctionName = "__miss__frnn";
    descriptions[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    descriptions[2].hitgroup.moduleIS = ctx->module;
    descriptions[2].hitgroup.entryFunctionNameIS = "__intersection__frnn";
    log_size = sizeof(log);
    result = optixProgramGroupCreate(ctx->context, descriptions, 3, &group_options, log, &log_size,
                                     ctx->groups);
    if (result != OPTIX_SUCCESS) {
        diagnostic(error, error_size, "optixProgramGroupCreate", log);
        goto fail;
    }
    link_options.maxTraceDepth = 1;
    log_size = sizeof(log);
    result = optixPipelineCreate(ctx->context, &pipeline_options, &link_options, ctx->groups, 3,
                                 log, &log_size, &ctx->pipeline);
    if (result != OPTIX_SUCCESS) {
        diagnostic(error, error_size, "optixPipelineCreate", log);
        goto fail;
    }
    /* This program has one trace, no callable programs, and one GAS level. */
    OPTIX_TRY(optixPipelineSetStackSizeFromCallDepths(ctx->pipeline, 1, 0, 0, 0, 1));
    for (int i = 0; i < 3; ++i)
        OPTIX_TRY(optixSbtRecordPackHeader(ctx->groups[i], headers[i]));
    CUDA_TRY(cudaMalloc(&ctx->sbt_records, sizeof(headers)));
    CUDA_TRY(cudaMemcpy(ctx->sbt_records, headers, sizeof(headers), cudaMemcpyHostToDevice));
    ctx->sbt.raygenRecord = (CUdeviceptr)ctx->sbt_records;
    ctx->sbt.missRecordBase = (CUdeviceptr)ctx->sbt_records + OPTIX_SBT_RECORD_HEADER_SIZE;
    ctx->sbt.missRecordStrideInBytes = (unsigned)OPTIX_SBT_RECORD_HEADER_SIZE;
    ctx->sbt.missRecordCount = 1;
    ctx->sbt.hitgroupRecordBase = (CUdeviceptr)ctx->sbt_records + 2 * OPTIX_SBT_RECORD_HEADER_SIZE;
    ctx->sbt.hitgroupRecordStrideInBytes = (unsigned)OPTIX_SBT_RECORD_HEADER_SIZE;
    ctx->sbt.hitgroupRecordCount = 1;
    *out = ctx;
    return 0;
fail:
    rt_context_destroy(ctx);
    return -1;
}

void rt_tile_free(RtTile *tile) {
    if (!tile)
        return;
    if (tile->host)
        cudaFreeHost(tile->host);
    memset(tile, 0, sizeof(*tile));
}

int rt_tile_build(RtContext *ctx, const Point *points, uint32_t count, float radius, uint32_t group,
                  RtTile *out, char *error, size_t error_size) {
    RtTile tile = {0};
    OptixAabb *aabbs = NULL;
    void *d_points = NULL, *d_aabbs = NULL, *d_temp = NULL, *d_output = NULL, *d_size = NULL,
         *d_compact = NULL;
    OptixAccelBuildOptions options = {0};
    OptixBuildInput input = {0};
    OptixAccelBufferSizes sizes = {0};
    OptixAccelEmitDesc emit = {0};
    OptixTraversableHandle handle = 0, compact_handle = 0;
    CUdeviceptr aabb_ptr = 0;
    uint64_t compact_bytes = 0;
    unsigned flags =
        OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT | OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL;
    size_t padded, point_bytes;
    uint32_t primitives;
    float max_abs = 0, half_width;
    int compatible = 0;
    if (!ctx || !out || out->host || !points || !count || !group || !isfinite(radius) ||
        radius <= 0 || !isfinite(radius * radius) || radius * radius < FLT_MIN)
        return diagnostic(error, error_size, "rt_tile_build",
                          "invalid points, group, radius, or output");
    padded = ((size_t)count + group - 1) / group * group;
    if (padded > SIZE_MAX / sizeof(Point))
        return diagnostic(error, error_size, "rt_tile_build", "point size overflow");
    point_bytes = padded * sizeof(Point);
    primitives = (uint32_t)(padded / group);
    aabbs = malloc((size_t)primitives * sizeof(*aabbs));
    if (!aabbs)
        return diagnostic(error, error_size, "rt_tile_build", "out of host memory for bounds");
    for (uint32_t i = 0; i < count; ++i) {
        if (!isfinite(points[i].x) || !isfinite(points[i].y) || !isfinite(points[i].z)) {
            diagnostic(error, error_size, "rt_tile_build", "point coordinates must be finite");
            goto fail;
        }
        max_abs = fmaxf(max_abs,
                        fmaxf(fabsf(points[i].x), fmaxf(fabsf(points[i].y), fabsf(points[i].z))));
    }
    half_width = radius * (1.f + 1e-5f) + 4.f * FLT_EPSILON * max_abs;
    if (!isfinite(half_width)) {
        diagnostic(error, error_size, "rt_tile_build", "expanded radius overflow");
        goto fail;
    }
    for (uint32_t i = 0; i < primitives; ++i) {
        OptixAabb box = {INFINITY, INFINITY, INFINITY, -INFINITY, -INFINITY, -INFINITY};
        size_t end = ((size_t)i + 1) * group;
        if (end > count)
            end = count;
        for (size_t j = (size_t)i * group; j < end; ++j) {
            box.minX = fminf(box.minX, points[j].x);
            box.maxX = fmaxf(box.maxX, points[j].x);
            box.minY = fminf(box.minY, points[j].y);
            box.maxY = fmaxf(box.maxY, points[j].y);
            box.minZ = fminf(box.minZ, points[j].z);
            box.maxZ = fmaxf(box.maxZ, points[j].z);
        }
        box.minX -= half_width;
        box.minY -= half_width;
        box.minZ -= half_width;
        box.maxX += half_width;
        box.maxY += half_width;
        box.maxZ += half_width;
        aabbs[i] = box;
    }
    CUDA_TRY(cudaMalloc(&d_points, point_bytes));
    CUDA_TRY(cudaMemcpy(d_points, points, (size_t)count * sizeof(Point), cudaMemcpyHostToDevice));
    if (padded > count)
        CUDA_TRY(cudaMemset((char *)d_points + (size_t)count * sizeof(Point), 0xff,
                            (padded - count) * sizeof(Point)));
    CUDA_TRY(cudaMalloc(&d_aabbs, (size_t)primitives * sizeof(*aabbs)));
    CUDA_TRY(
        cudaMemcpy(d_aabbs, aabbs, (size_t)primitives * sizeof(*aabbs), cudaMemcpyHostToDevice));
    free(aabbs);
    aabbs = NULL;
    options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    options.operation = OPTIX_BUILD_OPERATION_BUILD;
    aabb_ptr = (CUdeviceptr)d_aabbs;
    input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    input.customPrimitiveArray.aabbBuffers = &aabb_ptr;
    input.customPrimitiveArray.numPrimitives = primitives;
    input.customPrimitiveArray.flags = &flags;
    input.customPrimitiveArray.numSbtRecords = 1;
    OPTIX_TRY(optixAccelComputeMemoryUsage(ctx->context, &options, &input, 1, &sizes));
    CUDA_TRY(cudaMalloc(&d_temp, sizes.tempSizeInBytes));
    CUDA_TRY(cudaMalloc(&d_output, sizes.outputSizeInBytes));
    CUDA_TRY(cudaMalloc(&d_size, sizeof(compact_bytes)));
    emit.result = (CUdeviceptr)d_size;
    emit.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    OPTIX_TRY(optixAccelBuild(ctx->context, 0, &options, &input, 1, (CUdeviceptr)d_temp,
                              sizes.tempSizeInBytes, (CUdeviceptr)d_output, sizes.outputSizeInBytes,
                              &handle, &emit, 1));
    CUDA_TRY(cudaMemcpy(&compact_bytes, d_size, sizeof(compact_bytes), cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaFree(d_aabbs));
    d_aabbs = NULL;
    CUDA_TRY(cudaFree(d_temp));
    d_temp = NULL;
    if (!compact_bytes || compact_bytes > SIZE_MAX - point_bytes - 127) {
        diagnostic(error, error_size, "rt_tile_build", "invalid compacted GAS size");
        goto fail;
    }
    CUDA_TRY(cudaMalloc(&d_compact, (size_t)compact_bytes));
    OPTIX_TRY(optixAccelCompact(ctx->context, 0, handle, (CUdeviceptr)d_compact,
                                (size_t)compact_bytes, &compact_handle));
    CUDA_TRY(cudaDeviceSynchronize());
    OPTIX_TRY(optixAccelGetRelocationInfo(ctx->context, compact_handle, &tile.relocation));
    OPTIX_TRY(optixCheckRelocationCompatibility(ctx->context, &tile.relocation, &compatible));
    if (!compatible) {
        diagnostic(error, error_size, "rt_tile_build", "GAS relocation unsupported");
        goto fail;
    }
    tile.gas_offset = (point_bytes + 127) & ~(size_t)127;
    tile.gas_bytes = (size_t)compact_bytes;
    tile.bytes = tile.gas_offset + tile.gas_bytes;
    tile.n_points = count;
    tile.group = group;
    CUDA_TRY(cudaHostAlloc(&tile.host, tile.bytes, cudaHostAllocDefault));
    memcpy(tile.host, points, (size_t)count * sizeof(Point));
    memset((char *)tile.host + (size_t)count * sizeof(Point), 0xff,
           tile.gas_offset - (size_t)count * sizeof(Point));
    CUDA_TRY(cudaMemcpy((char *)tile.host + tile.gas_offset, d_compact, tile.gas_bytes,
                        cudaMemcpyDeviceToHost));
    cudaFree(d_points);
    cudaFree(d_output);
    cudaFree(d_size);
    cudaFree(d_compact);
    *out = tile;
    return 0;
fail:
    free(aabbs);
    if (d_points)
        cudaFree(d_points);
    if (d_aabbs)
        cudaFree(d_aabbs);
    if (d_temp)
        cudaFree(d_temp);
    if (d_output)
        cudaFree(d_output);
    if (d_size)
        cudaFree(d_size);
    if (d_compact)
        cudaFree(d_compact);
    rt_tile_free(&tile);
    return -1;
}

void rt_slot_free(RtSlot *slot) {
    if (!slot)
        return;
    /* cudaFree waits for uses of the allocation, including pending launches. */
    if (slot->device)
        cudaFree(slot->device);
    if (slot->device_params)
        cudaFree(slot->device_params);
    if (slot->params_uploaded)
        cudaEventDestroy(slot->params_uploaded);
    if (slot->host_params)
        cudaFreeHost(slot->host_params);
    memset(slot, 0, sizeof(*slot));
}
size_t rt_slot_overhead_bytes(void) { return sizeof(RtParams); }
int rt_slot_alloc(RtSlot *slot, size_t bytes, char *error, size_t error_size) {
    if (!slot || slot->device || !bytes)
        return diagnostic(error, error_size, "rt_slot_alloc", "invalid slot or size");
    CUDA_TRY(cudaMalloc(&slot->device, bytes));
    slot->capacity = bytes;
    CUDA_TRY(cudaMalloc(&slot->device_params, sizeof(RtParams)));
    CUDA_TRY(cudaHostAlloc((void **)&slot->host_params, sizeof(RtParams), cudaHostAllocDefault));
    CUDA_TRY(cudaEventCreateWithFlags(&slot->params_uploaded, cudaEventDisableTiming));
    return 0;
fail:
    rt_slot_free(slot);
    return -1;
}
int rt_slot_upload(RtContext *ctx, const RtTile *tile, RtSlot *slot, cudaStream_t stream,
                   char *error, size_t error_size) {
    if (!ctx || !tile || !tile->host || !slot || !slot->device || slot->capacity < tile->bytes)
        return diagnostic(error, error_size, "rt_slot_upload",
                          "invalid tile or insufficient slot capacity");
    CUDA_TRY(
        cudaMemcpyAsync(slot->device, tile->host, tile->bytes, cudaMemcpyHostToDevice, stream));
    OPTIX_TRY(optixAccelRelocate(ctx->context, stream, &tile->relocation, NULL, 0,
                                 (CUdeviceptr)slot->device + tile->gas_offset, tile->gas_bytes,
                                 &slot->handle));
    return 0;
fail:
    return -1;
}
int rt_launch(RtContext *ctx, RtSlot *slot, uint32_t group, const Point *queries,
              const uint32_t *query_list, uint32_t count, uint32_t *counts, uint32_t *neighbors,
              uint32_t max_neighbors, float radius2, cudaStream_t stream, char *error,
              size_t error_size) {
    RtParams *p;
    if (!count)
        return 0;
    if (!ctx || !slot || !slot->handle || !slot->host_params || !queries || !counts || !group ||
        (max_neighbors && !neighbors) || !isfinite(radius2) || radius2 < FLT_MIN)
        return diagnostic(error, error_size, "rt_launch", "invalid launch parameters");
    /* Pin staging cannot be overwritten until the previous upload consumed it. */
    if (slot->params_pending)
        CUDA_TRY(cudaEventSynchronize(slot->params_uploaded));
    p = slot->host_params;
    memset(p, 0, sizeof(*p));
    p->handle = slot->handle;
    p->points = (const Point *)slot->device;
    p->queries = queries;
    p->query_list = query_list;
    p->counts = counts;
    p->neighbors = neighbors;
    p->group = group;
    p->num_queries = count;
    p->max_neighbors = max_neighbors;
    p->radius2 = radius2;
    CUDA_TRY(cudaMemcpyAsync(slot->device_params, p, sizeof(*p), cudaMemcpyHostToDevice, stream));
    CUDA_TRY(cudaEventRecord(slot->params_uploaded, stream));
    slot->params_pending = 1;
    OPTIX_TRY(optixLaunch(ctx->pipeline, stream, (CUdeviceptr)slot->device_params, sizeof(*p),
                          &ctx->sbt, count, 1, 1));
    return 0;
fail:
    return -1;
}
