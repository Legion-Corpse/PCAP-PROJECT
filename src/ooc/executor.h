// Pipelined execution of a Plan on the device.
//
// Three streams:
//   copy    - DMA only: window queries + launch parameters, tile blobs (H2D)
//   compute - per loaded tile: optixAccelRelocate + a 1-thread kernel that
//             publishes the new handle, then the OptiX launch
//   d2h     - window results
// Nothing on the copy stream needs SMs, so a transfer never queues behind a
// running traversal. A tile slot is refilled only after the last launch that
// read it (slotFree event) and traversed only after its transfer finished
// (tileReady event). Window buffers are double buffered, so the host packs
// window w+1 and unpacks window w-1 while the GPU works on window w.
#pragma once

#include <vector>

#include "common/search_types.h"
#include "ooc/planner.h"
#include "ooc/tile_store.h"

namespace ooc {

class OptixEngine;

struct ExecConfig {
  int slots = 2;
  bool overlap = true;   // false: one stream for everything (no copy/compute overlap)
  bool profile = true;   // per-operation CUDA events for busy-time accounting
};

void executePlan(OptixEngine& engine, const TileSet& tiles, const Plan& plan, const std::vector<float3>& queries,
                 const SearchParams& params, const ExecConfig& cfg, SearchResult& out, Metrics& metrics);

// Device bytes needed by the executor besides the tile slots.
size_t windowBufferBytes(size_t windowQueries, size_t maxPairs, size_t maxSteps, uint32_t K);

}  // namespace ooc
