// Launch parameters shared by the host and the OptiX device programs.
#pragma once

#include <optix_types.h>
#include <vector_types.h>

#include <cstdint>

namespace ooc {

struct LaunchParams {
  OptixTraversableHandle handle;  // GAS of the tile being traversed (used when handleSlot is null)
  // Device-side handle written by the compute stream after relocation, so the
  // parameters of a whole window can be uploaded before its tiles are loaded.
  const OptixTraversableHandle* handleSlot;
  const float4* points;           // tile points: xyz + global point id (uint bits in .w), NaN-padded
  uint32_t group;                 // points per primitive (primitive i = points[i*group, (i+1)*group))
  const float4* queries;          // window queries (xyz)
  const uint32_t* queryList;      // window-query slots served by this launch (nullptr = 0..numQueries-1)
  uint32_t* counts;               // running |N(q)| per window slot; accumulates across tiles
  uint32_t* neighbors;            // slot * maxNeighbors ids, or nullptr in count-only mode
  uint32_t numQueries;
  uint32_t maxNeighbors;
  float radius2;
};

}  // namespace ooc
