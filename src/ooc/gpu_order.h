// Tile-major query ordering on the device, in chunks that fit a memory budget.
//
// Per chunk: kd-tree descent (owner tile) and Morton key in one kernel, CUB
// radix sort by (tile, Morton). Only the permutation comes back (4 bytes per
// query); the host concatenates the per-tile runs of all chunks. Inside a
// tile, queries are in Morton order per chunk, which keeps rays coherent.
#pragma once

#include <cstdint>
#include <vector>

#include "common/vec_math.h"
#include "ooc/kdtree.h"

namespace ooc {

struct TiledOrderResult {
  std::vector<uint32_t> order;      // original query index per processing position
  std::vector<uint32_t> tileStart;  // queries of tile t: order[tileStart[t], tileStart[t+1])
  double deviceMs = 0, hostMs = 0;
};

TiledOrderResult gpuTiledOrder(const std::vector<float3>& queries, const KdTree& tree,
                               const std::vector<int>& tileOfNode, size_t numTiles, size_t budgetBytes);

}  // namespace ooc
