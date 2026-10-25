// Host-resident tiles: one compacted, relocatable GAS plus its points per tile.
//
// Correctness across tile boundaries is guaranteed by an ownership rule, so no
// result is ever produced twice and no de-duplication pass is needed:
//   HaloMode::Points  - a tile stores its owned points plus every point within
//                       R of its cell (halo); each query is routed to the single
//                       tile whose cell contains it.
//   HaloMode::Queries - tiles store only owned points (disjoint); a query is
//                       routed to every tile whose cell is within R of it and the
//                       partial counts are summed.
// Points duplicates data (more bytes to transfer), Queries duplicates rays
// (more traversals); the cost model chooses between them.
#pragma once

#include <optix_types.h>

#include <vector>

#include "common/search_types.h"
#include "cuda/buffers.h"
#include "ooc/kdtree.h"

namespace ooc {

class OptixEngine;

enum class HaloMode { Points, Queries };

struct TileHost {
  int node = -1;
  uint32_t ownedBegin = 0, ownedCount = 0;
  uint32_t numPoints = 0;     // owned + halo
  uint32_t paddedPoints = 0;  // numPoints rounded up to whole primitives (NaN padding)
  size_t gasBytes = 0;
  size_t pointsOffset = 0;  // offset of the float4 points inside the blob
  size_t blobBytes = 0;     // bytes transferred when the tile is loaded
  OptixRelocationInfo reloc{};
  const char* blob = nullptr;  // [compacted GAS | pad | points], inside TileSet::arena
};

// Bump allocator over large pinned chunks: one cudaHostAlloc per 256 MB
// instead of one per tile (page-locking is slow on Windows).
class PinnedArena {
 public:
  explicit PinnedArena(size_t chunkBytes = size_t(256) << 20) : chunkBytes_(chunkBytes) {}
  char* alloc(size_t bytes);
  size_t reservedBytes() const;

 private:
  size_t chunkBytes_;
  size_t used_ = 0;
  std::vector<PinnedBuffer> chunks_;
};

struct TileSet {
  HaloMode halo = HaloMode::Points;
  int depth = 0;
  float R = 0.f;  // conservative radius used for AABBs and halos
  uint32_t group = 1;  // points per BVH primitive
  std::vector<int> frontier;
  std::vector<int> tileOfNode;
  std::vector<TileHost> tiles;
  size_t maxBlobBytes = 0, totalBlobBytes = 0, totalGasBytes = 0;
  uint64_t haloPoints = 0;
  uint32_t maxTilePoints = 0;
  PinnedArena arena;
};

// Builds every tile's GAS on the device one at a time (only one tile is ever
// resident), compacts it and moves it to pinned host memory.
TileSet buildTiles(OptixEngine& engine, const KdTree& tree, const std::vector<float4>& sortedPoints, int depth,
                   HaloMode mode, float R, uint32_t group, Metrics& metrics);

}  // namespace ooc
