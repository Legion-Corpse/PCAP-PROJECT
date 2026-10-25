// Median-split kd-tree over the point set, used to cut space into tiles.
//
// The tree is built once down to a small leaf size. A tiling is a "frontier"
// of the tree: every node at depth d (or a shallower leaf). Each tile owns the
// points of its subtree, which are contiguous in the permuted point array, and
// its cell is the axis-aligned region of space carved out by the splits above
// it (outer faces are infinite, so every query has exactly one owner tile).
// Cutting one tree at different depths lets the cost model compare tile sizes
// without re-partitioning.
#pragma once

#include <cstdint>
#include <vector>

#include "common/vec_math.h"

namespace ooc {

struct KdNode {
  Aabb cell;             // region of space owned by the subtree (may be infinite)
  float split = 0.f;     // left child: coord < split, right child: coord >= split
  int8_t axis = -1;      // -1 for leaves
  uint8_t depth = 0;
  int32_t left = -1, right = -1;
  uint32_t begin = 0, end = 0;  // owned range in the permuted point array

  bool leaf() const { return axis < 0; }
  uint32_t count() const { return end - begin; }
};

class KdTree {
 public:
  // Permutes `pts` (xyz + id bits) in place so that every subtree is contiguous.
  static KdTree build(std::vector<float4>& pts, uint32_t leafCapacity);

  const std::vector<KdNode>& nodes() const { return nodes_; }
  int maxDepth() const { return maxDepth_; }

  // Tiles at `depth`: nodes at that depth plus shallower leaves, in DFS order.
  std::vector<int> frontier(int depth) const;
  // Smallest depth whose largest tile holds at most `capacity` owned points.
  int depthForCapacity(uint32_t capacity) const;

  // Maps node -> tile index for a frontier (-1 for nodes that are not tiles).
  std::vector<int> tileMap(const std::vector<int>& frontier) const;

  // Tile whose cell contains p.
  int owner(float3 p, const std::vector<int>& tileOfNode) const;

  // Calls fn(tile) for every tile whose cell is within distance R of p.
  template <class Fn>
  void within(float3 p, double R, const std::vector<int>& tileOfNode, Fn&& fn) const {
    const double R2 = R * R;
    int stack[128];
    int top = 0;
    stack[top++] = 0;
    while (top) {
      const KdNode& n = nodes_[size_t(stack[--top])];
      const int self = int(&n - nodes_.data());
      if (tileOfNode[size_t(self)] >= 0) {
        if (n.cell.dist2To(p) <= R2) fn(tileOfNode[size_t(self)]);
        continue;
      }
      const float c = comp(p, n.axis);
      if (double(c) - double(n.split) <= R) stack[top++] = n.left;   // gap to the half-space coord < split
      if (double(n.split) - double(c) <= R) stack[top++] = n.right;  // gap to the half-space coord >= split
    }
  }

  // cover[node] = number of sample positions within distance R of the node's
  // cell (owned ones included). With sample = points this estimates owned +
  // halo points of each candidate tile; with sample = queries it estimates the
  // per-tile load of the multi-tile (query-halo) routing.
  std::vector<uint32_t> nodeCoverage(const std::vector<float3>& sample, double R) const;

  // owners[node] = number of sample positions inside the node's cell.
  std::vector<uint32_t> ownerCounts(const std::vector<float3>& sample) const;

 private:
  std::vector<KdNode> nodes_;
  int maxDepth_ = 0;
};

}  // namespace ooc
