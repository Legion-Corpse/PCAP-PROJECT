#include "ooc/planner.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

#include "common/dataset.h"
#include "common/host_utils.h"
#include "common/morton.h"
#include "common/timer.h"
#include "cuda/kernels.h"
#include "ooc/gpu_order.h"

namespace ooc {

std::vector<uint32_t> queryOrder(const std::vector<float3>& queries, QueryOrder order, uint64_t seed) {
  const int64_t m = int64_t(queries.size());
  if (order == QueryOrder::Morton) {
    const Aabb box = computeBounds(queries);
    std::vector<uint64_t> keys(queries.size());
    parallelFor(m, 1 << 16, [&](int64_t b, int64_t e) {
      for (int64_t i = b; i < e; ++i) keys[size_t(i)] = mortonCode(queries[size_t(i)], box) >> (63 - kOrderKeyBits);
    });
    return sortedOrder(keys, kOrderKeyBits);
  }
  std::vector<uint32_t> idx(queries.size());
  std::iota(idx.begin(), idx.end(), 0u);
  if (order == QueryOrder::Random) {
    std::mt19937_64 rng(seed);
    for (size_t i = idx.size(); i > 1; --i) std::swap(idx[i - 1], idx[size_t(rng() % i)]);
  }
  return idx;
}

namespace {

// Tile-major order: stable counting sort by owner tile, then Morton order
// inside every tile. Tile t's queries end up at [tileStart[t], tileStart[t+1]).
std::vector<uint32_t> tiledOrder(const std::vector<float3>& queries, const KdTree& tree, const TileSet& tiles,
                                 std::vector<uint32_t>& tileStart, double* phaseMs) {
  Timer timer;
  const int64_t M = int64_t(queries.size());
  const int64_t T = int64_t(tiles.tiles.size());
  std::vector<uint32_t> owner(queries.size());
  parallelFor(M, 1 << 14, [&](int64_t b, int64_t e) {
    for (int64_t i = b; i < e; ++i) owner[size_t(i)] = uint32_t(tree.owner(queries[size_t(i)], tiles.tileOfNode));
  });
  phaseMs[0] = timer.ms();
  timer.reset();
  const int64_t parts = std::min<int64_t>(64, std::max<int64_t>(1, M / (1 << 16)));
  const int64_t span = (M + parts - 1) / parts;
  std::vector<uint32_t> hist(static_cast<size_t>(parts * T), 0u);
  parallelFor(parts, 1, [&](int64_t b, int64_t e) {
    for (int64_t p = b; p < e; ++p)
      for (int64_t i = p * span, end = std::min(M, (p + 1) * span); i < end; ++i) ++hist[size_t(p * T + owner[size_t(i)])];
  });
  tileStart.assign(static_cast<size_t>(T) + 1, 0u);
  uint32_t run = 0;
  for (int64_t t = 0; t < T; ++t) {
    tileStart[size_t(t)] = run;
    for (int64_t p = 0; p < parts; ++p) {
      uint32_t& h = hist[size_t(p * T + t)];
      const uint32_t c = h;
      h = run;
      run += c;
    }
  }
  tileStart[size_t(T)] = run;
  std::vector<uint32_t> order(queries.size());
  parallelFor(parts, 1, [&](int64_t b, int64_t e) {
    for (int64_t p = b; p < e; ++p)
      for (int64_t i = p * span, end = std::min(M, (p + 1) * span); i < end; ++i)
        order[hist[size_t(p * T + owner[size_t(i)])]++] = uint32_t(i);
  });
  phaseMs[1] = timer.ms();
  timer.reset();
  const Aabb box = computeBounds(queries);
  parallelFor(T, 1, [&](int64_t b, int64_t e) {
    std::vector<uint64_t> keys, tmpKeys;
    std::vector<uint32_t> tmpVals;
    for (int64_t t = b; t < e; ++t) {
      const uint32_t s0 = tileStart[size_t(t)], n = tileStart[size_t(t) + 1] - s0;
      keys.resize(n);
      tmpKeys.resize(n);
      tmpVals.resize(n);
      for (uint32_t k = 0; k < n; ++k) keys[k] = mortonCode(queries[order[s0 + k]], box) >> (63 - kOrderKeyBits);
      // Stable, so equal keys keep the original query order.
      radixSortPairsSerial(keys.data(), order.data() + s0, tmpKeys.data(), tmpVals.data(), n, kOrderKeyBits);
    }
  });
  phaseMs[2] = timer.ms();
  return order;
}

}  // namespace

void simulateCache(Plan& plan, const TileSet& tiles, int slots, Eviction eviction) {
  Timer timer;
  const size_t T = tiles.tiles.size();
  const int S = std::max(1, slots);
  // Flatten the access sequence; Belady needs every step's next use of the same tile.
  std::vector<PlanStep*> seq;
  seq.reserve(plan.steps);
  for (auto& w : plan.windows)
    for (auto& s : w.steps) seq.push_back(&s);
  const size_t n = seq.size();
  constexpr size_t kNever = std::numeric_limits<size_t>::max();
  std::vector<size_t> nextUse(n, kNever);
  if (eviction == Eviction::Belady) {
    std::vector<size_t> seen(T, kNever);
    for (size_t k = n; k-- > 0;) {
      nextUse[k] = seen[seq[k]->tile];
      seen[seq[k]->tile] = k;
    }
  }
  std::vector<int64_t> slotTile(size_t(S), -1);
  std::vector<size_t> slotLast(size_t(S), 0), slotNext(size_t(S), kNever);
  std::vector<int> tileSlot(T, -1);
  // Slots used by the last L steps are never victims. L >= 1 lets the next
  // transfer overlap the current traversal; L = 2 also keeps a second transfer
  // in flight. Without it, Belady tends to evict the slot used two steps ago
  // (its tile is needed again soonest of all - in the next window), which caps
  // prefetching at one tile and loses ~13% when transfer-bound.
  const size_t L = size_t(std::min(2, S - 1));
  plan.loads = plan.hits = plan.loadBytes = 0;
  for (size_t k = 0; k < n; ++k) {
    PlanStep& st = *seq[k];
    int slot = tileSlot[st.tile];
    if (slot >= 0) {
      st.load = false;
      ++plan.hits;
    } else {
      st.load = true;
      ++plan.loads;
      plan.loadBytes += tiles.tiles[st.tile].blobBytes;
      for (int s = 0; s < S && slot < 0; ++s)
        if (slotTile[size_t(s)] < 0) slot = s;
      if (slot < 0) {
        for (int s = 0; s < S; ++s) {
          if (slotLast[size_t(s)] + L >= k) continue;  // used by one of the last L steps
          if (slot < 0) { slot = s; continue; }
          const bool better = eviction == Eviction::Belady ? slotNext[size_t(s)] > slotNext[size_t(slot)]
                                                           : slotLast[size_t(s)] < slotLast[size_t(slot)];
          if (better) slot = s;
        }
        tileSlot[size_t(slotTile[size_t(slot)])] = -1;
      }
      slotTile[size_t(slot)] = st.tile;
      tileSlot[st.tile] = slot;
    }
    st.slot = uint32_t(slot);
    slotLast[size_t(slot)] = k;
    slotNext[size_t(slot)] = nextUse[k];
  }
  plan.cacheMs = timer.ms();
}

Plan makePlan(const std::vector<float3>& queries, const KdTree& tree, const TileSet& tiles, const PlanConfig& cfg) {
  NvtxRange range("plan");
  Plan plan;
  Timer timer;
  const bool tiled = cfg.order == QueryOrder::Tiled;
  if (tiled && cfg.deviceOrder) {
    TiledOrderResult r = gpuTiledOrder(queries, tree, tiles.tileOfNode, tiles.tiles.size(), cfg.deviceOrderBudget);
    plan.order = std::move(r.order);
    plan.tileStart = std::move(r.tileStart);
    plan.orderDeviceMs = r.deviceMs;
    plan.orderHostMs = r.hostMs;
  } else {
    plan.order = tiled ? tiledOrder(queries, tree, tiles, plan.tileStart, plan.orderPhaseMs)
                       : queryOrder(queries, cfg.order, cfg.seed);
  }
  // Tile-major + single-tile routing needs no coordinates on the host.
  const bool contiguous = tiled && tiles.halo == HaloMode::Points;
  if (!contiguous) {
    Timer gatherTimer;
    plan.queries.resize(queries.size());
    parallelFor(int64_t(queries.size()), 1 << 16, [&](int64_t b, int64_t e) {
      for (int64_t i = b; i < e; ++i) plan.queries[size_t(i)] = queries[plan.order[size_t(i)]];
    });
    plan.orderPhaseMs[3] = gatherTimer.ms();
  }
  plan.orderMs = timer.ms();

  timer.reset();
  const size_t M = queries.size();
  const int64_t T = int64_t(tiles.tiles.size());
  const uint32_t W = std::max<uint32_t>(1, cfg.window);
  std::vector<uint32_t> routes, pairOffset, pairTile, pairSlot;
  std::vector<uint32_t> hist, tileCount(static_cast<size_t>(T)), tileStart(static_cast<size_t>(T));
  std::vector<uint32_t> recent;  // last `slots` distinct tiles visited
  for (size_t first = 0; first < M; first += W) {
    WindowPlan wp;
    wp.first = first;
    wp.count = uint32_t(std::min<size_t>(W, M - first));
    const int64_t count = int64_t(wp.count);
    const float3* wq = contiguous ? nullptr : plan.queries.data() + first;

    // Route: (tile, window slot) pairs in window order.
    const bool single = tiles.halo == HaloMode::Points;
    if (contiguous) {
      // Tile-major order: every tile's queries form one run; no lists needed.
      wp.contiguous = true;
      const size_t last = first + wp.count;
      size_t t = size_t(std::upper_bound(plan.tileStart.begin(), plan.tileStart.end(), uint32_t(first)) -
                        plan.tileStart.begin()) - 1;
      for (; t + 1 < plan.tileStart.size() && plan.tileStart[t] < last; ++t) {
        const size_t b = std::max<size_t>(plan.tileStart[t], first), e = std::min<size_t>(plan.tileStart[t + 1], last);
        if (e <= b) continue;
        PlanStep st;
        st.tile = uint32_t(t);
        st.listOffset = uint32_t(b - first);
        st.listCount = uint32_t(e - b);
        wp.steps.push_back(st);
      }
      plan.pairs += wp.count;
      plan.steps += wp.steps.size();
      plan.maxSteps = std::max(plan.maxSteps, wp.steps.size());
      plan.maxPairs = std::max<size_t>(plan.maxPairs, 1);
      plan.maxWindow = std::max<size_t>(plan.maxWindow, wp.count);
      plan.windows.push_back(std::move(wp));
      continue;
    }
    if (single) {
      pairTile.resize(size_t(count));
      parallelFor(count, 1 << 14, [&](int64_t b, int64_t e) {
        for (int64_t j = b; j < e; ++j) pairTile[size_t(j)] = uint32_t(tree.owner(wq[j], tiles.tileOfNode));
      });
    } else {
      routes.assign(size_t(count), 0);
      parallelFor(count, 1 << 12, [&](int64_t b, int64_t e) {
        for (int64_t j = b; j < e; ++j) {
          uint32_t c = 0;
          tree.within(wq[j], tiles.R, tiles.tileOfNode, [&](int) { ++c; });
          routes[size_t(j)] = c;
        }
      });
      pairOffset.assign(size_t(count) + 1, 0);
      for (int64_t j = 0; j < count; ++j) pairOffset[size_t(j) + 1] = pairOffset[size_t(j)] + routes[size_t(j)];
      pairTile.resize(pairOffset[size_t(count)]);
      pairSlot.resize(pairTile.size());
      parallelFor(count, 1 << 12, [&](int64_t b, int64_t e) {
        for (int64_t j = b; j < e; ++j) {
          uint32_t o = pairOffset[size_t(j)];
          tree.within(wq[j], tiles.R, tiles.tileOfNode, [&](int t) {
            pairSlot[o] = uint32_t(j);
            pairTile[o++] = uint32_t(t);
          });
        }
      });
    }
    const int64_t P = int64_t(pairTile.size());

    // Stable parallel counting sort of the pairs by tile.
    const int64_t parts = std::min<int64_t>(64, std::max<int64_t>(1, P / (1 << 16)));
    const int64_t span = (P + parts - 1) / std::max<int64_t>(parts, 1);
    hist.assign(size_t(parts * T), 0u);
    parallelFor(parts, 1, [&](int64_t b, int64_t e) {
      for (int64_t p = b; p < e; ++p)
        for (int64_t i = p * span, end = std::min(P, (p + 1) * span); i < end; ++i) ++hist[size_t(p * T + pairTile[size_t(i)])];
    });
    for (int64_t t = 0; t < T; ++t) {
      uint32_t c = 0;
      for (int64_t p = 0; p < parts; ++p) c += hist[size_t(p * T + t)];
      tileCount[size_t(t)] = c;
    }
    // Visit order: tiles that are probably still resident (the last `slots`
    // distinct tiles used) first, so they are reused before being evicted;
    // then the others in ascending (kd DFS = spatial) order.
    std::vector<uint32_t> touched, fresh;
    for (int64_t t = 0; t < T; ++t) {
      if (!tileCount[size_t(t)]) continue;
      const bool warm = std::find(recent.begin(), recent.end(), uint32_t(t)) != recent.end();
      (warm ? touched : fresh).push_back(uint32_t(t));
    }
    touched.insert(touched.end(), fresh.begin(), fresh.end());
    for (uint32_t t : touched) {
      recent.erase(std::remove(recent.begin(), recent.end(), t), recent.end());
      recent.push_back(t);
      if (recent.size() > size_t(std::max(1, cfg.slots))) recent.erase(recent.begin());
    }
    uint32_t offset = 0;
    for (uint32_t t : touched) {
      PlanStep st;
      st.tile = t;
      st.listOffset = offset;
      st.listCount = tileCount[t];
      tileStart[t] = offset;
      offset += tileCount[t];
      wp.steps.push_back(st);
    }
    for (uint32_t t : touched) {  // per-part write cursors
      uint32_t run = tileStart[t];
      for (int64_t p = 0; p < parts; ++p) {
        uint32_t& h = hist[size_t(p * T + t)];
        const uint32_t c = h;
        h = run;
        run += c;
      }
    }
    wp.lists.resize(offset);
    parallelFor(parts, 1, [&](int64_t b, int64_t e) {
      for (int64_t p = b; p < e; ++p)
        for (int64_t i = p * span, end = std::min(P, (p + 1) * span); i < end; ++i)
          wp.lists[hist[size_t(p * T + pairTile[size_t(i)])]++] = single ? uint32_t(i) : pairSlot[size_t(i)];
    });
    plan.pairs += offset;
    plan.steps += wp.steps.size();
    plan.maxSteps = std::max(plan.maxSteps, wp.steps.size());
    plan.maxPairs = std::max<size_t>(plan.maxPairs, offset);
    plan.maxWindow = std::max<size_t>(plan.maxWindow, wp.count);
    plan.windows.push_back(std::move(wp));
  }
  plan.routeMs = timer.ms();
  simulateCache(plan, tiles, cfg.slots, cfg.eviction);
  return plan;
}

QueryOrder parseQueryOrder(const std::string& s) {
  if (s == "original") return QueryOrder::Original;
  if (s == "random") return QueryOrder::Random;
  if (s == "morton") return QueryOrder::Morton;
  if (s == "tiled") return QueryOrder::Tiled;
  throw std::invalid_argument("unknown query order: " + s);
}
Eviction parseEviction(const std::string& s) {
  if (s == "lru") return Eviction::LRU;
  if (s == "belady" || s == "min" || s == "opt") return Eviction::Belady;
  throw std::invalid_argument("unknown eviction policy: " + s);
}
HaloMode parseHaloMode(const std::string& s) {
  if (s == "points") return HaloMode::Points;
  if (s == "queries") return HaloMode::Queries;
  throw std::invalid_argument("unknown halo mode: " + s);
}
const char* toString(QueryOrder o) {
  switch (o) {
    case QueryOrder::Original: return "original";
    case QueryOrder::Random: return "random";
    case QueryOrder::Morton: return "morton";
    default: return "tiled";
  }
}
const char* toString(Eviction e) { return e == Eviction::LRU ? "lru" : "belady"; }
const char* toString(HaloMode h) { return h == HaloMode::Points ? "points" : "queries"; }

}  // namespace ooc
