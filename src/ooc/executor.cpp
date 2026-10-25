#include "ooc/executor.h"

#include <algorithm>
#include <cstring>
#include <memory>

#include "common/host_utils.h"
#include "common/timer.h"
#include "cuda/buffers.h"
#include "cuda/kernels.h"
#include "rt/optix_engine.h"

namespace ooc {
namespace {

// Device + pinned buffers for one in-flight window.
struct WindowBuffers {
  DeviceBuffer q, lists, counts, nbrs, params;
  PinnedBuffer hq, hlists, hcounts, hnbrs, hparams;
  CudaEvent uploaded, computed, downloaded;
  std::vector<CudaEvent> ev;  // timing pairs: [load s/e]* [launch s/e]* + upload + d2h
  size_t evUsed = 0;
  std::vector<std::pair<int, int>> loadPairs, launchPairs;
  int uploadPair = -1, d2hPair = -1;
  size_t first = 0;
  uint32_t count = 0;

  int pair() {
    if (evUsed + 2 > ev.size()) {
      ev.reserve(std::max<size_t>(16, ev.size() * 2));
      while (ev.size() < evUsed + 2) ev.emplace_back(true);
    }
    const int p = int(evUsed);
    evUsed += 2;
    return p;
  }
};

float elapsed(const CudaEvent& a, const CudaEvent& b) {
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
  return ms;
}

}  // namespace

size_t windowBufferBytes(size_t W, size_t P, size_t ST, uint32_t K) {
  return 2 * (sizeof(float4) * W + 4 * P + 4 * W + 4ull * K * W + sizeof(LaunchParams) * ST + 3 * 256);
}

void executePlan(OptixEngine& engine, const TileSet& tiles, const Plan& plan, const std::vector<float3>& queries,
                 const SearchParams& params, const ExecConfig& cfg, SearchResult& out, Metrics& m) {
  NvtxRange range("execute");
  const uint32_t K = params.maxNeighbors;
  const float r2 = params.radius * params.radius;
  out.reset(queries.size(), K);
  if (plan.windows.empty()) return;

  const int S = std::max(1, cfg.slots);
  const size_t W = std::max<size_t>(1, plan.maxWindow), P = std::max<size_t>(1, plan.maxPairs);
  const size_t ST = std::max<size_t>(1, plan.maxSteps);

  std::vector<DeviceBuffer> slot(static_cast<size_t>(S));
  for (auto& s : slot) s.alloc(alignUp(std::max<size_t>(tiles.maxBlobBytes, 256), 256));
  DeviceBuffer dSlotHandles(sizeof(OptixTraversableHandle) * size_t(S));
  std::vector<std::unique_ptr<CudaEvent>> tileReady, slotFree;
  for (int s = 0; s < S; ++s) {
    tileReady.push_back(std::make_unique<CudaEvent>());
    slotFree.push_back(std::make_unique<CudaEvent>());
  }

  WindowBuffers win[2];
  for (auto& b : win) {
    b.q.alloc(sizeof(float4) * W);
    b.lists.alloc(4 * P);
    b.counts.alloc(4 * W);
    if (K) b.nbrs.alloc(4ull * K * W);
    b.params.alloc(sizeof(LaunchParams) * ST);
    b.hq.alloc(sizeof(float4) * W);
    b.hlists.alloc(4 * P);
    b.hcounts.alloc(4 * W);
    if (K) b.hnbrs.alloc(4ull * K * W);
    b.hparams.alloc(sizeof(LaunchParams) * ST);
  }

  CudaStream copyStream, computeStream, d2hStream;
  const cudaStream_t sCopy = cfg.overlap ? copyStream.get() : computeStream.get();
  const cudaStream_t sComp = computeStream.get();
  const cudaStream_t sD2H = cfg.overlap ? d2hStream.get() : computeStream.get();

  double tileMs = 0, launchMs = 0, uploadMs = 0, d2hMs = 0, hostPackMs = 0, hostScatterMs = 0;
  uint64_t bytesH2D = 0, launches = 0;

  auto issue = [&](size_t wi) {
    const WindowPlan& wp = plan.windows[wi];
    WindowBuffers& b = win[wi % 2];
    b.first = wp.first;
    b.count = wp.count;
    b.evUsed = 0;
    b.loadPairs.clear();
    b.launchPairs.clear();

    // Launch parameters of every step are known on the host up front; the
    // traversable handle is read through slotHandles, which the compute
    // stream rewrites after relocating a freshly loaded tile.
    Timer pack;
    if (plan.queries.empty()) packPoints(queries.data(), plan.order.data() + wp.first, wp.count, b.hq.as<float4>());
    else packPoints(plan.queries.data() + wp.first, nullptr, wp.count, b.hq.as<float4>());
    std::memcpy(b.hlists.get(), wp.lists.data(), 4 * wp.lists.size());
    LaunchParams* hp = b.hparams.as<LaunchParams>();
    for (size_t k = 0; k < wp.steps.size(); ++k) {
      const PlanStep& st = wp.steps[k];
      const TileHost& tile = tiles.tiles[st.tile];
      LaunchParams& lp = hp[k];
      lp = LaunchParams{};
      lp.handleSlot = dSlotHandles.as<OptixTraversableHandle>() + st.slot;
      lp.points = reinterpret_cast<const float4*>(slot[st.slot].as<char>() + tile.pointsOffset);
      lp.group = tiles.group;
      if (wp.contiguous) {  // slots [listOffset, listOffset + listCount): offset the base pointers
        lp.queries = b.q.as<float4>() + st.listOffset;
        lp.queryList = nullptr;
        lp.counts = b.counts.as<uint32_t>() + st.listOffset;
        lp.neighbors = K ? b.nbrs.as<uint32_t>() + size_t(st.listOffset) * K : nullptr;
      } else {
        lp.queries = b.q.as<float4>();
        lp.queryList = b.lists.as<uint32_t>() + st.listOffset;
        lp.counts = b.counts.as<uint32_t>();
        lp.neighbors = K ? b.nbrs.as<uint32_t>() : nullptr;
      }
      lp.numQueries = st.listCount;
      lp.maxNeighbors = K;
      lp.radius2 = r2;
    }
    hostPackMs += pack.ms();

    CUDA_CHECK(cudaStreamWaitEvent(sCopy, b.downloaded, 0));  // buffers of window wi-2 are free
    if (cfg.profile) { b.uploadPair = b.pair(); CUDA_CHECK(cudaEventRecord(b.ev[size_t(b.uploadPair)], sCopy)); }
    CUDA_CHECK(cudaMemcpyAsync(b.q.get(), b.hq.get(), sizeof(float4) * wp.count, cudaMemcpyHostToDevice, sCopy));
    if (!wp.lists.empty())
      CUDA_CHECK(cudaMemcpyAsync(b.lists.get(), b.hlists.get(), 4 * wp.lists.size(), cudaMemcpyHostToDevice, sCopy));
    CUDA_CHECK(cudaMemcpyAsync(b.params.get(), b.hparams.get(), sizeof(LaunchParams) * wp.steps.size(),
                               cudaMemcpyHostToDevice, sCopy));
    if (cfg.profile) CUDA_CHECK(cudaEventRecord(b.ev[size_t(b.uploadPair) + 1], sCopy));
    CUDA_CHECK(cudaEventRecord(b.uploaded, sCopy));
    bytesH2D += sizeof(float4) * wp.count + 4 * wp.lists.size() + sizeof(LaunchParams) * wp.steps.size();

    CUDA_CHECK(cudaStreamWaitEvent(sComp, b.uploaded, 0));
    CUDA_CHECK(cudaMemsetAsync(b.counts.get(), 0, 4 * wp.count, sComp));

    for (size_t k = 0; k < wp.steps.size(); ++k) {
      const PlanStep& st = wp.steps[k];
      const TileHost& tile = tiles.tiles[st.tile];
      char* base = slot[st.slot].as<char>();
      if (st.load) {
        // Copy stream: DMA only, so it never waits for SMs held by a traversal.
        CUDA_CHECK(cudaStreamWaitEvent(sCopy, *slotFree[st.slot], 0));
        int pr = -1;
        if (cfg.profile) { pr = b.pair(); CUDA_CHECK(cudaEventRecord(b.ev[size_t(pr)], sCopy)); }
        CUDA_CHECK(cudaMemcpyAsync(base, tile.blob, tile.blobBytes, cudaMemcpyHostToDevice, sCopy));
        if (cfg.profile) { CUDA_CHECK(cudaEventRecord(b.ev[size_t(pr) + 1], sCopy)); b.loadPairs.emplace_back(pr, pr + 1); }
        CUDA_CHECK(cudaEventRecord(*tileReady[st.slot], sCopy));
        bytesH2D += tile.blobBytes;
        // Compute stream: fix up the GAS pointers, then publish the new handle.
        CUDA_CHECK(cudaStreamWaitEvent(sComp, *tileReady[st.slot], 0));
        const OptixTraversableHandle h =
            engine.relocate(tile.reloc, reinterpret_cast<uint64_t>(base), tile.gasBytes, sComp);
        launchWriteU64(reinterpret_cast<uint64_t*>(dSlotHandles.as<OptixTraversableHandle>() + st.slot), h, sComp);
      }
      LaunchParams* dParams = b.params.as<LaunchParams>() + k;
      int pr = -1;
      if (cfg.profile) { pr = b.pair(); CUDA_CHECK(cudaEventRecord(b.ev[size_t(pr)], sComp)); }
      engine.launch(reinterpret_cast<uint64_t>(dParams), st.listCount, sComp);
      if (cfg.profile) { CUDA_CHECK(cudaEventRecord(b.ev[size_t(pr) + 1], sComp)); b.launchPairs.emplace_back(pr, pr + 1); }
      CUDA_CHECK(cudaEventRecord(*slotFree[st.slot], sComp));
      ++launches;
    }
    CUDA_CHECK(cudaEventRecord(b.computed, sComp));

    CUDA_CHECK(cudaStreamWaitEvent(sD2H, b.computed, 0));
    if (cfg.profile) { b.d2hPair = b.pair(); CUDA_CHECK(cudaEventRecord(b.ev[size_t(b.d2hPair)], sD2H)); }
    CUDA_CHECK(cudaMemcpyAsync(b.hcounts.get(), b.counts.get(), 4 * wp.count, cudaMemcpyDeviceToHost, sD2H));
    if (K) CUDA_CHECK(cudaMemcpyAsync(b.hnbrs.get(), b.nbrs.get(), 4ull * K * wp.count, cudaMemcpyDeviceToHost, sD2H));
    if (cfg.profile) CUDA_CHECK(cudaEventRecord(b.ev[size_t(b.d2hPair) + 1], sD2H));
    CUDA_CHECK(cudaEventRecord(b.downloaded, sD2H));
  };

  auto finish = [&](size_t wi) {
    WindowBuffers& b = win[wi % 2];
    CUDA_CHECK(cudaEventSynchronize(b.downloaded));
    if (cfg.profile) {
      for (auto [s, e] : b.loadPairs) tileMs += elapsed(b.ev[size_t(s)], b.ev[size_t(e)]);
      for (auto [s, e] : b.launchPairs) launchMs += elapsed(b.ev[size_t(s)], b.ev[size_t(e)]);
      uploadMs += elapsed(b.ev[size_t(b.uploadPair)], b.ev[size_t(b.uploadPair) + 1]);
      d2hMs += elapsed(b.ev[size_t(b.d2hPair)], b.ev[size_t(b.d2hPair) + 1]);
    }
    Timer scatter;
    scatterResults(plan.order.data() + b.first, b.count, b.hcounts.as<uint32_t>(), b.hnbrs.as<uint32_t>(), out);
    hostScatterMs += scatter.ms();
  };

  CUDA_CHECK(cudaDeviceSynchronize());
  Timer wall;
  for (size_t wi = 0; wi < plan.windows.size(); ++wi) {
    issue(wi);
    if (wi > 0) finish(wi - 1);
  }
  finish(plan.windows.size() - 1);
  CUDA_CHECK(cudaDeviceSynchronize());
  const double execMs = wall.ms();

  m.set("exec_ms", execMs);
  m.set("slots", double(S));
  m.set("launches", double(launches));
  m.set("h2d_bytes", double(bytesH2D));
  m.set("host_pack_ms", hostPackMs);
  m.set("host_scatter_ms", hostScatterMs);
  if (cfg.profile) {
    m.set("tile_h2d_ms", tileMs);
    m.set("traverse_ms", launchMs);
    m.set("query_h2d_ms", uploadMs);
    m.set("result_d2h_ms", d2hMs);
    m.set("tile_h2d_gbps", tileMs > 0 ? double(plan.loadBytes) / (tileMs * 1e6) : 0.0);
    // Share of tile-transfer time that did not extend the run: compute stream
    // busy time plus exposed transfer equals the execution time.
    const double serial = tileMs + launchMs + uploadMs + d2hMs;
    m.set("serial_sum_ms", serial);
    const double exposed = std::max(0.0, execMs - launchMs);
    m.set("transfer_hidden_fraction", tileMs > 0 ? std::clamp(1.0 - exposed / (tileMs + uploadMs + d2hMs), 0.0, 1.0) : 1.0);
  }
}

}  // namespace ooc
