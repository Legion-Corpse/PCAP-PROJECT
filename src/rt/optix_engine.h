// OptiX context, pipeline and acceleration-structure management.
#pragma once

#include <optix_types.h>

#include <cstdint>
#include <string>
#include <utility>

#include "cuda/buffers.h"
#include "rt/launch_params.h"

namespace ooc {

// Reusable device scratch for repeated GAS builds (one tile after another).
struct GasScratch {
  DeviceBuffer aabbs, temp, output, compactedSize;
};

struct GasInfo {
  OptixTraversableHandle handle = 0;
  size_t bytes = 0;  // compacted size
};

class OptixEngine {
 public:
  explicit OptixEngine(bool validation = false);
  ~OptixEngine();
  OptixEngine(const OptixEngine&) = delete;
  OptixEngine& operator=(const OptixEngine&) = delete;

  OptixDeviceContext context() const { return context_; }

  // Builds a compacted GAS over `n` device points, one primitive per `group`
  // consecutive points (the buffer must hold paddedCount(n, group) entries).
  // The compacted structure is written to `dst` (capacity `dstBytes`, 128-B aligned).
  // Synchronizes `stream` (the compacted size must be read back).
  GasInfo buildGas(const float4* dPoints, uint32_t n, float halfWidth, uint64_t dst, size_t dstBytes,
                   GasScratch& scratch, cudaStream_t stream, uint32_t group = 1);

  // Upper bound of the compacted size for `n` primitives (= uncompacted output size).
  size_t gasOutputBound(uint32_t n);

  // Same as buildGas, but allocates `out` to exactly the compacted size.
  GasInfo buildGasOwned(const float4* dPoints, uint32_t n, float halfWidth, DeviceBuffer& out, GasScratch& scratch,
                        cudaStream_t stream, uint32_t group = 1);

  OptixRelocationInfo relocationInfo(OptixTraversableHandle handle);
  // Fixes up a GAS whose bytes were copied to `dst`; stream-ordered.
  OptixTraversableHandle relocate(const OptixRelocationInfo& info, uint64_t dst, size_t bytes, cudaStream_t stream);

  // Launches `width` rays with the parameters stored at device address `dParams`.
  void launch(uint64_t dParams, uint32_t width, cudaStream_t stream);

  std::string deviceName() const { return deviceName_; }

 private:
  void createPipeline();
  // Builds into scratch.output; returns the handle and the compacted size.
  std::pair<OptixTraversableHandle, size_t> buildUncompacted(const float4* dPoints, uint32_t n, float halfWidth,
                                                             GasScratch& scratch, cudaStream_t stream, uint32_t group);

  OptixDeviceContext context_ = nullptr;
  OptixModule module_ = nullptr;
  OptixProgramGroup raygen_ = nullptr, miss_ = nullptr, hitgroup_ = nullptr;
  OptixPipeline pipeline_ = nullptr;
  OptixShaderBindingTable sbt_{};
  DeviceBuffer sbtRecords_;
  std::string deviceName_;
};

}  // namespace ooc
