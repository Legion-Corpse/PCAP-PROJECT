// Host wall-clock timer and a scoped NVTX range for Nsight Systems timelines.
#pragma once

#include <chrono>
#include <nvtx3/nvToolsExt.h>

namespace ooc {

class Timer {
 public:
  Timer() : start_(Clock::now()) {}
  void reset() { start_ = Clock::now(); }
  double ms() const { return std::chrono::duration<double, std::milli>(Clock::now() - start_).count(); }

 private:
  using Clock = std::chrono::steady_clock;
  Clock::time_point start_;
};

class NvtxRange {
 public:
  explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
  ~NvtxRange() { nvtxRangePop(); }
  NvtxRange(const NvtxRange&) = delete;
  NvtxRange& operator=(const NvtxRange&) = delete;
};

}  // namespace ooc
