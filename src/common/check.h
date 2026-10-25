// Error-checking helpers for CUDA runtime and OptiX calls.
#pragma once

#include <cuda_runtime.h>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ooc {

// Thrown when a device allocation fails, so callers can report "does not fit"
// (the whole point of the out-of-core path) instead of crashing.
struct DeviceOutOfMemory : std::runtime_error {
  using std::runtime_error::runtime_error;
};

[[noreturn]] inline void throwCuda(cudaError_t err, const char* expr, const char* file, int line) {
  std::ostringstream os;
  os << "CUDA error " << cudaGetErrorName(err) << " (" << cudaGetErrorString(err) << ") at " << file << ":"
     << line << " in `" << expr << "`";
  if (err == cudaErrorMemoryAllocation) {
    cudaGetLastError();  // clear the sticky-free error so the context stays usable
    throw DeviceOutOfMemory(os.str());
  }
  throw std::runtime_error(os.str());
}

}  // namespace ooc

#define CUDA_CHECK(expr)                                                  \
  do {                                                                    \
    cudaError_t ooc_err_ = (expr);                                        \
    if (ooc_err_ != cudaSuccess) ::ooc::throwCuda(ooc_err_, #expr, __FILE__, __LINE__); \
  } while (0)

#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

#define OPTIX_CHECK(expr)                                                             \
  do {                                                                                \
    OptixResult ooc_res_ = (expr);                                                    \
    if (ooc_res_ != OPTIX_SUCCESS) {                                                  \
      std::ostringstream ooc_os_;                                                     \
      ooc_os_ << "OptiX error " << optixGetErrorName(ooc_res_) << " ("               \
              << optixGetErrorString(ooc_res_) << ") at " << __FILE__ << ":" << __LINE__ \
              << " in `" #expr "`";                                                   \
      if (ooc_res_ == OPTIX_ERROR_CUDA_ERROR) cudaGetLastError();                     \
      throw std::runtime_error(ooc_os_.str());                                        \
    }                                                                                 \
  } while (0)
