// RAII wrappers for device and pinned host memory.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <utility>

#include "common/check.h"

namespace ooc {

class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t bytes) { alloc(bytes); }
  ~DeviceBuffer() { release(); }
  DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(std::exchange(o.ptr_, nullptr)), bytes_(std::exchange(o.bytes_, 0)) {}
  DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
    if (this != &o) { release(); ptr_ = std::exchange(o.ptr_, nullptr); bytes_ = std::exchange(o.bytes_, 0); }
    return *this;
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  void alloc(size_t bytes) {
    release();
    if (bytes) CUDA_CHECK(cudaMalloc(&ptr_, bytes));
    bytes_ = bytes;
  }
  // Grows (never shrinks) the allocation; contents are not preserved.
  void ensure(size_t bytes) { if (bytes > bytes_) alloc(bytes); }
  void release() {
    if (ptr_) cudaFree(ptr_);
    ptr_ = nullptr;
    bytes_ = 0;
  }

  template <class T> T* as() const { return static_cast<T*>(ptr_); }
  void* get() const { return ptr_; }
  uint64_t dptr() const { return reinterpret_cast<uint64_t>(ptr_); }
  size_t bytes() const { return bytes_; }

 private:
  void* ptr_ = nullptr;
  size_t bytes_ = 0;
};

class PinnedBuffer {
 public:
  PinnedBuffer() = default;
  explicit PinnedBuffer(size_t bytes) { alloc(bytes); }
  ~PinnedBuffer() { release(); }
  PinnedBuffer(PinnedBuffer&& o) noexcept : ptr_(std::exchange(o.ptr_, nullptr)), bytes_(std::exchange(o.bytes_, 0)) {}
  PinnedBuffer& operator=(PinnedBuffer&& o) noexcept {
    if (this != &o) { release(); ptr_ = std::exchange(o.ptr_, nullptr); bytes_ = std::exchange(o.bytes_, 0); }
    return *this;
  }
  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;

  void alloc(size_t bytes) {
    release();
    if (bytes) CUDA_CHECK(cudaHostAlloc(&ptr_, bytes, cudaHostAllocDefault));
    bytes_ = bytes;
  }
  void ensure(size_t bytes) { if (bytes > bytes_) alloc(bytes); }
  void release() {
    if (ptr_) cudaFreeHost(ptr_);
    ptr_ = nullptr;
    bytes_ = 0;
  }

  template <class T> T* as() const { return static_cast<T*>(ptr_); }
  void* get() const { return ptr_; }
  size_t bytes() const { return bytes_; }

 private:
  void* ptr_ = nullptr;
  size_t bytes_ = 0;
};

class CudaStream {
 public:
  CudaStream() { CUDA_CHECK(cudaStreamCreateWithFlags(&s_, cudaStreamNonBlocking)); }
  ~CudaStream() { if (s_) cudaStreamDestroy(s_); }
  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;
  cudaStream_t get() const { return s_; }
  operator cudaStream_t() const { return s_; }

 private:
  cudaStream_t s_ = nullptr;
};

class CudaEvent {
 public:
  explicit CudaEvent(bool timing = false) {
    CUDA_CHECK(cudaEventCreateWithFlags(&e_, timing ? cudaEventDefault : cudaEventDisableTiming));
  }
  ~CudaEvent() { if (e_) cudaEventDestroy(e_); }
  CudaEvent(CudaEvent&& o) noexcept : e_(std::exchange(o.e_, nullptr)) {}
  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;
  cudaEvent_t get() const { return e_; }
  operator cudaEvent_t() const { return e_; }

 private:
  cudaEvent_t e_ = nullptr;
};

inline size_t alignUp(size_t v, size_t a) { return (v + a - 1) / a * a; }

}  // namespace ooc
