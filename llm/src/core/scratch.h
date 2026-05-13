#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/tensor.h"

namespace modernllm {

// Pre-allocated bump-pointer arena on the device. Used to hand out per-call
// scratch tensors without `cudaMalloc` / `cudaFree` in the hot path.
//
// Lifecycle:
//   1. `initialize(bytes)` allocates one big buffer on the device.
//   2. Each call to `allocate()` returns a non-owning Tensor view into it,
//      bumping the offset.
//   3. `reset()` rolls the offset back to zero. All previously-handed-out
//      Tensors are now invalid; callers must not retain them across reset.
//
// Allocations are aligned to 256 bytes (cuBLAS happiness).
//
// Not thread-safe and not stream-safe in any deep sense — intended to be used
// by a single training thread on a single CUDA stream where the typical
// pattern is "scratch within one forward/backward then reset between steps".
class ScratchArena {
   public:
    ScratchArena() = default;
    ~ScratchArena();

    ScratchArena(const ScratchArena&) = delete;
    ScratchArena& operator=(const ScratchArena&) = delete;

    // Allocate the underlying device buffer. Idempotent if same size; throws
    // on shrink (call free_buffer() first).
    void initialize(std::size_t bytes);
    void free_buffer();

    // Reset the offset pointer to 0. Cheap (just an integer assignment).
    void reset() noexcept { offset_ = 0; }

    // Hand out a tensor view of `shape` / `dtype` into the arena.
    // Throws if the request would overflow the buffer.
    Tensor allocate(std::vector<std::int64_t> shape, DType dtype);

    std::size_t capacity() const noexcept { return capacity_; }
    std::size_t in_use() const noexcept { return offset_; }

   private:
    void* base_ = nullptr;
    std::size_t capacity_ = 0;
    std::size_t offset_ = 0;
};

}  // namespace modernllm
