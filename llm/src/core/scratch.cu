#include "core/scratch.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>

#include "core/cuda_check.h"
#include "core/dtype.h"

namespace modernllm {

namespace {

constexpr std::size_t kAlign = 256;

std::size_t round_up(std::size_t x, std::size_t a) {
    return (x + a - 1) / a * a;
}

std::int64_t numel_of(const std::vector<std::int64_t>& s) {
    std::int64_t n = 1;
    for (auto d : s) {
        if (d < 0) throw std::invalid_argument("scratch: negative dim");
        n *= d;
    }
    return n;
}

}  // namespace

ScratchArena::~ScratchArena() {
    free_buffer();
}

void ScratchArena::initialize(std::size_t bytes) {
    if (bytes == 0) {
        free_buffer();
        return;
    }
    if (base_ && capacity_ == bytes) return;
    if (base_) free_buffer();
    MLLM_CUDA_CHECK(cudaMalloc(&base_, bytes));
    capacity_ = bytes;
    offset_ = 0;
}

void ScratchArena::free_buffer() {
    if (!base_) return;
    cudaError_t e = cudaFree(base_);
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[scratch] cudaFree failed: %s\n",
                     cudaGetErrorString(e));
    }
    base_ = nullptr;
    capacity_ = 0;
    offset_ = 0;
}

Tensor ScratchArena::allocate(std::vector<std::int64_t> shape, DType dtype) {
    std::int64_t n = numel_of(shape);
    std::size_t bytes = static_cast<std::size_t>(n) * dtype_bytes(dtype);
    std::size_t aligned_off = round_up(offset_, kAlign);
    if (aligned_off + bytes > capacity_) {
        throw std::runtime_error(
            "ScratchArena overflow: requested " + std::to_string(bytes) +
            " B, capacity " + std::to_string(capacity_) +
            " B, in_use " + std::to_string(offset_) + " B");
    }
    void* ptr = static_cast<char*>(base_) + aligned_off;
    offset_ = aligned_off + bytes;
    return Tensor::from_blob(ptr, std::move(shape), dtype, Device::Cuda);
}

}  // namespace modernllm
