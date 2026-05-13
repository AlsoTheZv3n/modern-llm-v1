#include "train/grad_utils.h"

#include <cmath>
#include <stdexcept>

#include "core/cuda_check.h"

namespace modernllm {

namespace {

// Block reduction over input, atomicAdd to *out (single FP32 accumulator).
// blockDim.x must be a power of two.
__global__ void sum_sq_atomic_kernel(const float* __restrict__ x,
                                     float* __restrict__ out,
                                     std::int64_t n) {
    extern __shared__ float smem[];
    std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
    float v = (i < n) ? x[i] : 0.f;
    smem[threadIdx.x] = v * v;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (threadIdx.x < off) smem[threadIdx.x] += smem[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out, smem[0]);
}

__global__ void scale_inplace_kernel(float* __restrict__ x, float s,
                                      std::int64_t n) {
    std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
    if (i < n) x[i] *= s;
}

}  // namespace

float compute_grad_norm(const std::vector<Tensor*>& grads) {
    Tensor scalar = Tensor::zeros({1}, DType::FP32, Device::Cuda);

    const int block = 256;
    for (auto* g : grads) {
        if (!g || g->numel() == 0) continue;
        if (g->dtype() != DType::FP32 || g->device() != Device::Cuda) {
            throw std::invalid_argument(
                "compute_grad_norm: tensors must be FP32 CUDA");
        }
        std::int64_t n = g->numel();
        unsigned grid = static_cast<unsigned>((n + block - 1) / block);
        sum_sq_atomic_kernel<<<grid, block, block * sizeof(float)>>>(
            g->data_as<float>(), scalar.data_as<float>(), n);
        MLLM_CUDA_CHECK(cudaGetLastError());
    }

    Tensor h = scalar.to(Device::Host);
    float sq = *h.data_as<float>();
    return std::sqrt(sq);
}

void scale_grads(std::vector<Tensor*>& grads, float scale) {
    const int block = 256;
    for (auto* g : grads) {
        if (!g || g->numel() == 0) continue;
        std::int64_t n = g->numel();
        unsigned grid = static_cast<unsigned>((n + block - 1) / block);
        scale_inplace_kernel<<<grid, block>>>(g->data_as<float>(), scale, n);
        MLLM_CUDA_CHECK(cudaGetLastError());
    }
}

float clip_grad_norm(std::vector<Tensor*>& grads, float max_norm) {
    float n = compute_grad_norm(grads);
    if (n > max_norm && n > 0.f) {
        float s = max_norm / (n + 1e-6f);
        scale_grads(grads, s);
    }
    return n;
}

}  // namespace modernllm
