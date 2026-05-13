#include "train/loss.h"

#include <cmath>
#include <stdexcept>
#include <vector>

#include "core/cuda_check.h"

namespace modernllm {

namespace {

// One block per row. blockDim.x must be a power of two.
//
// Each thread strides over the V dimension. We do three passes:
//   1) row max (block reduction)
//   2) row sum of exp(z - max) (block reduction)
//   3) write dlogits and (thread 0) the per-row loss
__global__ void softmax_ce_kernel(const float* __restrict__ logits,
                                  const int* __restrict__ targets,
                                  float* __restrict__ loss_per_row,
                                  float* __restrict__ dlogits,
                                  int V, int B, float scale_grad) {
    int b = blockIdx.x;
    if (b >= B) return;

    extern __shared__ float smem[];

    const float* row = logits + static_cast<long long>(b) * V;
    int target = targets[b];

    // Pass 1: max
    float local_max = -INFINITY;
    for (int v = threadIdx.x; v < V; v += blockDim.x) {
        local_max = fmaxf(local_max, row[v]);
    }
    smem[threadIdx.x] = local_max;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (threadIdx.x < off) {
            smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + off]);
        }
        __syncthreads();
    }
    float row_max = smem[0];
    __syncthreads();

    // Pass 2: sum exp
    float local_sum = 0.f;
    for (int v = threadIdx.x; v < V; v += blockDim.x) {
        local_sum += expf(row[v] - row_max);
    }
    smem[threadIdx.x] = local_sum;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (threadIdx.x < off) {
            smem[threadIdx.x] += smem[threadIdx.x + off];
        }
        __syncthreads();
    }
    float row_sum = smem[0];
    __syncthreads();

    if (threadIdx.x == 0) {
        // loss = log_sum_exp(logits) - logits[target]
        //      = row_max + log(row_sum) - row[target]
        loss_per_row[b] = row_max + logf(row_sum) - row[target];
    }

    // Pass 3: gradient
    float inv_sum = 1.f / row_sum;
    float* drow = dlogits + static_cast<long long>(b) * V;
    for (int v = threadIdx.x; v < V; v += blockDim.x) {
        float prob = expf(row[v] - row_max) * inv_sum;
        float ind = (v == target) ? 1.f : 0.f;
        drow[v] = (prob - ind) * scale_grad;
    }
}

}  // namespace

void softmax_ce_forward_backward(const Tensor& logits,
                                  const Tensor& targets,
                                  Tensor& loss_per_row,
                                  Tensor& dlogits) {
    if (logits.ndim() != 2)
        throw std::invalid_argument("logits must be 2D [B, V]");
    if (targets.ndim() != 1)
        throw std::invalid_argument("targets must be 1D [B]");
    if (logits.dtype() != DType::FP32 || dlogits.dtype() != DType::FP32 ||
        loss_per_row.dtype() != DType::FP32) {
        throw std::invalid_argument("CE loss: only FP32 supported");
    }
    if (logits.device() != Device::Cuda || targets.device() != Device::Cuda ||
        loss_per_row.device() != Device::Cuda ||
        dlogits.device() != Device::Cuda) {
        throw std::invalid_argument("CE loss: all tensors must be CUDA");
    }

    int B = static_cast<int>(logits.shape()[0]);
    int V = static_cast<int>(logits.shape()[1]);
    if (targets.numel() != B || loss_per_row.numel() != B) {
        throw std::invalid_argument("CE loss: B mismatch");
    }
    if (dlogits.numel() != logits.numel()) {
        throw std::invalid_argument("CE loss: dlogits shape mismatch");
    }

    const int block = 256;  // power of two for the reduction
    const float scale = 1.f / static_cast<float>(B);
    softmax_ce_kernel<<<static_cast<unsigned>(B), block, block * sizeof(float)>>>(
        logits.data_as<float>(),
        targets.data_as<int>(),
        loss_per_row.data_as<float>(),
        dlogits.data_as<float>(),
        V, B, scale);
    MLLM_CUDA_CHECK(cudaGetLastError());
}

float reduce_mean_to_scalar(const Tensor& loss_per_row) {
    Tensor h = loss_per_row.to(Device::Host);
    const float* p = h.data_as<float>();
    double s = 0.0;
    for (std::int64_t i = 0; i < h.numel(); ++i) s += p[i];
    return static_cast<float>(s / static_cast<double>(h.numel()));
}

}  // namespace modernllm
