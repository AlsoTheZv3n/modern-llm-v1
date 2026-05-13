#include "model/layernorm.h"

#include <stdexcept>

#include "core/cuda_check.h"

namespace modernllm {

namespace {

// One block per row. blockDim.x must be a power of two.
// Each thread strides over D. Two block reductions: mean and (x-mean)^2.
__global__ void layernorm_forward_kernel(const float* __restrict__ x,
                                         const float* __restrict__ gamma,
                                         const float* __restrict__ beta,
                                         float* __restrict__ y,
                                         float* __restrict__ mean_out,
                                         float* __restrict__ rstd_out,
                                         int D, float eps) {
    int n = blockIdx.x;
    extern __shared__ float smem[];
    const float* row = x + static_cast<long long>(n) * D;
    float* yrow = y + static_cast<long long>(n) * D;

    // Pass 1: sum
    float local_sum = 0.f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) local_sum += row[d];
    smem[threadIdx.x] = local_sum;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (threadIdx.x < off) smem[threadIdx.x] += smem[threadIdx.x + off];
        __syncthreads();
    }
    float mean = smem[0] / static_cast<float>(D);
    __syncthreads();

    // Pass 2: sum of squared diffs
    float local_sq = 0.f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float diff = row[d] - mean;
        local_sq += diff * diff;
    }
    smem[threadIdx.x] = local_sq;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (threadIdx.x < off) smem[threadIdx.x] += smem[threadIdx.x + off];
        __syncthreads();
    }
    float var = smem[0] / static_cast<float>(D);
    float rstd = rsqrtf(var + eps);

    if (threadIdx.x == 0) {
        mean_out[n] = mean;
        rstd_out[n] = rstd;
    }

    // Pass 3: write y
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float xhat = (row[d] - mean) * rstd;
        yrow[d] = xhat * gamma[d] + beta[d];
    }
}

// Per-row backward kernel. Computes d_x and atomicAdds into d_gamma, d_beta.
__global__ void layernorm_backward_kernel(const float* __restrict__ d_y,
                                          const float* __restrict__ x,
                                          const float* __restrict__ gamma,
                                          const float* __restrict__ mean_in,
                                          const float* __restrict__ rstd_in,
                                          float* __restrict__ d_x,
                                          float* __restrict__ d_gamma,
                                          float* __restrict__ d_beta,
                                          int D) {
    int n = blockIdx.x;
    extern __shared__ float smem[];

    const float* xrow = x + static_cast<long long>(n) * D;
    const float* dyrow = d_y + static_cast<long long>(n) * D;
    float* dxrow = d_x + static_cast<long long>(n) * D;

    float mean = mean_in[n];
    float rstd = rstd_in[n];
    float invD = 1.f / static_cast<float>(D);

    // Pass 1: sum1 = sum_d (dy * gamma)
    float s1 = 0.f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        s1 += dyrow[d] * gamma[d];
    }
    smem[threadIdx.x] = s1;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (threadIdx.x < off) smem[threadIdx.x] += smem[threadIdx.x + off];
        __syncthreads();
    }
    float sum1 = smem[0];
    __syncthreads();

    // Pass 2: sum2 = sum_d (dy * gamma * xhat)
    float s2 = 0.f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float xhat = (xrow[d] - mean) * rstd;
        s2 += dyrow[d] * gamma[d] * xhat;
    }
    smem[threadIdx.x] = s2;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (threadIdx.x < off) smem[threadIdx.x] += smem[threadIdx.x + off];
        __syncthreads();
    }
    float sum2 = smem[0];
    __syncthreads();

    // Pass 3: write d_x and atomicAdd to d_gamma, d_beta
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float xhat = (xrow[d] - mean) * rstd;
        float g = dyrow[d] * gamma[d];
        // dx = rstd * (g - sum1/D - xhat * sum2 / D)
        dxrow[d] = rstd * (g - invD * (sum1 + xhat * sum2));
        atomicAdd(&d_gamma[d], dyrow[d] * xhat);
        atomicAdd(&d_beta[d], dyrow[d]);
    }
}

void check_2d_fp32_cuda(const Tensor& t, const char* name) {
    if (t.ndim() != 2 || t.dtype() != DType::FP32 ||
        t.device() != Device::Cuda) {
        throw std::invalid_argument(std::string("layernorm: bad ") + name);
    }
}

}  // namespace

void layernorm_forward(const Tensor& x, const Tensor& gamma,
                        const Tensor& beta, float eps,
                        Tensor& y, Tensor& mean, Tensor& rstd) {
    check_2d_fp32_cuda(x, "x");
    check_2d_fp32_cuda(y, "y");
    int N = static_cast<int>(x.shape()[0]);
    int D = static_cast<int>(x.shape()[1]);
    if (y.shape() != x.shape()) throw std::invalid_argument("ln: y shape");
    if (gamma.numel() != D || beta.numel() != D)
        throw std::invalid_argument("ln: gamma/beta shape");
    if (mean.numel() != N || rstd.numel() != N)
        throw std::invalid_argument("ln: mean/rstd shape");

    const int block = 256;
    layernorm_forward_kernel<<<static_cast<unsigned>(N), block,
                                block * sizeof(float)>>>(
        x.data_as<float>(), gamma.data_as<float>(), beta.data_as<float>(),
        y.data_as<float>(), mean.data_as<float>(), rstd.data_as<float>(),
        D, eps);
    MLLM_CUDA_CHECK(cudaGetLastError());
}

void layernorm_backward(const Tensor& d_y, const Tensor& x,
                         const Tensor& gamma,
                         const Tensor& mean, const Tensor& rstd,
                         Tensor& d_x, Tensor& d_gamma, Tensor& d_beta) {
    check_2d_fp32_cuda(d_y, "d_y");
    check_2d_fp32_cuda(x, "x");
    check_2d_fp32_cuda(d_x, "d_x");
    int N = static_cast<int>(x.shape()[0]);
    int D = static_cast<int>(x.shape()[1]);
    if (d_y.shape() != x.shape() || d_x.shape() != x.shape())
        throw std::invalid_argument("ln_bwd: shape mismatch");
    if (gamma.numel() != D || d_gamma.numel() != D || d_beta.numel() != D)
        throw std::invalid_argument("ln_bwd: gamma/d_gamma/d_beta shape");
    if (mean.numel() != N || rstd.numel() != N)
        throw std::invalid_argument("ln_bwd: mean/rstd shape");

    const int block = 256;
    layernorm_backward_kernel<<<static_cast<unsigned>(N), block,
                                 block * sizeof(float)>>>(
        d_y.data_as<float>(), x.data_as<float>(), gamma.data_as<float>(),
        mean.data_as<float>(), rstd.data_as<float>(),
        d_x.data_as<float>(), d_gamma.data_as<float>(),
        d_beta.data_as<float>(), D);
    MLLM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace modernllm
