#include "model/rmsnorm.h"

#include <stdexcept>

#include "core/cuda_check.h"

namespace modernllm {

namespace {

__device__ __forceinline__ float bf16_to_f32_d(unsigned short b) {
    unsigned int bits = static_cast<unsigned int>(b) << 16;
    return __uint_as_float(bits);
}
__device__ __forceinline__ unsigned short f32_to_bf16_rne_d(float f) {
    unsigned int bits = __float_as_uint(f);
    unsigned int lsb = (bits >> 16) & 1u;
    unsigned int rounding_bias = 0x7FFFu + lsb;
    bits += rounding_bias;
    return static_cast<unsigned short>(bits >> 16);
}

// One block per row. blockDim.x must be a power of two.
__global__ void rmsnorm_forward_kernel(const float* __restrict__ x,
                                       const float* __restrict__ gamma,
                                       float* __restrict__ y,
                                       float* __restrict__ rstd_out,
                                       int D, float eps) {
    int n = blockIdx.x;
    extern __shared__ float smem[];
    const float* row = x + static_cast<long long>(n) * D;
    float* yrow = y + static_cast<long long>(n) * D;

    // Sum of squares
    float local = 0.f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float v = row[d];
        local += v * v;
    }
    smem[threadIdx.x] = local;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (threadIdx.x < off) smem[threadIdx.x] += smem[threadIdx.x + off];
        __syncthreads();
    }
    float ms = smem[0] / static_cast<float>(D);
    float rstd = rsqrtf(ms + eps);
    if (threadIdx.x == 0) rstd_out[n] = rstd;

    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        yrow[d] = row[d] * rstd * gamma[d];
    }
}

// Backward.
//   g[d]    = dy[d] * gamma[d]
//   sum_gx  = sum_d g[d] * x[d]
//   dx[k]   = rstd * (g[k] - rstd^2 / D * x[k] * sum_gx)
//   dgamma[d] += dy[d] * x[d] * rstd
__global__ void rmsnorm_backward_kernel(const float* __restrict__ d_y,
                                        const float* __restrict__ x,
                                        const float* __restrict__ gamma,
                                        const float* __restrict__ rstd_in,
                                        float* __restrict__ d_x,
                                        float* __restrict__ d_gamma,
                                        int D) {
    int n = blockIdx.x;
    extern __shared__ float smem[];

    const float* xrow = x + static_cast<long long>(n) * D;
    const float* dyrow = d_y + static_cast<long long>(n) * D;
    float* dxrow = d_x + static_cast<long long>(n) * D;
    float rstd = rstd_in[n];
    float invD = 1.f / static_cast<float>(D);

    // Compute sum_gx = sum_d (dy * gamma * x)
    float local = 0.f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        local += dyrow[d] * gamma[d] * xrow[d];
    }
    smem[threadIdx.x] = local;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (threadIdx.x < off) smem[threadIdx.x] += smem[threadIdx.x + off];
        __syncthreads();
    }
    float sum_gx = smem[0];
    __syncthreads();

    float coeff = rstd * rstd * rstd * invD * sum_gx;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        // Cache dyrow[d] in a local: rmsnorm_backward must support in-place
        // (d_y and d_x pointing to the same memory) for the QK-Norm pattern,
        // so we cannot read d_y after writing d_x.
        float dy_d = dyrow[d];
        float xd  = xrow[d];
        float g = dy_d * gamma[d];
        dxrow[d] = rstd * g - coeff * xd;
        atomicAdd(&d_gamma[d], dy_d * xd * rstd);
    }
}

// BF16 sister kernels — x/y/d_y/d_x stored as BF16; gamma, rstd, d_gamma stay FP32
// (small / numerically sensitive).
__global__ void rmsnorm_forward_bf16_kernel(const unsigned short* __restrict__ x,
                                             const float* __restrict__ gamma,
                                             unsigned short* __restrict__ y,
                                             float* __restrict__ rstd_out,
                                             int D, float eps) {
    int n = blockIdx.x;
    extern __shared__ float smem[];
    long long row_off = static_cast<long long>(n) * D;

    float local = 0.f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float v = bf16_to_f32_d(x[row_off + d]);
        local += v * v;
    }
    smem[threadIdx.x] = local;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (threadIdx.x < off) smem[threadIdx.x] += smem[threadIdx.x + off];
        __syncthreads();
    }
    float ms = smem[0] / static_cast<float>(D);
    float rstd = rsqrtf(ms + eps);
    if (threadIdx.x == 0) rstd_out[n] = rstd;

    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float v = bf16_to_f32_d(x[row_off + d]);
        y[row_off + d] = f32_to_bf16_rne_d(v * rstd * gamma[d]);
    }
}

__global__ void rmsnorm_backward_bf16_kernel(const unsigned short* __restrict__ d_y,
                                              const unsigned short* __restrict__ x,
                                              const float* __restrict__ gamma,
                                              const float* __restrict__ rstd_in,
                                              unsigned short* __restrict__ d_x,
                                              float* __restrict__ d_gamma,
                                              int D) {
    int n = blockIdx.x;
    extern __shared__ float smem[];
    long long row_off = static_cast<long long>(n) * D;
    float rstd = rstd_in[n];
    float invD = 1.f / static_cast<float>(D);

    // sum_gx = sum_d (dy * gamma * x) — all in FP32
    float local = 0.f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float dy_d = bf16_to_f32_d(d_y[row_off + d]);
        float xd = bf16_to_f32_d(x[row_off + d]);
        local += dy_d * gamma[d] * xd;
    }
    smem[threadIdx.x] = local;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (threadIdx.x < off) smem[threadIdx.x] += smem[threadIdx.x + off];
        __syncthreads();
    }
    float sum_gx = smem[0];
    __syncthreads();

    float coeff = rstd * rstd * rstd * invD * sum_gx;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float dy_d = bf16_to_f32_d(d_y[row_off + d]);
        float xd  = bf16_to_f32_d(x[row_off + d]);
        float g = dy_d * gamma[d];
        d_x[row_off + d] = f32_to_bf16_rne_d(rstd * g - coeff * xd);
        atomicAdd(&d_gamma[d], dy_d * xd * rstd);
    }
}

}  // namespace

void rmsnorm_forward(const Tensor& x, const Tensor& gamma, float eps,
                      Tensor& y, Tensor& rstd) {
    if (x.ndim() != 2 || y.shape() != x.shape())
        throw std::invalid_argument("rmsnorm_fwd: x/y shape");
    if (x.dtype() != y.dtype())
        throw std::invalid_argument("rmsnorm_fwd: x/y dtype mismatch");
    int N = static_cast<int>(x.shape()[0]);
    int D = static_cast<int>(x.shape()[1]);
    if (gamma.numel() != D || rstd.numel() != N)
        throw std::invalid_argument("rmsnorm_fwd: gamma/rstd shape");
    if (gamma.dtype() != DType::FP32 || rstd.dtype() != DType::FP32)
        throw std::invalid_argument("rmsnorm_fwd: gamma/rstd must be FP32");

    const int block = 256;
    if (x.dtype() == DType::FP32) {
        rmsnorm_forward_kernel<<<static_cast<unsigned>(N), block,
                                  block * sizeof(float)>>>(
            x.data_as<float>(), gamma.data_as<float>(),
            y.data_as<float>(), rstd.data_as<float>(),
            D, eps);
    } else if (x.dtype() == DType::BF16) {
        rmsnorm_forward_bf16_kernel<<<static_cast<unsigned>(N), block,
                                       block * sizeof(float)>>>(
            static_cast<const unsigned short*>(x.data()),
            gamma.data_as<float>(),
            static_cast<unsigned short*>(y.data()),
            rstd.data_as<float>(),
            D, eps);
    } else {
        throw std::invalid_argument("rmsnorm_fwd: dtype must be FP32 or BF16");
    }
    MLLM_CUDA_CHECK(cudaGetLastError());
}

void rmsnorm_backward(const Tensor& d_y, const Tensor& x, const Tensor& gamma,
                       const Tensor& rstd,
                       Tensor& d_x, Tensor& d_gamma) {
    if (d_y.shape() != x.shape() || d_x.shape() != x.shape())
        throw std::invalid_argument("rmsnorm_bwd: shape mismatch");
    if (d_y.dtype() != x.dtype() || x.dtype() != d_x.dtype())
        throw std::invalid_argument("rmsnorm_bwd: x/y/d_y dtype mismatch");
    int N = static_cast<int>(x.shape()[0]);
    int D = static_cast<int>(x.shape()[1]);
    if (gamma.numel() != D || d_gamma.numel() != D || rstd.numel() != N)
        throw std::invalid_argument("rmsnorm_bwd: gamma/rstd shape");
    if (gamma.dtype() != DType::FP32 || rstd.dtype() != DType::FP32 ||
        d_gamma.dtype() != DType::FP32) {
        throw std::invalid_argument(
            "rmsnorm_bwd: gamma/rstd/d_gamma must be FP32");
    }

    const int block = 256;
    if (x.dtype() == DType::FP32) {
        rmsnorm_backward_kernel<<<static_cast<unsigned>(N), block,
                                   block * sizeof(float)>>>(
            d_y.data_as<float>(), x.data_as<float>(), gamma.data_as<float>(),
            rstd.data_as<float>(),
            d_x.data_as<float>(), d_gamma.data_as<float>(),
            D);
    } else if (x.dtype() == DType::BF16) {
        rmsnorm_backward_bf16_kernel<<<static_cast<unsigned>(N), block,
                                        block * sizeof(float)>>>(
            static_cast<const unsigned short*>(d_y.data()),
            static_cast<const unsigned short*>(x.data()),
            gamma.data_as<float>(),
            rstd.data_as<float>(),
            static_cast<unsigned short*>(d_x.data()),
            d_gamma.data_as<float>(),
            D);
    } else {
        throw std::invalid_argument("rmsnorm_bwd: dtype must be FP32 or BF16");
    }
    MLLM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace modernllm
