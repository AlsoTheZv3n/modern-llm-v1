#include "model/rope.h"

#include <cmath>
#include <stdexcept>
#include <vector>

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

__global__ void rope_inplace_fwd_kernel(float* __restrict__ x,
                                         const float* __restrict__ cos_t,
                                         const float* __restrict__ sin_t,
                                         int N, int T, int d_h_half) {
    long long total = static_cast<long long>(N) * T * d_h_half;
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
    if (idx >= total) return;
    int i = static_cast<int>(idx % d_h_half);
    int t = static_cast<int>((idx / d_h_half) % T);
    int n = static_cast<int>(idx / (static_cast<long long>(T) * d_h_half));

    long long base = (static_cast<long long>(n) * T + t) * (2 * d_h_half);
    float c = cos_t[t * d_h_half + i];
    float s = sin_t[t * d_h_half + i];
    float x0 = x[base + 2 * i];
    float x1 = x[base + 2 * i + 1];
    x[base + 2 * i]     = x0 * c - x1 * s;
    x[base + 2 * i + 1] = x0 * s + x1 * c;
}

__global__ void rope_inplace_bwd_kernel(float* __restrict__ dx,
                                         const float* __restrict__ cos_t,
                                         const float* __restrict__ sin_t,
                                         int N, int T, int d_h_half) {
    long long total = static_cast<long long>(N) * T * d_h_half;
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
    if (idx >= total) return;
    int i = static_cast<int>(idx % d_h_half);
    int t = static_cast<int>((idx / d_h_half) % T);
    int n = static_cast<int>(idx / (static_cast<long long>(T) * d_h_half));

    long long base = (static_cast<long long>(n) * T + t) * (2 * d_h_half);
    float c = cos_t[t * d_h_half + i];
    float s = sin_t[t * d_h_half + i];
    float d0 = dx[base + 2 * i];
    float d1 = dx[base + 2 * i + 1];
    dx[base + 2 * i]     = d0 * c + d1 * s;
    dx[base + 2 * i + 1] = -d0 * s + d1 * c;
}

// BF16 sister kernels — x stored as BF16, cos/sin tables stay FP32.
__global__ void rope_inplace_fwd_bf16_kernel(unsigned short* __restrict__ x,
                                              const float* __restrict__ cos_t,
                                              const float* __restrict__ sin_t,
                                              int N, int T, int d_h_half) {
    long long total = static_cast<long long>(N) * T * d_h_half;
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
    if (idx >= total) return;
    int i = static_cast<int>(idx % d_h_half);
    int t = static_cast<int>((idx / d_h_half) % T);
    int n = static_cast<int>(idx / (static_cast<long long>(T) * d_h_half));

    long long base = (static_cast<long long>(n) * T + t) * (2 * d_h_half);
    float c = cos_t[t * d_h_half + i];
    float s = sin_t[t * d_h_half + i];
    float x0 = bf16_to_f32_d(x[base + 2 * i]);
    float x1 = bf16_to_f32_d(x[base + 2 * i + 1]);
    x[base + 2 * i]     = f32_to_bf16_rne_d(x0 * c - x1 * s);
    x[base + 2 * i + 1] = f32_to_bf16_rne_d(x0 * s + x1 * c);
}

__global__ void rope_inplace_bwd_bf16_kernel(unsigned short* __restrict__ dx,
                                              const float* __restrict__ cos_t,
                                              const float* __restrict__ sin_t,
                                              int N, int T, int d_h_half) {
    long long total = static_cast<long long>(N) * T * d_h_half;
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
    if (idx >= total) return;
    int i = static_cast<int>(idx % d_h_half);
    int t = static_cast<int>((idx / d_h_half) % T);
    int n = static_cast<int>(idx / (static_cast<long long>(T) * d_h_half));

    long long base = (static_cast<long long>(n) * T + t) * (2 * d_h_half);
    float c = cos_t[t * d_h_half + i];
    float s = sin_t[t * d_h_half + i];
    float d0 = bf16_to_f32_d(dx[base + 2 * i]);
    float d1 = bf16_to_f32_d(dx[base + 2 * i + 1]);
    dx[base + 2 * i]     = f32_to_bf16_rne_d(d0 * c + d1 * s);
    dx[base + 2 * i + 1] = f32_to_bf16_rne_d(-d0 * s + d1 * c);
}

}  // namespace

std::pair<Tensor, Tensor> make_rope_cache(int T_max, int d_h, float base,
                                            Device device) {
    if (d_h % 2 != 0) throw std::invalid_argument("d_h must be even");
    int half = d_h / 2;

    std::vector<float> cos_h(static_cast<std::size_t>(T_max) * half);
    std::vector<float> sin_h(static_cast<std::size_t>(T_max) * half);
    for (int t = 0; t < T_max; ++t) {
        for (int i = 0; i < half; ++i) {
            float theta_i = std::pow(base,
                                     -static_cast<float>(2 * i) /
                                         static_cast<float>(d_h));
            float angle = static_cast<float>(t) * theta_i;
            cos_h[t * half + i] = std::cos(angle);
            sin_h[t * half + i] = std::sin(angle);
        }
    }

    Tensor cos_d({T_max, half}, DType::FP32, device);
    Tensor sin_d({T_max, half}, DType::FP32, device);
    if (device == Device::Host) {
        std::memcpy(cos_d.data(), cos_h.data(), cos_h.size() * sizeof(float));
        std::memcpy(sin_d.data(), sin_h.data(), sin_h.size() * sizeof(float));
    } else {
        MLLM_CUDA_CHECK(cudaMemcpy(cos_d.data(), cos_h.data(),
                                    cos_h.size() * sizeof(float),
                                    cudaMemcpyHostToDevice));
        MLLM_CUDA_CHECK(cudaMemcpy(sin_d.data(), sin_h.data(),
                                    sin_h.size() * sizeof(float),
                                    cudaMemcpyHostToDevice));
    }
    return {std::move(cos_d), std::move(sin_d)};
}

void rope_apply_inplace(Tensor& x, const Tensor& cos, const Tensor& sin,
                         int N, int T, int d_h) {
    if (d_h % 2 != 0) throw std::invalid_argument("rope: d_h must be even");
    if (x.numel() != static_cast<std::int64_t>(N) * T * d_h)
        throw std::invalid_argument("rope: x numel mismatch");
    int half = d_h / 2;
    long long total = static_cast<long long>(N) * T * half;
    if (total == 0) return;
    const int block = 256;
    unsigned grid = static_cast<unsigned>((total + block - 1) / block);
    if (x.dtype() == DType::FP32) {
        rope_inplace_fwd_kernel<<<grid, block>>>(
            x.data_as<float>(), cos.data_as<float>(), sin.data_as<float>(),
            N, T, half);
    } else if (x.dtype() == DType::BF16) {
        rope_inplace_fwd_bf16_kernel<<<grid, block>>>(
            static_cast<unsigned short*>(x.data()),
            cos.data_as<float>(), sin.data_as<float>(),
            N, T, half);
    } else {
        throw std::invalid_argument("rope_fwd: dtype must be FP32 or BF16");
    }
    MLLM_CUDA_CHECK(cudaGetLastError());
}

void rope_apply_backward_inplace(Tensor& dx, const Tensor& cos,
                                   const Tensor& sin,
                                   int N, int T, int d_h) {
    if (d_h % 2 != 0) throw std::invalid_argument("rope: d_h must be even");
    int half = d_h / 2;
    long long total = static_cast<long long>(N) * T * half;
    if (total == 0) return;
    const int block = 256;
    unsigned grid = static_cast<unsigned>((total + block - 1) / block);
    if (dx.dtype() == DType::FP32) {
        rope_inplace_bwd_kernel<<<grid, block>>>(
            dx.data_as<float>(), cos.data_as<float>(), sin.data_as<float>(),
            N, T, half);
    } else if (dx.dtype() == DType::BF16) {
        rope_inplace_bwd_bf16_kernel<<<grid, block>>>(
            static_cast<unsigned short*>(dx.data()),
            cos.data_as<float>(), sin.data_as<float>(),
            N, T, half);
    } else {
        throw std::invalid_argument("rope_bwd: dtype must be FP32 or BF16");
    }
    MLLM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace modernllm
