#include "model/activations.h"

#include <cmath>
#include <stdexcept>

#include "core/cuda_check.h"

namespace modernllm {

namespace {

// Local BF16 <-> FP32 device helpers (kept here so this TU is self-contained).
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

constexpr float kGeluC = 0.7978845608028654f;          // sqrt(2 / pi)
constexpr float kGeluA = 0.044715f;

__device__ __forceinline__ float gelu_value(float x) {
    float u = kGeluC * (x + kGeluA * x * x * x);
    return 0.5f * x * (1.f + tanhf(u));
}

__global__ void gelu_forward_kernel(const float* __restrict__ x,
                                    float* __restrict__ y, std::int64_t n) {
    std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
    if (i < n) y[i] = gelu_value(x[i]);
}

__global__ void gelu_backward_kernel(const float* __restrict__ x,
                                     const float* __restrict__ d_y,
                                     float* __restrict__ d_x,
                                     std::int64_t n) {
    std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
    if (i >= n) return;
    float xi = x[i];
    float u = kGeluC * (xi + kGeluA * xi * xi * xi);
    float t = tanhf(u);
    float du_dx = kGeluC * (1.f + 3.f * kGeluA * xi * xi);
    // dy/dx = 0.5 * (1 + tanh(u)) + 0.5 * x * (1 - tanh^2(u)) * du/dx
    float dy_dx = 0.5f * (1.f + t) + 0.5f * xi * (1.f - t * t) * du_dx;
    d_x[i] = d_y[i] * dy_dx;
}

__global__ void add_inplace_kernel(float* __restrict__ a,
                                   const float* __restrict__ b,
                                   std::int64_t n) {
    std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
    if (i < n) a[i] += b[i];
}

__global__ void add_inplace_bf16_kernel(unsigned short* __restrict__ a,
                                         const unsigned short* __restrict__ b,
                                         std::int64_t n) {
    std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
    if (i >= n) return;
    float fa = bf16_to_f32_d(a[i]);
    float fb = bf16_to_f32_d(b[i]);
    a[i] = f32_to_bf16_rne_d(fa + fb);
}

}  // namespace

void gelu_forward(const Tensor& x, Tensor& y) {
    if (x.numel() != y.numel())
        throw std::invalid_argument("gelu_forward: numel mismatch");
    if (x.dtype() != DType::FP32 || y.dtype() != DType::FP32)
        throw std::invalid_argument("gelu_forward: only FP32");
    if (x.device() != Device::Cuda || y.device() != Device::Cuda)
        throw std::invalid_argument("gelu_forward: must be CUDA");

    std::int64_t n = x.numel();
    if (n == 0) return;
    const int block = 256;
    unsigned grid = static_cast<unsigned>((n + block - 1) / block);
    gelu_forward_kernel<<<grid, block>>>(x.data_as<float>(), y.data_as<float>(), n);
    MLLM_CUDA_CHECK(cudaGetLastError());
}

void gelu_backward(const Tensor& x, const Tensor& d_y, Tensor& d_x) {
    if (x.numel() != d_y.numel() || x.numel() != d_x.numel())
        throw std::invalid_argument("gelu_backward: numel mismatch");

    std::int64_t n = x.numel();
    if (n == 0) return;
    const int block = 256;
    unsigned grid = static_cast<unsigned>((n + block - 1) / block);
    gelu_backward_kernel<<<grid, block>>>(x.data_as<float>(),
                                          d_y.data_as<float>(),
                                          d_x.data_as<float>(), n);
    MLLM_CUDA_CHECK(cudaGetLastError());
}

void add_inplace(Tensor& a, const Tensor& b) {
    if (a.numel() != b.numel())
        throw std::invalid_argument("add_inplace: numel mismatch");
    if (a.dtype() != b.dtype())
        throw std::invalid_argument("add_inplace: dtype mismatch");
    std::int64_t n = a.numel();
    if (n == 0) return;
    const int block = 256;
    unsigned grid = static_cast<unsigned>((n + block - 1) / block);
    if (a.dtype() == DType::FP32) {
        add_inplace_kernel<<<grid, block>>>(a.data_as<float>(),
                                              b.data_as<float>(), n);
    } else if (a.dtype() == DType::BF16) {
        add_inplace_bf16_kernel<<<grid, block>>>(
            static_cast<unsigned short*>(a.data()),
            static_cast<const unsigned short*>(b.data()), n);
    } else {
        throw std::invalid_argument("add_inplace: dtype must be FP32 or BF16");
    }
    MLLM_CUDA_CHECK(cudaGetLastError());
}

namespace {

__global__ void silu_mul_fwd_kernel(const float* __restrict__ gate,
                                    const float* __restrict__ up,
                                    float* __restrict__ out,
                                    std::int64_t n) {
    std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float sig = 1.f / (1.f + expf(-g));
    out[i] = g * sig * up[i];
}

__global__ void silu_mul_bwd_kernel(const float* __restrict__ gate,
                                    const float* __restrict__ up,
                                    const float* __restrict__ d_out,
                                    float* __restrict__ d_gate,
                                    float* __restrict__ d_up,
                                    std::int64_t n) {
    std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    float u = up[i];
    float dy = d_out[i];
    float sig = 1.f / (1.f + expf(-g));
    float silu = g * sig;
    // d_silu/dg = sig * (1 + g*(1 - sig))
    float dsilu_dg = sig * (1.f + g * (1.f - sig));
    d_up[i] = dy * silu;
    d_gate[i] = dy * u * dsilu_dg;
}

__global__ void silu_mul_fwd_bf16_kernel(const unsigned short* __restrict__ gate,
                                          const unsigned short* __restrict__ up,
                                          unsigned short* __restrict__ out,
                                          std::int64_t n) {
    std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
    if (i >= n) return;
    float g = bf16_to_f32_d(gate[i]);
    float u = bf16_to_f32_d(up[i]);
    float sig = 1.f / (1.f + expf(-g));
    out[i] = f32_to_bf16_rne_d(g * sig * u);
}

__global__ void silu_mul_bwd_bf16_kernel(const unsigned short* __restrict__ gate,
                                          const unsigned short* __restrict__ up,
                                          const unsigned short* __restrict__ d_out,
                                          unsigned short* __restrict__ d_gate,
                                          unsigned short* __restrict__ d_up,
                                          std::int64_t n) {
    std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
    if (i >= n) return;
    float g = bf16_to_f32_d(gate[i]);
    float u = bf16_to_f32_d(up[i]);
    float dy = bf16_to_f32_d(d_out[i]);
    float sig = 1.f / (1.f + expf(-g));
    float silu = g * sig;
    float dsilu_dg = sig * (1.f + g * (1.f - sig));
    d_up[i] = f32_to_bf16_rne_d(dy * silu);
    d_gate[i] = f32_to_bf16_rne_d(dy * u * dsilu_dg);
}

}  // namespace

void silu_mul_forward(const Tensor& gate, const Tensor& up, Tensor& out) {
    if (gate.numel() != up.numel() || gate.numel() != out.numel())
        throw std::invalid_argument("silu_mul_fwd: numel mismatch");
    if (gate.dtype() != up.dtype() || gate.dtype() != out.dtype())
        throw std::invalid_argument("silu_mul_fwd: dtype mismatch");
    std::int64_t n = gate.numel();
    if (n == 0) return;
    const int block = 256;
    unsigned grid = static_cast<unsigned>((n + block - 1) / block);
    if (gate.dtype() == DType::FP32) {
        silu_mul_fwd_kernel<<<grid, block>>>(gate.data_as<float>(),
                                              up.data_as<float>(),
                                              out.data_as<float>(), n);
    } else if (gate.dtype() == DType::BF16) {
        silu_mul_fwd_bf16_kernel<<<grid, block>>>(
            static_cast<const unsigned short*>(gate.data()),
            static_cast<const unsigned short*>(up.data()),
            static_cast<unsigned short*>(out.data()), n);
    } else {
        throw std::invalid_argument("silu_mul_fwd: dtype must be FP32 or BF16");
    }
    MLLM_CUDA_CHECK(cudaGetLastError());
}

void silu_mul_backward(const Tensor& gate, const Tensor& up,
                        const Tensor& d_out,
                        Tensor& d_gate, Tensor& d_up) {
    std::int64_t n = gate.numel();
    if (up.numel() != n || d_out.numel() != n ||
        d_gate.numel() != n || d_up.numel() != n)
        throw std::invalid_argument("silu_mul_bwd: numel mismatch");
    if (gate.dtype() != up.dtype() || gate.dtype() != d_out.dtype() ||
        gate.dtype() != d_gate.dtype() || gate.dtype() != d_up.dtype())
        throw std::invalid_argument("silu_mul_bwd: dtype mismatch");
    if (n == 0) return;
    const int block = 256;
    unsigned grid = static_cast<unsigned>((n + block - 1) / block);
    if (gate.dtype() == DType::FP32) {
        silu_mul_bwd_kernel<<<grid, block>>>(gate.data_as<float>(),
                                              up.data_as<float>(),
                                              d_out.data_as<float>(),
                                              d_gate.data_as<float>(),
                                              d_up.data_as<float>(), n);
    } else if (gate.dtype() == DType::BF16) {
        silu_mul_bwd_bf16_kernel<<<grid, block>>>(
            static_cast<const unsigned short*>(gate.data()),
            static_cast<const unsigned short*>(up.data()),
            static_cast<const unsigned short*>(d_out.data()),
            static_cast<unsigned short*>(d_gate.data()),
            static_cast<unsigned short*>(d_up.data()), n);
    } else {
        throw std::invalid_argument("silu_mul_bwd: dtype must be FP32 or BF16");
    }
    MLLM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace modernllm
