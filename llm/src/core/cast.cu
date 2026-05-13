#include "core/cast.h"

#include <cstdint>
#include <stdexcept>

#include "core/cuda_check.h"

namespace modernllm {

namespace {

// Round-to-nearest-even FP32 -> BF16. Same algorithm as the host
// f32_to_bf16 helper but inlined for the device kernel.
__device__ __forceinline__ unsigned short f32_to_bf16_rne(float f) {
    unsigned int bits = __float_as_uint(f);
    unsigned int lsb = (bits >> 16) & 1u;
    unsigned int rounding_bias = 0x7FFFu + lsb;
    bits += rounding_bias;
    return static_cast<unsigned short>(bits >> 16);
}

__device__ __forceinline__ float bf16_to_f32_dev(unsigned short b) {
    unsigned int bits = static_cast<unsigned int>(b) << 16;
    return __uint_as_float(bits);
}

__global__ void cast_fp32_to_bf16_kernel(const float* __restrict__ src,
                                         unsigned short* __restrict__ dst,
                                         std::int64_t n) {
    std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
    if (i < n) dst[i] = f32_to_bf16_rne(src[i]);
}

__global__ void cast_bf16_to_fp32_kernel(const unsigned short* __restrict__ src,
                                         float* __restrict__ dst,
                                         std::int64_t n) {
    std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
    if (i < n) dst[i] = bf16_to_f32_dev(src[i]);
}

void check_same_layout(const Tensor& a, const Tensor& b) {
    if (a.numel() != b.numel())
        throw std::invalid_argument("cast: numel mismatch");
    if (a.device() != Device::Cuda || b.device() != Device::Cuda)
        throw std::invalid_argument("cast: tensors must be CUDA");
}

}  // namespace

void cast_fp32_to_bf16(const Tensor& src, Tensor& dst) {
    check_same_layout(src, dst);
    if (src.dtype() != DType::FP32 || dst.dtype() != DType::BF16)
        throw std::invalid_argument("cast_fp32_to_bf16: bad dtypes");
    std::int64_t n = src.numel();
    if (n == 0) return;
    const int block = 256;
    unsigned grid = static_cast<unsigned>((n + block - 1) / block);
    cast_fp32_to_bf16_kernel<<<grid, block>>>(
        src.data_as<float>(),
        static_cast<unsigned short*>(dst.data()),
        n);
    MLLM_CUDA_CHECK(cudaGetLastError());
}

void cast_bf16_to_fp32(const Tensor& src, Tensor& dst) {
    check_same_layout(src, dst);
    if (src.dtype() != DType::BF16 || dst.dtype() != DType::FP32)
        throw std::invalid_argument("cast_bf16_to_fp32: bad dtypes");
    std::int64_t n = src.numel();
    if (n == 0) return;
    const int block = 256;
    unsigned grid = static_cast<unsigned>((n + block - 1) / block);
    cast_bf16_to_fp32_kernel<<<grid, block>>>(
        static_cast<const unsigned short*>(src.data()),
        dst.data_as<float>(),
        n);
    MLLM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace modernllm
