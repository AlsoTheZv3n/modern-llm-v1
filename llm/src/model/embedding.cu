#include "model/embedding.h"

#include <stdexcept>

#include "core/cuda_check.h"

namespace modernllm {

namespace {

__device__ __forceinline__ unsigned short f32_to_bf16_d(float f) {
    unsigned int bits = __float_as_uint(f);
    unsigned int lsb = (bits >> 16) & 1u;
    unsigned int rounding_bias = 0x7FFFu + lsb;
    bits += rounding_bias;
    return static_cast<unsigned short>(bits >> 16);
}
__device__ __forceinline__ float bf16_to_f32_d(unsigned short b) {
    unsigned int bits = static_cast<unsigned int>(b) << 16;
    return __uint_as_float(bits);
}

__global__ void embedding_forward_kernel(const float* __restrict__ weight,
                                         const int* __restrict__ ids,
                                         float* __restrict__ out,
                                         int N, int D) {
    int n = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D || n >= N) return;
    int id = ids[n];
    out[n * D + d] = weight[id * D + d];
}

// FP32 weight → BF16 output (cast on read).
__global__ void embedding_forward_bf16out_kernel(
        const float* __restrict__ weight,
        const int* __restrict__ ids,
        unsigned short* __restrict__ out,
        int N, int D) {
    int n = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D || n >= N) return;
    int id = ids[n];
    out[n * D + d] = f32_to_bf16_d(weight[id * D + d]);
}

__global__ void embedding_backward_kernel(const float* __restrict__ d_out,
                                          const int* __restrict__ ids,
                                          float* __restrict__ d_weight,
                                          int N, int D) {
    int n = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D || n >= N) return;
    int id = ids[n];
    atomicAdd(&d_weight[id * D + d], d_out[n * D + d]);
}

// BF16 d_out → FP32 d_weight (atomicAdd in FP32 with BF16-load cast).
__global__ void embedding_backward_bf16in_kernel(
        const unsigned short* __restrict__ d_out,
        const int* __restrict__ ids,
        float* __restrict__ d_weight,
        int N, int D) {
    int n = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D || n >= N) return;
    int id = ids[n];
    atomicAdd(&d_weight[id * D + d], bf16_to_f32_d(d_out[n * D + d]));
}

void check_shapes_forward(const Tensor& w, const Tensor& ids, const Tensor& out) {
    if (w.ndim() != 2) throw std::invalid_argument("embedding: weight must be [V, D]");
    if (ids.ndim() != 1) throw std::invalid_argument("embedding: ids must be [N]");
    if (out.ndim() != 2) throw std::invalid_argument("embedding: out must be [N, D]");
    if (out.shape()[0] != ids.shape()[0])
        throw std::invalid_argument("embedding: out[0] != ids[0]");
    if (out.shape()[1] != w.shape()[1])
        throw std::invalid_argument("embedding: out[1] != weight[1]");
    if (w.dtype() != DType::FP32 ||
        (out.dtype() != DType::FP32 && out.dtype() != DType::BF16) ||
        ids.dtype() != DType::INT32)
        throw std::invalid_argument("embedding: dtype mismatch");
    if (w.device() != Device::Cuda || out.device() != Device::Cuda ||
        ids.device() != Device::Cuda)
        throw std::invalid_argument("embedding: all tensors must be CUDA");
}

}  // namespace

void embedding_forward(const Tensor& weight, const Tensor& ids, Tensor& out) {
    check_shapes_forward(weight, ids, out);
    int N = static_cast<int>(ids.shape()[0]);
    int D = static_cast<int>(weight.shape()[1]);
    if (N == 0) return;

    const int block_x = 128;
    dim3 block(block_x);
    dim3 grid((D + block_x - 1) / block_x, N);
    if (out.dtype() == DType::FP32) {
        embedding_forward_kernel<<<grid, block>>>(
            weight.data_as<float>(),
            ids.data_as<int>(),
            out.data_as<float>(),
            N, D);
    } else {
        embedding_forward_bf16out_kernel<<<grid, block>>>(
            weight.data_as<float>(),
            ids.data_as<int>(),
            static_cast<unsigned short*>(out.data()),
            N, D);
    }
    MLLM_CUDA_CHECK(cudaGetLastError());
}

void embedding_backward(const Tensor& d_out, const Tensor& ids,
                         Tensor& d_weight) {
    if (d_out.ndim() != 2 || d_weight.ndim() != 2 || ids.ndim() != 1)
        throw std::invalid_argument("embedding_backward: bad shapes");
    if (d_out.shape()[0] != ids.shape()[0])
        throw std::invalid_argument("embedding_backward: N mismatch");
    if (d_out.shape()[1] != d_weight.shape()[1])
        throw std::invalid_argument("embedding_backward: D mismatch");
    if (d_weight.dtype() != DType::FP32 || ids.dtype() != DType::INT32)
        throw std::invalid_argument(
            "embedding_backward: d_weight must be FP32, ids INT32");
    if (d_out.dtype() != DType::FP32 && d_out.dtype() != DType::BF16)
        throw std::invalid_argument(
            "embedding_backward: d_out must be FP32 or BF16");

    int N = static_cast<int>(ids.shape()[0]);
    int D = static_cast<int>(d_weight.shape()[1]);
    if (N == 0) return;

    const int block_x = 128;
    dim3 block(block_x);
    dim3 grid((D + block_x - 1) / block_x, N);
    if (d_out.dtype() == DType::FP32) {
        embedding_backward_kernel<<<grid, block>>>(
            d_out.data_as<float>(),
            ids.data_as<int>(),
            d_weight.data_as<float>(),
            N, D);
    } else {
        embedding_backward_bf16in_kernel<<<grid, block>>>(
            static_cast<const unsigned short*>(d_out.data()),
            ids.data_as<int>(),
            d_weight.data_as<float>(),
            N, D);
    }
    MLLM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace modernllm
