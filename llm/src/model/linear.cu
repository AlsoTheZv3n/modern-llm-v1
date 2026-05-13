#include "model/linear.h"

#include <stdexcept>

#include "core/cast.h"
#include "core/cuda_check.h"

namespace modernllm {

namespace {

// Y[n, o] += b[o]
__global__ void add_bias_kernel(float* __restrict__ Y,
                                const float* __restrict__ b,
                                int N, int Out) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y;
    if (o >= Out || n >= N) return;
    Y[static_cast<long long>(n) * Out + o] += b[o];
}

// BF16 output variant: Y[n, o] (BF16) += b[o] (FP32). Read BF16, add as FP32,
// write back as BF16.
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

__global__ void add_bias_bf16_kernel(unsigned short* __restrict__ Y,
                                      const float* __restrict__ b,
                                      int N, int Out) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y;
    if (o >= Out || n >= N) return;
    long long idx = static_cast<long long>(n) * Out + o;
    float yf = bf16_to_f32_d(Y[idx]) + b[o];
    Y[idx] = f32_to_bf16_rne_d(yf);
}

// BF16 dY → FP32 dB: row-sum of BF16 dY along N, accumulated into FP32 dB.
// blockDim must be a power of two.
__global__ void bias_grad_bf16_kernel(const unsigned short* __restrict__ d_Y,
                                       float* __restrict__ d_b,
                                       int N, int Out) {
    int o = blockIdx.x;
    if (o >= Out) return;
    extern __shared__ float smem[];
    float local_sum = 0.f;
    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        local_sum += bf16_to_f32_d(d_Y[static_cast<long long>(n) * Out + o]);
    }
    smem[threadIdx.x] = local_sum;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (threadIdx.x < off) smem[threadIdx.x] += smem[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(&d_b[o], smem[0]);
}

// d_b[o] += sum_n d_Y[n, o]
// One block per output feature, threads stride over N. blockDim must be a
// power of two for the reduction.
__global__ void bias_grad_kernel(const float* __restrict__ d_Y,
                                 float* __restrict__ d_b,
                                 int N, int Out) {
    int o = blockIdx.x;
    if (o >= Out) return;
    extern __shared__ float smem[];
    float local_sum = 0.f;
    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        local_sum += d_Y[static_cast<long long>(n) * Out + o];
    }
    smem[threadIdx.x] = local_sum;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (threadIdx.x < off) smem[threadIdx.x] += smem[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(&d_b[o], smem[0]);
    }
}

void check_2d_fp32_cuda(const Tensor& t, const char* name) {
    if (t.ndim() != 2 || t.dtype() != DType::FP32 ||
        t.device() != Device::Cuda) {
        throw std::invalid_argument(std::string("linear: bad ") + name +
                                     " (need 2D FP32 CUDA)");
    }
}

}  // namespace

void linear_forward(cublasHandle_t h,
                     const Tensor& X, const Tensor& W,
                     const Tensor* bias, Tensor& Y) {
    check_2d_fp32_cuda(X, "X");
    check_2d_fp32_cuda(W, "W");
    check_2d_fp32_cuda(Y, "Y");

    int N = static_cast<int>(X.shape()[0]);
    int In = static_cast<int>(X.shape()[1]);
    int Out = static_cast<int>(W.shape()[1]);
    if (W.shape()[0] != In || Y.shape()[0] != N || Y.shape()[1] != Out) {
        throw std::invalid_argument("linear_forward: shape mismatch");
    }

    // Y = X @ W
    gemm_fp32_rowmajor(h, Y.data_as<float>(), X.data_as<float>(),
                        W.data_as<float>(), N, Out, In,
                        /*trans_a=*/false, /*trans_b=*/false);

    if (bias) {
        if (bias->ndim() != 1 || bias->shape()[0] != Out ||
            bias->dtype() != DType::FP32 || bias->device() != Device::Cuda) {
            throw std::invalid_argument("linear_forward: bad bias shape/dtype");
        }
        const int block_x = 128;
        dim3 block(block_x);
        dim3 grid((Out + block_x - 1) / block_x, N);
        add_bias_kernel<<<grid, block>>>(Y.data_as<float>(),
                                         bias->data_as<float>(), N, Out);
        MLLM_CUDA_CHECK(cudaGetLastError());
    }
}

void linear_backward(cublasHandle_t h,
                      const Tensor& X, const Tensor& W,
                      const Tensor& d_Y,
                      Tensor& d_X, Tensor& d_W,
                      Tensor* d_b) {
    check_2d_fp32_cuda(X, "X");
    check_2d_fp32_cuda(W, "W");
    check_2d_fp32_cuda(d_Y, "d_Y");
    check_2d_fp32_cuda(d_X, "d_X");
    check_2d_fp32_cuda(d_W, "d_W");

    int N = static_cast<int>(X.shape()[0]);
    int In = static_cast<int>(X.shape()[1]);
    int Out = static_cast<int>(W.shape()[1]);
    if (d_Y.shape()[0] != N || d_Y.shape()[1] != Out ||
        d_X.shape() != X.shape() || d_W.shape() != W.shape()) {
        throw std::invalid_argument("linear_backward: shape mismatch");
    }

    // dX = dY @ W^T   (W is [In, Out], so W^T is [Out, In] — pass trans_b)
    // result shape [N, In]
    gemm_fp32_rowmajor(h, d_X.data_as<float>(), d_Y.data_as<float>(),
                        W.data_as<float>(), N, In, Out,
                        /*trans_a=*/false, /*trans_b=*/true);

    // dW = X^T @ dY   (accumulate; X is [N, In] so X^T is [In, N])
    // result shape [In, Out], beta=1 to accumulate into existing d_W
    gemm_fp32_rowmajor(h, d_W.data_as<float>(), X.data_as<float>(),
                        d_Y.data_as<float>(), In, Out, N,
                        /*trans_a=*/true, /*trans_b=*/false,
                        /*alpha=*/1.0f, /*beta=*/1.0f);

    // db = sum_n dY[n, :]
    if (d_b) {
        if (d_b->ndim() != 1 || d_b->shape()[0] != Out ||
            d_b->dtype() != DType::FP32 || d_b->device() != Device::Cuda) {
            throw std::invalid_argument("linear_backward: bad d_b shape");
        }
        const int block = 256;
        bias_grad_kernel<<<static_cast<unsigned>(Out), block,
                            block * sizeof(float)>>>(
            d_Y.data_as<float>(), d_b->data_as<float>(), N, Out);
        MLLM_CUDA_CHECK(cudaGetLastError());
    }
}

// ---------------------------------------------------------------------------
// BF16 mixed-precision Linear
// ---------------------------------------------------------------------------

void linear_forward_bf16(cublasHandle_t h,
                          const Tensor& X, const Tensor& W,
                          const Tensor* bias, Tensor& Y) {
    check_2d_fp32_cuda(X, "X");
    check_2d_fp32_cuda(W, "W");
    check_2d_fp32_cuda(Y, "Y");

    int N = static_cast<int>(X.shape()[0]);
    int In = static_cast<int>(X.shape()[1]);
    int Out = static_cast<int>(W.shape()[1]);
    if (W.shape()[0] != In || Y.shape()[0] != N || Y.shape()[1] != Out) {
        throw std::invalid_argument("linear_forward_bf16: shape mismatch");
    }

    Tensor X_bf16({N, In}, DType::BF16, Device::Cuda);
    Tensor W_bf16({In, Out}, DType::BF16, Device::Cuda);
    cast_fp32_to_bf16(X, X_bf16);
    cast_fp32_to_bf16(W, W_bf16);

    gemm_bf16in_fp32out_rowmajor(
        h, Y.data_as<float>(),
        static_cast<const unsigned short*>(X_bf16.data()),
        static_cast<const unsigned short*>(W_bf16.data()),
        N, Out, In,
        /*trans_a=*/false, /*trans_b=*/false);

    if (bias) {
        if (bias->ndim() != 1 || bias->shape()[0] != Out ||
            bias->dtype() != DType::FP32 || bias->device() != Device::Cuda) {
            throw std::invalid_argument(
                "linear_forward_bf16: bad bias shape/dtype");
        }
        const int block_x = 128;
        dim3 block(block_x);
        dim3 grid((Out + block_x - 1) / block_x, N);
        add_bias_kernel<<<grid, block>>>(Y.data_as<float>(),
                                         bias->data_as<float>(), N, Out);
        MLLM_CUDA_CHECK(cudaGetLastError());
    }
}

void linear_backward_bf16(cublasHandle_t h,
                           const Tensor& X, const Tensor& W,
                           const Tensor& d_Y,
                           Tensor& d_X, Tensor& d_W,
                           Tensor* d_b) {
    check_2d_fp32_cuda(X, "X");
    check_2d_fp32_cuda(W, "W");
    check_2d_fp32_cuda(d_Y, "d_Y");
    check_2d_fp32_cuda(d_X, "d_X");
    check_2d_fp32_cuda(d_W, "d_W");

    int N = static_cast<int>(X.shape()[0]);
    int In = static_cast<int>(X.shape()[1]);
    int Out = static_cast<int>(W.shape()[1]);
    if (d_Y.shape()[0] != N || d_Y.shape()[1] != Out ||
        d_X.shape() != X.shape() || d_W.shape() != W.shape()) {
        throw std::invalid_argument("linear_backward_bf16: shape mismatch");
    }

    // We cast each FP32 source once and reuse for both BF16 GEMMs.
    Tensor X_bf16({N, In}, DType::BF16, Device::Cuda);
    Tensor W_bf16({In, Out}, DType::BF16, Device::Cuda);
    Tensor dY_bf16({N, Out}, DType::BF16, Device::Cuda);
    cast_fp32_to_bf16(X, X_bf16);
    cast_fp32_to_bf16(W, W_bf16);
    cast_fp32_to_bf16(d_Y, dY_bf16);

    // dX = dY @ W^T  (BF16 inputs, FP32 output)
    gemm_bf16in_fp32out_rowmajor(
        h, d_X.data_as<float>(),
        static_cast<const unsigned short*>(dY_bf16.data()),
        static_cast<const unsigned short*>(W_bf16.data()),
        N, In, Out,
        /*trans_a=*/false, /*trans_b=*/true);

    // dW = X^T @ dY  (accumulate into FP32 d_W)
    gemm_bf16in_fp32out_rowmajor(
        h, d_W.data_as<float>(),
        static_cast<const unsigned short*>(X_bf16.data()),
        static_cast<const unsigned short*>(dY_bf16.data()),
        In, Out, N,
        /*trans_a=*/true, /*trans_b=*/false,
        /*alpha=*/1.0f, /*beta=*/1.0f);

    // db: row-sum of dY → small reduction, stay in FP32.
    if (d_b) {
        if (d_b->ndim() != 1 || d_b->shape()[0] != Out ||
            d_b->dtype() != DType::FP32 || d_b->device() != Device::Cuda) {
            throw std::invalid_argument(
                "linear_backward_bf16: bad d_b shape");
        }
        const int block = 256;
        bias_grad_kernel<<<static_cast<unsigned>(Out), block,
                            block * sizeof(float)>>>(
            d_Y.data_as<float>(), d_b->data_as<float>(), N, Out);
        MLLM_CUDA_CHECK(cudaGetLastError());
    }
}

// ---------------------------------------------------------------------------
// Stage T1 — fast BF16 path: persistent BF16 weight mirror + scratch arena.
// ---------------------------------------------------------------------------

namespace {

void check_bf16_cuda(const Tensor& t, const char* name) {
    if (t.dtype() != DType::BF16 || t.device() != Device::Cuda) {
        throw std::invalid_argument(std::string("linear_arena: bad ") + name +
                                     " (need BF16 CUDA)");
    }
}

}  // namespace

void linear_forward_bf16_arena(cublasHandle_t h,
                                const Tensor& X_fp32,
                                const Tensor& W_bf16,
                                const Tensor* bias_fp32,
                                Tensor& Y_fp32,
                                ScratchArena& arena) {
    check_2d_fp32_cuda(X_fp32, "X");
    check_2d_fp32_cuda(Y_fp32, "Y");
    check_bf16_cuda(W_bf16, "W_bf16");

    int N = static_cast<int>(X_fp32.shape()[0]);
    int In = static_cast<int>(X_fp32.shape()[1]);
    int Out = static_cast<int>(W_bf16.shape()[1]);
    if (W_bf16.shape()[0] != In || Y_fp32.shape()[0] != N ||
        Y_fp32.shape()[1] != Out) {
        throw std::invalid_argument(
            "linear_forward_bf16_arena: shape mismatch");
    }

    // Cast input X (only); W is already BF16.
    Tensor X_bf16 = arena.allocate({N, In}, DType::BF16);
    cast_fp32_to_bf16(X_fp32, X_bf16);

    gemm_bf16in_fp32out_rowmajor(
        h, Y_fp32.data_as<float>(),
        static_cast<const unsigned short*>(X_bf16.data()),
        static_cast<const unsigned short*>(W_bf16.data()),
        N, Out, In, /*trans_a=*/false, /*trans_b=*/false);

    if (bias_fp32) {
        if (bias_fp32->ndim() != 1 || bias_fp32->shape()[0] != Out ||
            bias_fp32->dtype() != DType::FP32 ||
            bias_fp32->device() != Device::Cuda) {
            throw std::invalid_argument(
                "linear_forward_bf16_arena: bad bias");
        }
        const int block_x = 128;
        dim3 block(block_x);
        dim3 grid((Out + block_x - 1) / block_x, N);
        add_bias_kernel<<<grid, block>>>(
            Y_fp32.data_as<float>(), bias_fp32->data_as<float>(), N, Out);
        MLLM_CUDA_CHECK(cudaGetLastError());
    }
}

void linear_backward_bf16_arena(cublasHandle_t h,
                                 const Tensor& X_fp32,
                                 const Tensor& W_bf16,
                                 const Tensor& d_Y_fp32,
                                 Tensor& d_X_fp32,
                                 Tensor& d_W_fp32,
                                 Tensor* d_b_fp32,
                                 ScratchArena& arena) {
    check_2d_fp32_cuda(X_fp32, "X");
    check_bf16_cuda(W_bf16, "W_bf16");
    check_2d_fp32_cuda(d_Y_fp32, "d_Y");
    check_2d_fp32_cuda(d_X_fp32, "d_X");
    check_2d_fp32_cuda(d_W_fp32, "d_W");

    int N = static_cast<int>(X_fp32.shape()[0]);
    int In = static_cast<int>(X_fp32.shape()[1]);
    int Out = static_cast<int>(W_bf16.shape()[1]);
    if (d_Y_fp32.shape()[0] != N || d_Y_fp32.shape()[1] != Out ||
        d_X_fp32.shape() != X_fp32.shape() ||
        d_W_fp32.shape() != W_bf16.shape()) {
        throw std::invalid_argument(
            "linear_backward_bf16_arena: shape mismatch");
    }

    Tensor X_bf16 = arena.allocate({N, In}, DType::BF16);
    Tensor dY_bf16 = arena.allocate({N, Out}, DType::BF16);
    cast_fp32_to_bf16(X_fp32, X_bf16);
    cast_fp32_to_bf16(d_Y_fp32, dY_bf16);

    // dX = dY @ W^T (BF16 inputs, FP32 output)
    gemm_bf16in_fp32out_rowmajor(
        h, d_X_fp32.data_as<float>(),
        static_cast<const unsigned short*>(dY_bf16.data()),
        static_cast<const unsigned short*>(W_bf16.data()),
        N, In, Out, /*trans_a=*/false, /*trans_b=*/true);

    // dW = X^T @ dY (accumulate)
    gemm_bf16in_fp32out_rowmajor(
        h, d_W_fp32.data_as<float>(),
        static_cast<const unsigned short*>(X_bf16.data()),
        static_cast<const unsigned short*>(dY_bf16.data()),
        In, Out, N, /*trans_a=*/true, /*trans_b=*/false,
        /*alpha=*/1.0f, /*beta=*/1.0f);

    if (d_b_fp32) {
        if (d_b_fp32->ndim() != 1 || d_b_fp32->shape()[0] != Out ||
            d_b_fp32->dtype() != DType::FP32 ||
            d_b_fp32->device() != Device::Cuda) {
            throw std::invalid_argument(
                "linear_backward_bf16_arena: bad d_b shape");
        }
        const int block = 256;
        bias_grad_kernel<<<static_cast<unsigned>(Out), block,
                            block * sizeof(float)>>>(
            d_Y_fp32.data_as<float>(), d_b_fp32->data_as<float>(), N, Out);
        MLLM_CUDA_CHECK(cudaGetLastError());
    }
}

// ---------------------------------------------------------------------------
// T9.B — full BF16 path: BF16 X, BF16 W (mirror), BF16 Y. No scratch needed.
// ---------------------------------------------------------------------------

void linear_forward_bf16inout(cublasHandle_t h,
                               const Tensor& X_bf16,
                               const Tensor& W_bf16,
                               const Tensor* bias_fp32,
                               Tensor& Y_bf16) {
    if (X_bf16.ndim() != 2 || W_bf16.ndim() != 2 || Y_bf16.ndim() != 2)
        throw std::invalid_argument("linear_bf16io_fwd: tensors must be 2D");
    if (X_bf16.dtype() != DType::BF16 || W_bf16.dtype() != DType::BF16 ||
        Y_bf16.dtype() != DType::BF16) {
        throw std::invalid_argument("linear_bf16io_fwd: dtypes must be BF16");
    }
    int N = static_cast<int>(X_bf16.shape()[0]);
    int In = static_cast<int>(X_bf16.shape()[1]);
    int Out = static_cast<int>(W_bf16.shape()[1]);
    if (W_bf16.shape()[0] != In || Y_bf16.shape()[0] != N ||
        Y_bf16.shape()[1] != Out) {
        throw std::invalid_argument("linear_bf16io_fwd: shape mismatch");
    }

    gemm_bf16inout_rowmajor(
        h, static_cast<unsigned short*>(Y_bf16.data()),
        static_cast<const unsigned short*>(X_bf16.data()),
        static_cast<const unsigned short*>(W_bf16.data()),
        N, Out, In);

    if (bias_fp32) {
        if (bias_fp32->ndim() != 1 || bias_fp32->shape()[0] != Out ||
            bias_fp32->dtype() != DType::FP32) {
            throw std::invalid_argument("linear_bf16io_fwd: bad bias");
        }
        const int block_x = 128;
        dim3 block(block_x);
        dim3 grid((Out + block_x - 1) / block_x, N);
        add_bias_bf16_kernel<<<grid, block>>>(
            static_cast<unsigned short*>(Y_bf16.data()),
            bias_fp32->data_as<float>(), N, Out);
        MLLM_CUDA_CHECK(cudaGetLastError());
    }
}

void linear_backward_bf16inout(cublasHandle_t h,
                                const Tensor& X_bf16,
                                const Tensor& W_bf16,
                                const Tensor& d_Y_bf16,
                                Tensor& d_X_bf16,
                                Tensor& d_W_fp32,
                                Tensor* d_b_fp32) {
    if (X_bf16.dtype() != DType::BF16 || W_bf16.dtype() != DType::BF16 ||
        d_Y_bf16.dtype() != DType::BF16 || d_X_bf16.dtype() != DType::BF16) {
        throw std::invalid_argument("linear_bf16io_bwd: X/W/dY/dX must be BF16");
    }
    if (d_W_fp32.dtype() != DType::FP32) {
        throw std::invalid_argument(
            "linear_bf16io_bwd: d_W must be FP32 (master grad)");
    }
    int N = static_cast<int>(X_bf16.shape()[0]);
    int In = static_cast<int>(X_bf16.shape()[1]);
    int Out = static_cast<int>(W_bf16.shape()[1]);
    if (d_Y_bf16.shape()[0] != N || d_Y_bf16.shape()[1] != Out ||
        d_X_bf16.shape() != X_bf16.shape() ||
        d_W_fp32.shape() != W_bf16.shape()) {
        throw std::invalid_argument("linear_bf16io_bwd: shape mismatch");
    }

    // dX = dY @ W^T (BF16 in/out)
    gemm_bf16inout_rowmajor(
        h, static_cast<unsigned short*>(d_X_bf16.data()),
        static_cast<const unsigned short*>(d_Y_bf16.data()),
        static_cast<const unsigned short*>(W_bf16.data()),
        N, In, Out, /*trans_a=*/false, /*trans_b=*/true);

    // dW = X^T @ dY (BF16 in, FP32 out, accumulate)
    gemm_bf16in_fp32out_rowmajor(
        h, d_W_fp32.data_as<float>(),
        static_cast<const unsigned short*>(X_bf16.data()),
        static_cast<const unsigned short*>(d_Y_bf16.data()),
        In, Out, N, /*trans_a=*/true, /*trans_b=*/false,
        /*alpha=*/1.0f, /*beta=*/1.0f);

    if (d_b_fp32) {
        if (d_b_fp32->ndim() != 1 || d_b_fp32->shape()[0] != Out ||
            d_b_fp32->dtype() != DType::FP32) {
            throw std::invalid_argument("linear_bf16io_bwd: bad d_b");
        }
        const int block = 256;
        bias_grad_bf16_kernel<<<static_cast<unsigned>(Out), block,
                                  block * sizeof(float)>>>(
            static_cast<const unsigned short*>(d_Y_bf16.data()),
            d_b_fp32->data_as<float>(), N, Out);
        MLLM_CUDA_CHECK(cudaGetLastError());
    }
}

}  // namespace modernllm
