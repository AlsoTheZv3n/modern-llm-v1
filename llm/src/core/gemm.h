#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace modernllm {

// RAII handle for cuBLAS, with optional stream binding.
class CublasHandle {
   public:
    CublasHandle();
    ~CublasHandle();

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    void set_stream(cudaStream_t stream);
    cublasHandle_t get() const noexcept { return handle_; }
    operator cublasHandle_t() const noexcept { return handle_; }

   private:
    cublasHandle_t handle_{nullptr};
};

// Row-major FP32 GEMM: C [M,N] = alpha * op(A) @ op(B) + beta * C
//
// op(A) has shape [M, K], op(B) has shape [K, N]. As-stored shapes:
//   trans_a = false: A is [M, K] row-major
//   trans_a = true : A is [K, M] row-major (transposed before multiply)
//   trans_b = false: B is [K, N] row-major
//   trans_b = true : B is [N, K] row-major
//
// All pointers must point to device memory.
//
// Implementation: cuBLAS is column-major; we compute C^T = op(B)^T @ op(A)^T
// in column-major. See gemm.cu for the index/leading-dim derivation.
void gemm_fp32_rowmajor(cublasHandle_t handle,
                        float* C, const float* A, const float* B,
                        int M, int N, int K,
                        bool trans_a = false, bool trans_b = false,
                        float alpha = 1.0f, float beta = 0.0f);

// BF16-input / FP32-accumulator / FP32-output GEMM via cublasGemmEx
// (Tensor-Core path on Ampere+).
//
// A and B are stored as 2-byte BF16 values (caller passes raw `unsigned short*`
// pointers to device memory). Shape semantics — row-major, transpose flags,
// alpha/beta — match `gemm_fp32_rowmajor`.
//
// On Ampere this picks up the 312 TFLOPS BF16 tensor-core path versus the
// 156 TFLOPS TF32 path that cublasSgemm uses by default.
void gemm_bf16in_fp32out_rowmajor(cublasHandle_t handle,
                                    float* C,
                                    const unsigned short* A,
                                    const unsigned short* B,
                                    int M, int N, int K,
                                    bool trans_a = false, bool trans_b = false,
                                    float alpha = 1.0f, float beta = 0.0f);

// BF16 in / BF16 out variant for the full-BF16 activation path. The
// accumulator is still FP32 internally (CUBLAS_COMPUTE_32F).
void gemm_bf16inout_rowmajor(cublasHandle_t handle,
                              unsigned short* C,
                              const unsigned short* A,
                              const unsigned short* B,
                              int M, int N, int K,
                              bool trans_a = false, bool trans_b = false,
                              float alpha = 1.0f, float beta = 0.0f);

}  // namespace modernllm
