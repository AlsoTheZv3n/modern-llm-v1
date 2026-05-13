#include "core/gemm.h"

#include "core/cuda_check.h"

namespace modernllm {

CublasHandle::CublasHandle() {
    MLLM_CUBLAS_CHECK(cublasCreate(&handle_));
}

CublasHandle::~CublasHandle() {
    if (handle_) cublasDestroy(handle_);
}

void CublasHandle::set_stream(cudaStream_t stream) {
    MLLM_CUBLAS_CHECK(cublasSetStream(handle_, stream));
}

void gemm_fp32_rowmajor(cublasHandle_t handle,
                        float* C, const float* A, const float* B,
                        int M, int N, int K,
                        bool trans_a, bool trans_b,
                        float alpha, float beta) {
    // We want row-major: C[M,N] = alpha * op(A) @ op(B) + beta * C
    //   trans_a=false: A stored as [M,K] row-major, lda_RM = K
    //   trans_a=true : A stored as [K,M] row-major, lda_RM = M
    //   trans_b=false: B stored as [K,N] row-major, ldb_RM = N
    //   trans_b=true : B stored as [N,K] row-major, ldb_RM = K
    //
    // In column-major we compute C^T = op(B)^T @ op(A)^T:
    //   M_cb=N, N_cb=M, K_cb=K
    //   cuBLAS-A operand = B_RM, transposed iff trans_b is true
    //   cuBLAS-B operand = A_RM, transposed iff trans_a is true
    //   leading dims = the row-major leading dims of the storage
    cublasOperation_t op_a_cb = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t op_b_cb = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    int lda_cb = trans_b ? K : N;  // row-major leading dim of B as stored
    int ldb_cb = trans_a ? M : K;  // row-major leading dim of A as stored
    int ldc_cb = N;                // C is always [M,N] row-major

    MLLM_CUBLAS_CHECK(cublasSgemm(handle,
                                  op_a_cb, op_b_cb,
                                  /*M=*/N, /*N=*/M, /*K=*/K,
                                  &alpha,
                                  /*A=*/B, /*lda=*/lda_cb,
                                  /*B=*/A, /*ldb=*/ldb_cb,
                                  &beta,
                                  /*C=*/C, /*ldc=*/ldc_cb));
}

void gemm_bf16in_fp32out_rowmajor(cublasHandle_t handle,
                                    float* C,
                                    const unsigned short* A,
                                    const unsigned short* B,
                                    int M, int N, int K,
                                    bool trans_a, bool trans_b,
                                    float alpha, float beta) {
    // Same row-major-via-column-major trick as the FP32 GEMM. Inputs are
    // BF16 (encoded as uint16_t); accumulator + output stay FP32.
    cublasOperation_t op_a_cb = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t op_b_cb = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    int lda_cb = trans_b ? K : N;
    int ldb_cb = trans_a ? M : K;
    int ldc_cb = N;

    MLLM_CUBLAS_CHECK(cublasGemmEx(handle,
                                    op_a_cb, op_b_cb,
                                    /*M=*/N, /*N=*/M, /*K=*/K,
                                    &alpha,
                                    /*A=*/B, CUDA_R_16BF, lda_cb,
                                    /*B=*/A, CUDA_R_16BF, ldb_cb,
                                    &beta,
                                    /*C=*/C, CUDA_R_32F, ldc_cb,
                                    CUBLAS_COMPUTE_32F,
                                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void gemm_bf16inout_rowmajor(cublasHandle_t handle,
                              unsigned short* C,
                              const unsigned short* A,
                              const unsigned short* B,
                              int M, int N, int K,
                              bool trans_a, bool trans_b,
                              float alpha, float beta) {
    cublasOperation_t op_a_cb = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t op_b_cb = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    int lda_cb = trans_b ? K : N;
    int ldb_cb = trans_a ? M : K;
    int ldc_cb = N;

    MLLM_CUBLAS_CHECK(cublasGemmEx(handle,
                                    op_a_cb, op_b_cb,
                                    /*M=*/N, /*N=*/M, /*K=*/K,
                                    &alpha,
                                    /*A=*/B, CUDA_R_16BF, lda_cb,
                                    /*B=*/A, CUDA_R_16BF, ldb_cb,
                                    &beta,
                                    /*C=*/C, CUDA_R_16BF, ldc_cb,
                                    CUBLAS_COMPUTE_32F,
                                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

}  // namespace modernllm
