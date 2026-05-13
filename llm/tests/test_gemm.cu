#include <cstdio>
#include <random>
#include <vector>

#include "core/gemm.h"
#include "core/tensor.h"
#include "tests/test_util.h"

using modernllm::CublasHandle;
using modernllm::Device;
using modernllm::DType;
using modernllm::Tensor;

namespace {

// Naive row-major matmul on host: C [M,N] = A [M,K] @ B [K,N]
void gemm_naive_host(float* C, const float* A, const float* B,
                     int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k) {
                acc += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = acc;
        }
    }
}

// Fill tensor with random values in [-1, 1]
void fill_uniform(float* p, int n, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < n; ++i) p[i] = dist(rng);
}

void run_case(int M, int N, int K, std::mt19937& rng) {
    std::printf("    case M=%d N=%d K=%d\n", M, N, K);

    Tensor A_h({M, K}, DType::FP32, Device::Host);
    Tensor B_h({K, N}, DType::FP32, Device::Host);
    Tensor C_ref({M, N}, DType::FP32, Device::Host);

    auto* a = A_h.data_as<float>();
    auto* b = B_h.data_as<float>();
    fill_uniform(a, M * K, rng);
    fill_uniform(b, K * N, rng);
    gemm_naive_host(C_ref.data_as<float>(), a, b, M, N, K);

    Tensor A_d = A_h.to(Device::Cuda);
    Tensor B_d = B_h.to(Device::Cuda);
    Tensor C_d({M, N}, DType::FP32, Device::Cuda);

    CublasHandle handle;
    modernllm::gemm_fp32_rowmajor(handle,
                                  C_d.data_as<float>(),
                                  A_d.data_as<float>(),
                                  B_d.data_as<float>(),
                                  M, N, K);

    Tensor C_h = C_d.to(Device::Host);
    auto* c = C_h.data_as<float>();
    auto* r = C_ref.data_as<float>();
    float tol = 1e-4f * static_cast<float>(K);
    float max_diff = 0.f;
    for (int i = 0; i < M * N; ++i) {
        max_diff = std::fmax(max_diff, std::fabs(c[i] - r[i]));
    }
    std::printf("      max_diff=%.3e tol=%.3e\n", max_diff, tol);
    MLLM_EXPECT(max_diff <= tol);
}

// Run with various transpose combinations: same logical C = A @ B,
// but A and/or B stored transposed in memory.
void run_case_transposed(int M, int N, int K, bool trans_a, bool trans_b,
                          std::mt19937& rng) {
    std::printf("    case M=%d N=%d K=%d trans_a=%d trans_b=%d\n",
                M, N, K, (int)trans_a, (int)trans_b);

    // Reference: produce A_logical [M,K] and B_logical [K,N], compute C_ref
    Tensor A_log_h({M, K}, DType::FP32, Device::Host);
    Tensor B_log_h({K, N}, DType::FP32, Device::Host);
    Tensor C_ref({M, N}, DType::FP32, Device::Host);
    fill_uniform(A_log_h.data_as<float>(), M * K, rng);
    fill_uniform(B_log_h.data_as<float>(), K * N, rng);
    gemm_naive_host(C_ref.data_as<float>(), A_log_h.data_as<float>(),
                    B_log_h.data_as<float>(), M, N, K);

    // Build the actually-stored tensors. If trans_a, store A transposed.
    auto transpose_host = [](const float* in, float* out, int rows, int cols) {
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                out[c * rows + r] = in[r * cols + c];
    };

    Tensor A_h, B_h;
    if (trans_a) {
        A_h = Tensor({K, M}, DType::FP32, Device::Host);
        transpose_host(A_log_h.data_as<float>(), A_h.data_as<float>(), M, K);
    } else {
        A_h = Tensor({M, K}, DType::FP32, Device::Host);
        std::memcpy(A_h.data(), A_log_h.data(),
                    static_cast<std::size_t>(M * K) * sizeof(float));
    }
    if (trans_b) {
        B_h = Tensor({N, K}, DType::FP32, Device::Host);
        transpose_host(B_log_h.data_as<float>(), B_h.data_as<float>(), K, N);
    } else {
        B_h = Tensor({K, N}, DType::FP32, Device::Host);
        std::memcpy(B_h.data(), B_log_h.data(),
                    static_cast<std::size_t>(K * N) * sizeof(float));
    }

    Tensor A_d = A_h.to(Device::Cuda);
    Tensor B_d = B_h.to(Device::Cuda);
    Tensor C_d({M, N}, DType::FP32, Device::Cuda);

    CublasHandle handle;
    modernllm::gemm_fp32_rowmajor(handle,
                                  C_d.data_as<float>(),
                                  A_d.data_as<float>(),
                                  B_d.data_as<float>(),
                                  M, N, K, trans_a, trans_b);

    Tensor C_h = C_d.to(Device::Host);
    auto* c = C_h.data_as<float>();
    auto* r = C_ref.data_as<float>();
    float tol = 1e-4f * static_cast<float>(K);
    float max_diff = 0.f;
    for (int i = 0; i < M * N; ++i) {
        max_diff = std::fmax(max_diff, std::fabs(c[i] - r[i]));
    }
    std::printf("      max_diff=%.3e tol=%.3e\n", max_diff, tol);
    MLLM_EXPECT(max_diff <= tol);
}

}  // namespace

MLLM_TEST(test_cublas_sgemm_correctness) {
    std::mt19937 rng(42);
    run_case(4, 6, 8, rng);
    run_case(17, 13, 11, rng);
    run_case(64, 64, 64, rng);
    run_case(128, 256, 96, rng);
}

MLLM_TEST(test_cublas_sgemm_transposes) {
    std::mt19937 rng(99);
    // All four trans_a/trans_b combinations
    run_case_transposed(8, 12, 10, false, false, rng);
    run_case_transposed(8, 12, 10, true, false, rng);
    run_case_transposed(8, 12, 10, false, true, rng);
    run_case_transposed(8, 12, 10, true, true, rng);
    run_case_transposed(33, 17, 25, true, false, rng);
    run_case_transposed(33, 17, 25, false, true, rng);
}

MLLM_TEST(test_cublas_sgemm_alpha_beta) {
    std::mt19937 rng(7);
    const int M = 8, N = 12, K = 10;
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    Tensor A_h({M, K}, DType::FP32, Device::Host);
    Tensor B_h({K, N}, DType::FP32, Device::Host);
    Tensor C_h({M, N}, DType::FP32, Device::Host);
    auto* a = A_h.data_as<float>();
    auto* b = B_h.data_as<float>();
    auto* c0 = C_h.data_as<float>();
    for (int i = 0; i < M * K; ++i) a[i] = dist(rng);
    for (int i = 0; i < K * N; ++i) b[i] = dist(rng);
    for (int i = 0; i < M * N; ++i) c0[i] = dist(rng);

    // Reference: C_ref = 2.0 * A @ B + 0.5 * C0
    Tensor C_ref({M, N}, DType::FP32, Device::Host);
    std::vector<float> tmp(M * N);
    gemm_naive_host(tmp.data(), a, b, M, N, K);
    auto* r = C_ref.data_as<float>();
    for (int i = 0; i < M * N; ++i) r[i] = 2.0f * tmp[i] + 0.5f * c0[i];

    // Device path
    Tensor A_d = A_h.to(Device::Cuda);
    Tensor B_d = B_h.to(Device::Cuda);
    Tensor C_d = C_h.to(Device::Cuda);  // initial C (will be scaled by beta)

    CublasHandle handle;
    modernllm::gemm_fp32_rowmajor(handle,
                                  C_d.data_as<float>(),
                                  A_d.data_as<float>(),
                                  B_d.data_as<float>(),
                                  M, N, K,
                                  /*trans_a=*/false, /*trans_b=*/false,
                                  /*alpha=*/2.0f, /*beta=*/0.5f);

    Tensor C_back = C_d.to(Device::Host);
    auto* cb = C_back.data_as<float>();

    float tol = 1e-4f * static_cast<float>(K);
    float max_diff = 0.f;
    for (int i = 0; i < M * N; ++i) {
        float d = std::fabs(cb[i] - r[i]);
        if (d > max_diff) max_diff = d;
    }
    std::printf("    alpha/beta max_diff=%.3e tol=%.3e\n", max_diff, tol);
    MLLM_EXPECT(max_diff <= tol);
}

int main() {
    MLLM_RUN_TEST(test_cublas_sgemm_correctness);
    MLLM_RUN_TEST(test_cublas_sgemm_transposes);
    MLLM_RUN_TEST(test_cublas_sgemm_alpha_beta);
    std::printf("\nAll gemm tests passed.\n");
    return 0;
}
