// BF16: cast round-trip + cuBLAS BF16 GEMM correctness vs FP32 reference.

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "core/cast.h"
#include "core/dtype.h"
#include "core/gemm.h"
#include "core/tensor.h"
#include "tests/test_util.h"

using modernllm::CublasHandle;
using modernllm::Device;
using modernllm::DType;
using modernllm::Tensor;

namespace {

void naive_gemm(const float* A, const float* B, float* C,
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

}  // namespace

MLLM_TEST(test_cast_fp32_bf16_round_trip) {
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-2.f, 2.f);
    const int N = 1024;

    Tensor h_fp32({N}, DType::FP32, Device::Host);
    auto* hp = h_fp32.data_as<float>();
    for (int i = 0; i < N; ++i) hp[i] = dist(rng);

    Tensor d_fp32 = h_fp32.to(Device::Cuda);
    Tensor d_bf16({N}, DType::BF16, Device::Cuda);
    Tensor d_back({N}, DType::FP32, Device::Cuda);

    modernllm::cast_fp32_to_bf16(d_fp32, d_bf16);
    modernllm::cast_bf16_to_fp32(d_bf16, d_back);

    Tensor h_back = d_back.to(Device::Host);
    auto* bp = h_back.data_as<float>();
    float max_d = 0.f;
    for (int i = 0; i < N; ++i) {
        // BF16 has 7 bits of mantissa; relative error ~1/128 ≈ 7.8e-3.
        float d = std::fabs(bp[i] - hp[i]);
        max_d = std::fmax(max_d, d);
    }
    std::printf("    max BF16 round-trip diff = %.3e\n", max_d);
    // Values up to ~2 in magnitude → absolute error <= 2 * 2^-7 = 0.0156.
    MLLM_EXPECT(max_d < 0.02f);
}

MLLM_TEST(test_bf16_gemm_correctness) {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    const int M = 64, N = 96, K = 80;

    std::vector<float> A(M * K), B(K * N), C_ref(M * N);
    for (auto& v : A) v = dist(rng);
    for (auto& v : B) v = dist(rng);
    naive_gemm(A.data(), B.data(), C_ref.data(), M, N, K);

    Tensor A_h({M, K}, DType::FP32, Device::Host);
    Tensor B_h({K, N}, DType::FP32, Device::Host);
    std::memcpy(A_h.data(), A.data(), A.size() * sizeof(float));
    std::memcpy(B_h.data(), B.data(), B.size() * sizeof(float));
    Tensor A_fp32 = A_h.to(Device::Cuda);
    Tensor B_fp32 = B_h.to(Device::Cuda);

    Tensor A_bf16({M, K}, DType::BF16, Device::Cuda);
    Tensor B_bf16({K, N}, DType::BF16, Device::Cuda);
    modernllm::cast_fp32_to_bf16(A_fp32, A_bf16);
    modernllm::cast_fp32_to_bf16(B_fp32, B_bf16);

    Tensor C_d({M, N}, DType::FP32, Device::Cuda);
    CublasHandle h;
    modernllm::gemm_bf16in_fp32out_rowmajor(
        h,
        C_d.data_as<float>(),
        static_cast<const unsigned short*>(A_bf16.data()),
        static_cast<const unsigned short*>(B_bf16.data()),
        M, N, K);

    Tensor C_h = C_d.to(Device::Host);
    auto* c = C_h.data_as<float>();
    auto* r = C_ref.data();
    float max_d = 0.f;
    for (int i = 0; i < M * N; ++i) {
        max_d = std::fmax(max_d, std::fabs(c[i] - r[i]));
    }
    // BF16 has 7 mantissa bits. K-sum of ~K products of [-1,1] values
    // accumulates roughly sqrt(K) * eps_bf16. For K=80 expect ~0.07,
    // we allow some headroom.
    float tol = 0.5f;
    std::printf("    BF16 GEMM max_diff=%.3e tol=%.3e (K=%d)\n",
                 max_d, tol, K);
    MLLM_EXPECT(max_d < tol);
}

MLLM_TEST(test_bf16_gemm_transpose) {
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    const int M = 32, N = 24, K = 40;

    // Logical: A_log [M,K], B_log [K,N], reference C = A_log @ B_log.
    std::vector<float> A_log(M * K), B_log(K * N), C_ref(M * N);
    for (auto& v : A_log) v = dist(rng);
    for (auto& v : B_log) v = dist(rng);
    naive_gemm(A_log.data(), B_log.data(), C_ref.data(), M, N, K);

    // Store A transposed [K,M] and pass trans_a=true. B normal [K,N].
    std::vector<float> A_T(K * M);
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k)
            A_T[k * M + i] = A_log[i * K + k];

    auto upload = [](const std::vector<float>& src,
                      std::initializer_list<std::int64_t> shape) {
        Tensor h(shape, DType::FP32, Device::Host);
        std::memcpy(h.data(), src.data(), src.size() * sizeof(float));
        Tensor d_fp32(shape, DType::FP32, Device::Cuda);
        d_fp32.copy_from(h);
        Tensor d_bf16(shape, DType::BF16, Device::Cuda);
        modernllm::cast_fp32_to_bf16(d_fp32, d_bf16);
        return d_bf16;
    };
    Tensor A_bf16 = upload(A_T, {K, M});
    Tensor B_bf16 = upload(B_log, {K, N});

    Tensor C_d({M, N}, DType::FP32, Device::Cuda);
    CublasHandle h;
    modernllm::gemm_bf16in_fp32out_rowmajor(
        h, C_d.data_as<float>(),
        static_cast<const unsigned short*>(A_bf16.data()),
        static_cast<const unsigned short*>(B_bf16.data()),
        M, N, K, /*trans_a=*/true, /*trans_b=*/false);

    Tensor C_h = C_d.to(Device::Host);
    auto* c = C_h.data_as<float>();
    float max_d = 0.f;
    for (int i = 0; i < M * N; ++i) {
        max_d = std::fmax(max_d, std::fabs(c[i] - C_ref[i]));
    }
    std::printf("    BF16 GEMM trans_a max_diff=%.3e\n", max_d);
    MLLM_EXPECT(max_d < 0.5f);
}

int main() {
    MLLM_RUN_TEST(test_cast_fp32_bf16_round_trip);
    MLLM_RUN_TEST(test_bf16_gemm_correctness);
    MLLM_RUN_TEST(test_bf16_gemm_transpose);
    std::printf("\nAll BF16 tests passed.\n");
    return 0;
}
