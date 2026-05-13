// FA2 forward correctness: must match scaled_dot_attention_forward to within
// the FP32 reordering tolerance (the online softmax computes things in a
// different order, so bit-equality is not guaranteed but values must match
// to ~1e-5 for our small sizes).

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "core/gemm.h"
#include "core/tensor.h"
#include "model/attention.h"
#include "model/flash_attn.h"
#include "tests/test_util.h"

using modernllm::CublasHandle;
using modernllm::Device;
using modernllm::DType;
using modernllm::Tensor;

namespace {

void run_case(int B, int T, int D, std::mt19937& rng, float tol) {
    std::printf("    case B=%d T=%d D=%d\n", B, T, D);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    auto upload = [&](std::initializer_list<std::int64_t> shape, int n) {
        Tensor h(shape, DType::FP32, Device::Host);
        auto* p = h.data_as<float>();
        for (int i = 0; i < n; ++i) p[i] = dist(rng);
        Tensor d(shape, DType::FP32, Device::Cuda);
        d.copy_from(h);
        return d;
    };

    Tensor q = upload({B, T, D}, B * T * D);
    Tensor k = upload({B, T, D}, B * T * D);
    Tensor v = upload({B, T, D}, B * T * D);

    // Reference: existing naive forward
    Tensor ctx_ref({B, T, D}, DType::FP32, Device::Cuda);
    Tensor probs_ref({B, T, T}, DType::FP32, Device::Cuda);
    CublasHandle handle;
    modernllm::scaled_dot_attention_forward(handle, q, k, v, ctx_ref,
                                              probs_ref, B, T, D);

    // FA2 forward
    Tensor ctx_fa({B, T, D}, DType::FP32, Device::Cuda);
    Tensor L({B, T}, DType::FP32, Device::Cuda);
    modernllm::flash_attn_forward(q, k, v, ctx_fa, L, B, T, D, /*tile_kv=*/16);

    Tensor a = ctx_ref.to(Device::Host);
    Tensor c = ctx_fa.to(Device::Host);
    auto* pa = a.data_as<float>();
    auto* pc = c.data_as<float>();
    float max_d = 0.f;
    for (int i = 0; i < B * T * D; ++i) {
        max_d = std::fmax(max_d, std::fabs(pa[i] - pc[i]));
    }
    std::printf("      max_diff=%.3e tol=%.3e\n", max_d, tol);
    MLLM_EXPECT(max_d < tol);
}

}  // namespace

MLLM_TEST(test_flash_attn_matches_naive_forward) {
    std::mt19937 rng(0);
    // Tolerances: small because online softmax is mathematically equivalent
    // to bulk softmax up to floating-point reassociation.
    run_case(/*B=*/2, /*T=*/8, /*D=*/4, rng, 1e-5f);
    run_case(/*B=*/2, /*T=*/16, /*D=*/8, rng, 1e-5f);
    run_case(/*B=*/2, /*T=*/64, /*D=*/16, rng, 1e-5f);
    run_case(/*B=*/4, /*T=*/128, /*D=*/64, rng, 5e-5f);
}

MLLM_TEST(test_flash_attn_logsumexp_consistent) {
    // Sanity: logsumexp output should match scalar logsumexp of the masked
    // pre-softmax scores (ground-truth via a host computation).
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    const int B = 1, T = 16, D = 8;

    std::vector<float> q(B * T * D), k(B * T * D), v(B * T * D);
    for (auto& x : q) x = dist(rng);
    for (auto& x : k) x = dist(rng);
    for (auto& x : v) x = dist(rng);

    auto upload = [&](const std::vector<float>& vec) {
        Tensor h({B, T, D}, DType::FP32, Device::Host);
        std::memcpy(h.data(), vec.data(), vec.size() * sizeof(float));
        Tensor d({B, T, D}, DType::FP32, Device::Cuda);
        d.copy_from(h);
        return d;
    };
    Tensor q_d = upload(q), k_d = upload(k), v_d = upload(v);
    Tensor ctx({B, T, D}, DType::FP32, Device::Cuda);
    Tensor L({B, T}, DType::FP32, Device::Cuda);
    modernllm::flash_attn_forward(q_d, k_d, v_d, ctx, L, B, T, D, 8);

    Tensor L_h = L.to(Device::Host);
    auto* lh = L_h.data_as<float>();

    float scale = 1.f / std::sqrt(static_cast<float>(D));
    float max_diff = 0.f;
    for (int b = 0; b < B; ++b) {
        for (int t1 = 0; t1 < T; ++t1) {
            // Compute logsumexp(scores[t1, :t1+1]) on host
            float row_max = -INFINITY;
            for (int t2 = 0; t2 <= t1; ++t2) {
                float s = 0.f;
                for (int d = 0; d < D; ++d)
                    s += q[(b * T + t1) * D + d] * k[(b * T + t2) * D + d];
                s *= scale;
                row_max = std::fmax(row_max, s);
            }
            float sum = 0.f;
            for (int t2 = 0; t2 <= t1; ++t2) {
                float s = 0.f;
                for (int d = 0; d < D; ++d)
                    s += q[(b * T + t1) * D + d] * k[(b * T + t2) * D + d];
                s *= scale;
                sum += std::exp(s - row_max);
            }
            float ref = row_max + std::log(sum);
            float got = lh[b * T + t1];
            max_diff = std::fmax(max_diff, std::fabs(got - ref));
        }
    }
    std::printf("    L max_diff=%.3e\n", max_diff);
    MLLM_EXPECT(max_diff < 1e-4f);
}

// FA backward correctness vs scaled_dot_attention_backward.
namespace {
void run_bwd_case(int B, int T, int D, int tile, std::mt19937& rng, float tol) {
    std::printf("    bwd case B=%d T=%d D=%d tile=%d\n", B, T, D, tile);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    auto upload = [&](std::initializer_list<std::int64_t> shape, int n) {
        Tensor h(shape, DType::FP32, Device::Host);
        auto* p = h.data_as<float>();
        for (int i = 0; i < n; ++i) p[i] = dist(rng);
        Tensor d(shape, DType::FP32, Device::Cuda);
        d.copy_from(h);
        return d;
    };

    Tensor q = upload({B, T, D}, B * T * D);
    Tensor k = upload({B, T, D}, B * T * D);
    Tensor v = upload({B, T, D}, B * T * D);
    Tensor d_ctx = upload({B, T, D}, B * T * D);

    CublasHandle handle;

    // Reference: naive forward + naive backward
    Tensor ctx_ref({B, T, D}, DType::FP32, Device::Cuda);
    Tensor probs_ref({B, T, T}, DType::FP32, Device::Cuda);
    modernllm::scaled_dot_attention_forward(handle, q, k, v, ctx_ref,
                                              probs_ref, B, T, D);
    Tensor dq_ref({B, T, D}, DType::FP32, Device::Cuda);
    Tensor dk_ref({B, T, D}, DType::FP32, Device::Cuda);
    Tensor dv_ref({B, T, D}, DType::FP32, Device::Cuda);
    modernllm::scaled_dot_attention_backward(handle, d_ctx, q, k, v, probs_ref,
                                              dq_ref, dk_ref, dv_ref, B, T, D);

    // FA forward + FA backward
    Tensor ctx_fa({B, T, D}, DType::FP32, Device::Cuda);
    Tensor L({B, T}, DType::FP32, Device::Cuda);
    modernllm::flash_attn_forward(q, k, v, ctx_fa, L, B, T, D, tile);
    Tensor dq_fa({B, T, D}, DType::FP32, Device::Cuda);
    Tensor dk_fa({B, T, D}, DType::FP32, Device::Cuda);
    Tensor dv_fa({B, T, D}, DType::FP32, Device::Cuda);
    modernllm::flash_attn_backward(q, k, v, ctx_fa, L, d_ctx,
                                    dq_fa, dk_fa, dv_fa, B, T, D, tile);

    auto max_diff = [&](const Tensor& a, const Tensor& b, int n) {
        Tensor ah = a.to(Device::Host);
        Tensor bh = b.to(Device::Host);
        auto* pa = ah.data_as<float>();
        auto* pb = bh.data_as<float>();
        float m = 0.f;
        for (int i = 0; i < n; ++i)
            m = std::fmax(m, std::fabs(pa[i] - pb[i]));
        return m;
    };
    float dq_diff = max_diff(dq_ref, dq_fa, B * T * D);
    float dk_diff = max_diff(dk_ref, dk_fa, B * T * D);
    float dv_diff = max_diff(dv_ref, dv_fa, B * T * D);
    std::printf("      dq=%.3e dk=%.3e dv=%.3e tol=%.3e\n",
                dq_diff, dk_diff, dv_diff, tol);
    MLLM_EXPECT(dq_diff < tol);
    MLLM_EXPECT(dk_diff < tol);
    MLLM_EXPECT(dv_diff < tol);
}
}  // namespace

MLLM_TEST(test_flash_attn_backward_matches_naive) {
    std::mt19937 rng(13);
    run_bwd_case(/*B=*/1, /*T=*/8,   /*D=*/4,  /*tile=*/4,  rng, 5e-5f);
    run_bwd_case(/*B=*/2, /*T=*/16,  /*D=*/8,  /*tile=*/8,  rng, 5e-5f);
    run_bwd_case(/*B=*/2, /*T=*/64,  /*D=*/16, /*tile=*/16, rng, 1e-4f);
    run_bwd_case(/*B=*/2, /*T=*/128, /*D=*/32, /*tile=*/16, rng, 2e-4f);
}

int main() {
    MLLM_RUN_TEST(test_flash_attn_matches_naive_forward);
    MLLM_RUN_TEST(test_flash_attn_logsumexp_consistent);
    MLLM_RUN_TEST(test_flash_attn_backward_matches_naive);
    std::printf("\nAll flash-attn tests passed.\n");
    return 0;
}
