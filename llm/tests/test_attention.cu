// Causal scaled-dot attention: forward correctness + finite-diff backward.

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "core/gemm.h"
#include "core/tensor.h"
#include "model/attention.h"
#include "tests/test_util.h"

using modernllm::CublasHandle;
using modernllm::Device;
using modernllm::DType;
using modernllm::Tensor;

namespace {

// Host reference. Returns ctx; if probs_out not null, also fills probs.
void attn_host(const float* q, const float* k, const float* v,
                int B, int T, int D, float* ctx, float* probs_out = nullptr) {
    float scale = 1.f / std::sqrt(static_cast<float>(D));
    std::vector<float> probs_buf(static_cast<std::size_t>(T) * T);
    for (int b = 0; b < B; ++b) {
        const float* qb = q + b * T * D;
        const float* kb = k + b * T * D;
        const float* vb = v + b * T * D;
        float* cb = ctx + b * T * D;

        // scores
        std::vector<float> scores(static_cast<std::size_t>(T) * T, 0.f);
        for (int t1 = 0; t1 < T; ++t1) {
            for (int t2 = 0; t2 < T; ++t2) {
                if (t2 > t1) { scores[t1 * T + t2] = -INFINITY; continue; }
                double s = 0.0;
                for (int d = 0; d < D; ++d) {
                    s += static_cast<double>(qb[t1 * D + d]) *
                         static_cast<double>(kb[t2 * D + d]);
                }
                scores[t1 * T + t2] = static_cast<float>(s * scale);
            }
        }
        // softmax per row
        for (int t1 = 0; t1 < T; ++t1) {
            float row_max = -INFINITY;
            for (int t2 = 0; t2 <= t1; ++t2) {
                row_max = std::fmax(row_max, scores[t1 * T + t2]);
            }
            double sum = 0.0;
            for (int t2 = 0; t2 <= t1; ++t2) {
                sum += std::exp(scores[t1 * T + t2] - row_max);
            }
            for (int t2 = 0; t2 < T; ++t2) {
                if (t2 > t1) probs_buf[t1 * T + t2] = 0.f;
                else probs_buf[t1 * T + t2] = static_cast<float>(
                    std::exp(scores[t1 * T + t2] - row_max) / sum);
            }
        }
        if (probs_out) {
            std::memcpy(probs_out + b * T * T, probs_buf.data(),
                         T * T * sizeof(float));
        }
        // ctx
        for (int t1 = 0; t1 < T; ++t1) {
            for (int d = 0; d < D; ++d) {
                double s = 0.0;
                for (int t2 = 0; t2 <= t1; ++t2) {
                    s += probs_buf[t1 * T + t2] * vb[t2 * D + d];
                }
                cb[t1 * D + d] = static_cast<float>(s);
            }
        }
    }
}

}  // namespace

MLLM_TEST(test_attn_forward_matches_host) {
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    const int B = 2, T = 5, D = 6;

    std::vector<float> q(B * T * D), k(B * T * D), v(B * T * D);
    std::vector<float> ctx_ref(B * T * D);
    std::vector<float> probs_ref(B * T * T);
    for (auto& x : q) x = d(rng);
    for (auto& x : k) x = d(rng);
    for (auto& x : v) x = d(rng);
    attn_host(q.data(), k.data(), v.data(), B, T, D,
               ctx_ref.data(), probs_ref.data());

    auto upload3 = [&](const std::vector<float>& src,
                        std::initializer_list<std::int64_t> shape) {
        Tensor h(shape, DType::FP32, Device::Host);
        std::memcpy(h.data(), src.data(), src.size() * sizeof(float));
        Tensor d_(shape, DType::FP32, Device::Cuda);
        d_.copy_from(h);
        return d_;
    };
    Tensor q_d = upload3(q, {B, T, D});
    Tensor k_d = upload3(k, {B, T, D});
    Tensor v_d = upload3(v, {B, T, D});
    Tensor ctx_d({B, T, D}, DType::FP32, Device::Cuda);
    Tensor probs_d({B, T, T}, DType::FP32, Device::Cuda);

    CublasHandle h;
    modernllm::scaled_dot_attention_forward(h, q_d, k_d, v_d, ctx_d, probs_d,
                                              B, T, D);

    Tensor ctx_h = ctx_d.to(Device::Host);
    Tensor pr_h = probs_d.to(Device::Host);
    auto* c = ctx_h.data_as<float>();
    auto* p = pr_h.data_as<float>();
    float max_c = 0.f, max_p = 0.f;
    for (int i = 0; i < B * T * D; ++i)
        max_c = std::fmax(max_c, std::fabs(c[i] - ctx_ref[i]));
    for (int i = 0; i < B * T * T; ++i)
        max_p = std::fmax(max_p, std::fabs(p[i] - probs_ref[i]));
    std::printf("    ctx max=%.3e  probs max=%.3e\n", max_c, max_p);
    MLLM_EXPECT(max_c < 1e-4f);
    MLLM_EXPECT(max_p < 1e-5f);
}

MLLM_TEST(test_attn_backward_finite_diff) {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> d(-0.5f, 0.5f);
    const int B = 2, T = 4, D = 5;

    std::vector<float> q(B * T * D), k(B * T * D), v(B * T * D);
    std::vector<float> upstream(B * T * D);
    for (auto& x : q) x = d(rng);
    for (auto& x : k) x = d(rng);
    for (auto& x : v) x = d(rng);
    for (auto& x : upstream) x = d(rng);

    auto loss = [&]() {
        std::vector<float> ctx(B * T * D);
        attn_host(q.data(), k.data(), v.data(), B, T, D, ctx.data());
        double s = 0.0;
        for (int i = 0; i < B * T * D; ++i)
            s += static_cast<double>(ctx[i]) * upstream[i];
        return static_cast<float>(s);
    };

    auto upload3 = [&](const std::vector<float>& src,
                        std::initializer_list<std::int64_t> shape) {
        Tensor h(shape, DType::FP32, Device::Host);
        std::memcpy(h.data(), src.data(), src.size() * sizeof(float));
        Tensor d_(shape, DType::FP32, Device::Cuda);
        d_.copy_from(h);
        return d_;
    };
    Tensor q_d = upload3(q, {B, T, D});
    Tensor k_d = upload3(k, {B, T, D});
    Tensor v_d = upload3(v, {B, T, D});
    Tensor ctx_d({B, T, D}, DType::FP32, Device::Cuda);
    Tensor probs_d({B, T, T}, DType::FP32, Device::Cuda);
    Tensor dctx_d = upload3(upstream, {B, T, D});
    Tensor dq_d({B, T, D}, DType::FP32, Device::Cuda);
    Tensor dk_d({B, T, D}, DType::FP32, Device::Cuda);
    Tensor dv_d({B, T, D}, DType::FP32, Device::Cuda);

    CublasHandle h;
    modernllm::scaled_dot_attention_forward(h, q_d, k_d, v_d, ctx_d, probs_d,
                                              B, T, D);
    modernllm::scaled_dot_attention_backward(h, dctx_d, q_d, k_d, v_d, probs_d,
                                               dq_d, dk_d, dv_d, B, T, D);

    auto download = [](Tensor& t) { return t.to(Device::Host); };
    Tensor dq_h = download(dq_d), dk_h = download(dk_d), dv_h = download(dv_d);
    auto* dqa = dq_h.data_as<float>();
    auto* dka = dk_h.data_as<float>();
    auto* dva = dv_h.data_as<float>();

    const float feps = 1e-3f;
    auto perturb = [&](std::vector<float>& vec, int i) {
        float saved = vec[i];
        vec[i] = saved + feps;
        float lp = loss();
        vec[i] = saved - feps;
        float lm = loss();
        vec[i] = saved;
        return (lp - lm) / (2.f * feps);
    };

    float max_q = 0.f, max_k = 0.f, max_v = 0.f;
    for (int i = 0; i < B * T * D; ++i) {
        max_q = std::fmax(max_q, std::fabs(perturb(q, i) - dqa[i]));
        max_k = std::fmax(max_k, std::fabs(perturb(k, i) - dka[i]));
        max_v = std::fmax(max_v, std::fabs(perturb(v, i) - dva[i]));
    }
    std::printf("    dq=%.3e dk=%.3e dv=%.3e\n", max_q, max_k, max_v);
    MLLM_EXPECT(max_q < 5e-3f);
    MLLM_EXPECT(max_k < 5e-3f);
    MLLM_EXPECT(max_v < 5e-3f);
}

int main() {
    MLLM_RUN_TEST(test_attn_forward_matches_host);
    MLLM_RUN_TEST(test_attn_backward_finite_diff);
    std::printf("\nAll attention tests passed.\n");
    return 0;
}
