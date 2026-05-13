// MHA support: split/merge are inverses; multi-head attention via
// split → attention → merge equals a direct host reference.

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

// Reference: multi-head causal attention on host.
//   q, k, v: [B, T, H*d_h]   (rows of d_h are interleaved per head)
//   per head h, per batch b: same as single-head causal attention.
void mha_host(const float* q, const float* k, const float* v,
               int B, int T, int H, int d_h, float* ctx) {
    int D = H * d_h;
    float scale = 1.f / std::sqrt(static_cast<float>(d_h));
    std::vector<float> probs(static_cast<std::size_t>(T) * T);

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < H; ++h) {
            // For this (b, h), build [T, d_h] views via strided indexing.
            auto qe = [&](int t, int k_idx) {
                return q[(b * T + t) * D + h * d_h + k_idx];
            };
            auto ke = [&](int t, int k_idx) {
                return k[(b * T + t) * D + h * d_h + k_idx];
            };
            auto ve = [&](int t, int k_idx) {
                return v[(b * T + t) * D + h * d_h + k_idx];
            };
            for (int t1 = 0; t1 < T; ++t1) {
                // scores
                float row_max = -INFINITY;
                std::vector<float> srow(T, -INFINITY);
                for (int t2 = 0; t2 <= t1; ++t2) {
                    double s = 0.0;
                    for (int j = 0; j < d_h; ++j) {
                        s += qe(t1, j) * ke(t2, j);
                    }
                    srow[t2] = static_cast<float>(s * scale);
                    row_max = std::fmax(row_max, srow[t2]);
                }
                double sum = 0.0;
                for (int t2 = 0; t2 <= t1; ++t2)
                    sum += std::exp(srow[t2] - row_max);
                for (int t2 = 0; t2 < T; ++t2) {
                    if (t2 > t1) probs[t1 * T + t2] = 0.f;
                    else probs[t1 * T + t2] = static_cast<float>(
                        std::exp(srow[t2] - row_max) / sum);
                }
                for (int j = 0; j < d_h; ++j) {
                    double c = 0.0;
                    for (int t2 = 0; t2 <= t1; ++t2)
                        c += probs[t1 * T + t2] * ve(t2, j);
                    ctx[(b * T + t1) * D + h * d_h + j] =
                        static_cast<float>(c);
                }
            }
        }
    }
}

}  // namespace

MLLM_TEST(test_split_merge_inverse) {
    const int B = 2, T = 3, H = 4, d_h = 5;
    const int D = H * d_h;
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    std::vector<float> in_h(B * T * D);
    for (auto& x : in_h) x = dist(rng);

    Tensor in_d({B, T, D}, DType::FP32, Device::Cuda);
    Tensor split_d({B * H, T, d_h}, DType::FP32, Device::Cuda);
    Tensor back_d({B, T, D}, DType::FP32, Device::Cuda);

    Tensor in_host({B, T, D}, DType::FP32, Device::Host);
    std::memcpy(in_host.data(), in_h.data(), in_h.size() * sizeof(float));
    in_d.copy_from(in_host);

    modernllm::split_heads(in_d, split_d, B, T, H, d_h);
    modernllm::merge_heads(split_d, back_d, B, T, H, d_h);

    Tensor back_host = back_d.to(Device::Host);
    auto* p = back_host.data_as<float>();
    float max_d = 0.f;
    for (int i = 0; i < static_cast<int>(in_h.size()); ++i)
        max_d = std::fmax(max_d, std::fabs(p[i] - in_h[i]));
    std::printf("    split-merge round-trip max=%.3e\n", max_d);
    MLLM_EXPECT(max_d == 0.f);
}

MLLM_TEST(test_mha_via_split_attn_merge) {
    const int B = 2, T = 5, H = 4, d_h = 6;
    const int D = H * d_h;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    std::vector<float> q(B * T * D), k(B * T * D), v(B * T * D);
    std::vector<float> ctx_ref(B * T * D);
    for (auto& x : q) x = dist(rng);
    for (auto& x : k) x = dist(rng);
    for (auto& x : v) x = dist(rng);
    mha_host(q.data(), k.data(), v.data(), B, T, H, d_h, ctx_ref.data());

    auto upload = [&](const std::vector<float>& src,
                       std::initializer_list<std::int64_t> shape) {
        Tensor h(shape, DType::FP32, Device::Host);
        std::memcpy(h.data(), src.data(), src.size() * sizeof(float));
        Tensor d(shape, DType::FP32, Device::Cuda);
        d.copy_from(h);
        return d;
    };
    Tensor q_d = upload(q, {B, T, D});
    Tensor k_d = upload(k, {B, T, D});
    Tensor v_d = upload(v, {B, T, D});
    Tensor ctx_d({B, T, D}, DType::FP32, Device::Cuda);

    Tensor q_split({B * H, T, d_h}, DType::FP32, Device::Cuda);
    Tensor k_split({B * H, T, d_h}, DType::FP32, Device::Cuda);
    Tensor v_split({B * H, T, d_h}, DType::FP32, Device::Cuda);
    Tensor ctx_split({B * H, T, d_h}, DType::FP32, Device::Cuda);
    Tensor probs({B * H, T, T}, DType::FP32, Device::Cuda);

    modernllm::split_heads(q_d, q_split, B, T, H, d_h);
    modernllm::split_heads(k_d, k_split, B, T, H, d_h);
    modernllm::split_heads(v_d, v_split, B, T, H, d_h);

    CublasHandle handle;
    modernllm::scaled_dot_attention_forward(handle, q_split, k_split, v_split,
                                              ctx_split, probs,
                                              B * H, T, d_h);
    modernllm::merge_heads(ctx_split, ctx_d, B, T, H, d_h);

    Tensor ctx_h = ctx_d.to(Device::Host);
    auto* c = ctx_h.data_as<float>();
    float max_d = 0.f;
    for (int i = 0; i < B * T * D; ++i)
        max_d = std::fmax(max_d, std::fabs(c[i] - ctx_ref[i]));
    std::printf("    mha vs host max=%.3e\n", max_d);
    MLLM_EXPECT(max_d < 1e-4f);
}

int main() {
    MLLM_RUN_TEST(test_split_merge_inverse);
    MLLM_RUN_TEST(test_mha_via_split_attn_merge);
    std::printf("\nAll MHA tests passed.\n");
    return 0;
}
