// RoPE: in-place forward matches host reference, backward is the transpose.

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "core/tensor.h"
#include "model/rope.h"
#include "tests/test_util.h"

using modernllm::Device;
using modernllm::DType;
using modernllm::Tensor;

namespace {

void rope_host(const float* in, float* out, int N, int T, int d_h,
                float base) {
    int half = d_h / 2;
    for (int n = 0; n < N; ++n) {
        for (int t = 0; t < T; ++t) {
            for (int i = 0; i < half; ++i) {
                float theta = std::pow(base,
                                        -static_cast<float>(2 * i) /
                                            static_cast<float>(d_h));
                float angle = static_cast<float>(t) * theta;
                float c = std::cos(angle);
                float s = std::sin(angle);
                int base_idx = (n * T + t) * d_h;
                float x0 = in[base_idx + 2 * i];
                float x1 = in[base_idx + 2 * i + 1];
                out[base_idx + 2 * i]     = x0 * c - x1 * s;
                out[base_idx + 2 * i + 1] = x0 * s + x1 * c;
            }
        }
    }
}

}  // namespace

MLLM_TEST(test_rope_forward_matches_host) {
    const int N = 3, T = 7, d_h = 8;
    const float base = 10000.f;
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    std::vector<float> in(N * T * d_h);
    std::vector<float> ref(N * T * d_h);
    for (auto& x : in) x = dist(rng);
    rope_host(in.data(), ref.data(), N, T, d_h, base);

    Tensor in_h({N, T, d_h}, DType::FP32, Device::Host);
    std::memcpy(in_h.data(), in.data(), in.size() * sizeof(float));
    Tensor x_d = in_h.to(Device::Cuda);

    auto [cos_d, sin_d] =
        modernllm::make_rope_cache(T, d_h, base, Device::Cuda);
    modernllm::rope_apply_inplace(x_d, cos_d, sin_d, N, T, d_h);

    Tensor x_h_back = x_d.to(Device::Host);
    auto* x = x_h_back.data_as<float>();
    float max_d = 0.f;
    for (int i = 0; i < N * T * d_h; ++i)
        max_d = std::fmax(max_d, std::fabs(x[i] - ref[i]));
    std::printf("    fwd max=%.3e\n", max_d);
    MLLM_EXPECT(max_d < 1e-5f);
}

MLLM_TEST(test_rope_backward_finite_diff) {
    const int N = 2, T = 4, d_h = 6;
    const float base = 10000.f;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    std::vector<float> in(N * T * d_h), upstream(N * T * d_h);
    for (auto& x : in) x = dist(rng);
    for (auto& x : upstream) x = dist(rng);

    auto loss = [&]() {
        std::vector<float> out(N * T * d_h);
        rope_host(in.data(), out.data(), N, T, d_h, base);
        double s = 0.0;
        for (int i = 0; i < N * T * d_h; ++i) s += out[i] * upstream[i];
        return static_cast<float>(s);
    };

    Tensor up_h({N, T, d_h}, DType::FP32, Device::Host);
    std::memcpy(up_h.data(), upstream.data(), upstream.size() * sizeof(float));
    Tensor dx_d = up_h.to(Device::Cuda);

    auto [cos_d, sin_d] =
        modernllm::make_rope_cache(T, d_h, base, Device::Cuda);
    modernllm::rope_apply_backward_inplace(dx_d, cos_d, sin_d, N, T, d_h);

    Tensor dx_h = dx_d.to(Device::Host);
    auto* dxa = dx_h.data_as<float>();

    const float feps = 1e-3f;
    float max_d = 0.f;
    for (int i = 0; i < N * T * d_h; ++i) {
        float saved = in[i];
        in[i] = saved + feps; float lp = loss();
        in[i] = saved - feps; float lm = loss();
        in[i] = saved;
        float num = (lp - lm) / (2.f * feps);
        max_d = std::fmax(max_d, std::fabs(num - dxa[i]));
    }
    std::printf("    bwd max=%.3e\n", max_d);
    MLLM_EXPECT(max_d < 5e-4f);
}

int main() {
    MLLM_RUN_TEST(test_rope_forward_matches_host);
    MLLM_RUN_TEST(test_rope_backward_finite_diff);
    std::printf("\nAll RoPE tests passed.\n");
    return 0;
}
