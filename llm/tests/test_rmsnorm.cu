#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "core/tensor.h"
#include "model/rmsnorm.h"
#include "tests/test_util.h"

using modernllm::Device;
using modernllm::DType;
using modernllm::Tensor;

namespace {

void rmsnorm_host(const float* x, const float* gamma, float* y,
                   int N, int D, float eps) {
    for (int n = 0; n < N; ++n) {
        double sq = 0.0;
        for (int d = 0; d < D; ++d) sq += x[n * D + d] * x[n * D + d];
        float ms = static_cast<float>(sq / D);
        float rstd = 1.f / std::sqrt(ms + eps);
        for (int d = 0; d < D; ++d) {
            y[n * D + d] = x[n * D + d] * rstd * gamma[d];
        }
    }
}

}  // namespace

MLLM_TEST(test_rmsnorm_forward) {
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> d(-2.f, 2.f);
    const int N = 7, D = 13;
    const float eps = 1e-5f;

    std::vector<float> x(N * D), gamma(D), y_ref(N * D);
    for (auto& v : x) v = d(rng);
    for (auto& v : gamma) v = d(rng) * 0.5f + 1.0f;
    rmsnorm_host(x.data(), gamma.data(), y_ref.data(), N, D, eps);

    Tensor x_h({N, D}, DType::FP32, Device::Host);
    Tensor g_h({D}, DType::FP32, Device::Host);
    std::memcpy(x_h.data(), x.data(), x.size() * sizeof(float));
    std::memcpy(g_h.data(), gamma.data(), gamma.size() * sizeof(float));
    Tensor x_d = x_h.to(Device::Cuda);
    Tensor g_d = g_h.to(Device::Cuda);
    Tensor y_d({N, D}, DType::FP32, Device::Cuda);
    Tensor rstd_d({N}, DType::FP32, Device::Cuda);

    modernllm::rmsnorm_forward(x_d, g_d, eps, y_d, rstd_d);

    Tensor y_h = y_d.to(Device::Host);
    auto* y = y_h.data_as<float>();
    float max = 0.f;
    for (int i = 0; i < N * D; ++i) max = std::fmax(max, std::fabs(y[i] - y_ref[i]));
    std::printf("    max=%.3e\n", max);
    MLLM_EXPECT(max < 1e-4f);
}

MLLM_TEST(test_rmsnorm_backward_finite_diff) {
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    const int N = 4, D = 6;
    const float eps = 1e-5f;

    std::vector<float> x(N * D), gamma(D), upstream(N * D);
    for (auto& v : x) v = d(rng);
    for (auto& v : gamma) v = d(rng) * 0.3f + 1.0f;
    for (auto& v : upstream) v = d(rng);

    auto loss = [&]() {
        std::vector<float> y(N * D);
        rmsnorm_host(x.data(), gamma.data(), y.data(), N, D, eps);
        double s = 0.0;
        for (int i = 0; i < N * D; ++i) s += y[i] * upstream[i];
        return static_cast<float>(s);
    };

    Tensor x_h({N, D}, DType::FP32, Device::Host);
    Tensor g_h({D}, DType::FP32, Device::Host);
    Tensor up_h({N, D}, DType::FP32, Device::Host);
    std::memcpy(x_h.data(), x.data(), x.size() * sizeof(float));
    std::memcpy(g_h.data(), gamma.data(), gamma.size() * sizeof(float));
    std::memcpy(up_h.data(), upstream.data(), upstream.size() * sizeof(float));
    Tensor x_d = x_h.to(Device::Cuda);
    Tensor g_d = g_h.to(Device::Cuda);
    Tensor up_d = up_h.to(Device::Cuda);
    Tensor y_d({N, D}, DType::FP32, Device::Cuda);
    Tensor rstd_d({N}, DType::FP32, Device::Cuda);
    Tensor dx_d({N, D}, DType::FP32, Device::Cuda);
    Tensor dg_d = Tensor::zeros({D}, DType::FP32, Device::Cuda);

    modernllm::rmsnorm_forward(x_d, g_d, eps, y_d, rstd_d);
    modernllm::rmsnorm_backward(up_d, x_d, g_d, rstd_d, dx_d, dg_d);

    Tensor dx_h = dx_d.to(Device::Host);
    Tensor dg_h = dg_d.to(Device::Host);
    auto* dxa = dx_h.data_as<float>();
    auto* dga = dg_h.data_as<float>();

    const float feps = 1e-3f;
    auto perturb = [&](std::vector<float>& v, int i) {
        float saved = v[i];
        v[i] = saved + feps; float lp = loss();
        v[i] = saved - feps; float lm = loss();
        v[i] = saved;
        return (lp - lm) / (2.f * feps);
    };

    float max_x = 0.f, max_g = 0.f;
    for (int i = 0; i < N * D; ++i)
        max_x = std::fmax(max_x, std::fabs(perturb(x, i) - dxa[i]));
    for (int i = 0; i < D; ++i)
        max_g = std::fmax(max_g, std::fabs(perturb(gamma, i) - dga[i]));
    std::printf("    dx=%.3e dgamma=%.3e\n", max_x, max_g);
    MLLM_EXPECT(max_x < 5e-3f);
    MLLM_EXPECT(max_g < 5e-3f);
}

int main() {
    MLLM_RUN_TEST(test_rmsnorm_forward);
    MLLM_RUN_TEST(test_rmsnorm_backward_finite_diff);
    std::printf("\nAll RMSNorm tests passed.\n");
    return 0;
}
