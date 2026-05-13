// LayerNorm: forward vs host reference + finite-diff backward check.

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "core/tensor.h"
#include "model/layernorm.h"
#include "tests/test_util.h"

using modernllm::Device;
using modernllm::DType;
using modernllm::Tensor;

namespace {

void layernorm_host(const float* x, const float* gamma, const float* beta,
                    float* y, float* mean_out, float* rstd_out,
                    int N, int D, float eps) {
    for (int n = 0; n < N; ++n) {
        double sum = 0.0;
        for (int d = 0; d < D; ++d) sum += x[n * D + d];
        float mean = static_cast<float>(sum / D);
        double sq = 0.0;
        for (int d = 0; d < D; ++d) {
            double diff = x[n * D + d] - mean;
            sq += diff * diff;
        }
        float var = static_cast<float>(sq / D);
        float rstd = 1.f / std::sqrt(var + eps);
        if (mean_out) mean_out[n] = mean;
        if (rstd_out) rstd_out[n] = rstd;
        for (int d = 0; d < D; ++d) {
            float xhat = (x[n * D + d] - mean) * rstd;
            y[n * D + d] = xhat * gamma[d] + beta[d];
        }
    }
}

}  // namespace

MLLM_TEST(test_layernorm_forward_matches_host) {
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-2.f, 2.f);
    const int N = 9, D = 17;
    const float eps = 1e-5f;

    std::vector<float> x(N * D), gamma(D), beta(D), y_ref(N * D);
    for (auto& v : x) v = dist(rng);
    for (auto& v : gamma) v = dist(rng);
    for (auto& v : beta) v = dist(rng);
    layernorm_host(x.data(), gamma.data(), beta.data(),
                   y_ref.data(), nullptr, nullptr, N, D, eps);

    Tensor x_d({N, D}, DType::FP32, Device::Cuda);
    Tensor g_d({D}, DType::FP32, Device::Cuda);
    Tensor b_d({D}, DType::FP32, Device::Cuda);
    Tensor y_d({N, D}, DType::FP32, Device::Cuda);
    Tensor mean_d({N}, DType::FP32, Device::Cuda);
    Tensor rstd_d({N}, DType::FP32, Device::Cuda);

    auto upload = [&](const std::vector<float>& src, Tensor& dst) {
        Tensor h(dst.shape(), DType::FP32, Device::Host);
        std::memcpy(h.data(), src.data(), src.size() * sizeof(float));
        dst.copy_from(h);
    };
    upload(x, x_d); upload(gamma, g_d); upload(beta, b_d);

    modernllm::layernorm_forward(x_d, g_d, b_d, eps, y_d, mean_d, rstd_d);

    Tensor y_h = y_d.to(Device::Host);
    auto* y = y_h.data_as<float>();
    float max_d = 0.f;
    for (int i = 0; i < N * D; ++i) {
        max_d = std::fmax(max_d, std::fabs(y[i] - y_ref[i]));
    }
    std::printf("    fwd max_diff=%.3e\n", max_d);
    MLLM_EXPECT(max_d < 1e-4f);
}

MLLM_TEST(test_layernorm_backward_finite_diff) {
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    const int N = 4, D = 6;
    const float eps = 1e-5f;

    std::vector<float> x(N * D), gamma(D), beta(D), upstream(N * D);
    for (auto& v : x) v = dist(rng);
    for (auto& v : gamma) v = dist(rng) * 0.5f + 1.0f;  // around 1
    for (auto& v : beta) v = dist(rng) * 0.1f;
    for (auto& v : upstream) v = dist(rng);

    auto loss = [&]() {
        std::vector<float> y(N * D);
        layernorm_host(x.data(), gamma.data(), beta.data(),
                       y.data(), nullptr, nullptr, N, D, eps);
        double s = 0.0;
        for (int i = 0; i < N * D; ++i) s += y[i] * upstream[i];
        return static_cast<float>(s);
    };

    // Analytical via GPU
    Tensor x_d({N, D}, DType::FP32, Device::Cuda);
    Tensor g_d({D}, DType::FP32, Device::Cuda);
    Tensor b_d({D}, DType::FP32, Device::Cuda);
    Tensor y_d({N, D}, DType::FP32, Device::Cuda);
    Tensor mean_d({N}, DType::FP32, Device::Cuda);
    Tensor rstd_d({N}, DType::FP32, Device::Cuda);
    Tensor dy_d({N, D}, DType::FP32, Device::Cuda);
    Tensor dx_d({N, D}, DType::FP32, Device::Cuda);
    Tensor dg_d = Tensor::zeros({D}, DType::FP32, Device::Cuda);
    Tensor db_d = Tensor::zeros({D}, DType::FP32, Device::Cuda);

    auto upload = [&](const std::vector<float>& src, Tensor& dst) {
        Tensor h(dst.shape(), DType::FP32, Device::Host);
        std::memcpy(h.data(), src.data(), src.size() * sizeof(float));
        dst.copy_from(h);
    };
    upload(x, x_d); upload(gamma, g_d); upload(beta, b_d);
    upload(upstream, dy_d);

    modernllm::layernorm_forward(x_d, g_d, b_d, eps, y_d, mean_d, rstd_d);
    modernllm::layernorm_backward(dy_d, x_d, g_d, mean_d, rstd_d,
                                    dx_d, dg_d, db_d);

    Tensor dx_h = dx_d.to(Device::Host);
    Tensor dg_h = dg_d.to(Device::Host);
    Tensor db_h = db_d.to(Device::Host);
    auto* dxa = dx_h.data_as<float>();
    auto* dga = dg_h.data_as<float>();
    auto* dba = db_h.data_as<float>();

    const float feps = 1e-3f;
    auto perturb = [&](std::vector<float>& v, int i) {
        float saved = v[i];
        v[i] = saved + feps;
        float lp = loss();
        v[i] = saved - feps;
        float lm = loss();
        v[i] = saved;
        return (lp - lm) / (2.f * feps);
    };

    float max_dX = 0.f, max_dG = 0.f, max_dB = 0.f;
    for (int i = 0; i < N * D; ++i) {
        max_dX = std::fmax(max_dX, std::fabs(perturb(x, i) - dxa[i]));
    }
    for (int i = 0; i < D; ++i) {
        max_dG = std::fmax(max_dG, std::fabs(perturb(gamma, i) - dga[i]));
        max_dB = std::fmax(max_dB, std::fabs(perturb(beta, i) - dba[i]));
    }
    std::printf("    dX max=%.3e  dG max=%.3e  dB max=%.3e\n",
                 max_dX, max_dG, max_dB);
    MLLM_EXPECT(max_dX < 5e-3f);
    MLLM_EXPECT(max_dG < 5e-3f);
    MLLM_EXPECT(max_dB < 5e-3f);
}

int main() {
    MLLM_RUN_TEST(test_layernorm_forward_matches_host);
    MLLM_RUN_TEST(test_layernorm_backward_finite_diff);
    std::printf("\nAll layernorm tests passed.\n");
    return 0;
}
