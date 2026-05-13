// GELU forward + backward (finite-diff against tanh-approx).

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "core/tensor.h"
#include "model/activations.h"
#include "tests/test_util.h"

using modernllm::Device;
using modernllm::DType;
using modernllm::Tensor;

namespace {

float gelu_host(float x) {
    constexpr float c = 0.7978845608028654f;
    constexpr float a = 0.044715f;
    float u = c * (x + a * x * x * x);
    return 0.5f * x * (1.f + std::tanh(u));
}

}  // namespace

MLLM_TEST(test_gelu_forward) {
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-3.f, 3.f);
    const int N = 1024;

    Tensor x_h({N}, DType::FP32, Device::Host);
    auto* xh = x_h.data_as<float>();
    for (int i = 0; i < N; ++i) xh[i] = dist(rng);

    Tensor x_d = x_h.to(Device::Cuda);
    Tensor y_d({N}, DType::FP32, Device::Cuda);
    modernllm::gelu_forward(x_d, y_d);
    Tensor y_h = y_d.to(Device::Host);
    auto* y = y_h.data_as<float>();

    float max_d = 0.f;
    for (int i = 0; i < N; ++i) {
        float ref = gelu_host(xh[i]);
        max_d = std::fmax(max_d, std::fabs(y[i] - ref));
    }
    std::printf("    max_diff=%.3e\n", max_d);
    MLLM_EXPECT(max_d < 1e-5f);
}

MLLM_TEST(test_gelu_backward_finite_diff) {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-2.f, 2.f);
    const int N = 64;

    std::vector<float> xv(N), upv(N);
    for (int i = 0; i < N; ++i) {
        xv[i] = dist(rng);
        upv[i] = dist(rng);
    }

    Tensor x_h({N}, DType::FP32, Device::Host);
    Tensor up_h({N}, DType::FP32, Device::Host);
    std::memcpy(x_h.data(), xv.data(), N * sizeof(float));
    std::memcpy(up_h.data(), upv.data(), N * sizeof(float));
    Tensor x_d = x_h.to(Device::Cuda);
    Tensor up_d = up_h.to(Device::Cuda);
    Tensor dx_d({N}, DType::FP32, Device::Cuda);
    modernllm::gelu_backward(x_d, up_d, dx_d);
    Tensor dx_h = dx_d.to(Device::Host);
    auto* dxa = dx_h.data_as<float>();

    const float eps = 1e-3f;
    float max_d = 0.f;
    for (int i = 0; i < N; ++i) {
        float lp = gelu_host(xv[i] + eps) * upv[i];
        float lm = gelu_host(xv[i] - eps) * upv[i];
        float num = (lp - lm) / (2.f * eps);
        max_d = std::fmax(max_d, std::fabs(num - dxa[i]));
    }
    std::printf("    max_diff=%.3e\n", max_d);
    MLLM_EXPECT(max_d < 1e-3f);
}

namespace {
float silu_host(float x) {
    return x / (1.f + std::exp(-x));
}
}

MLLM_TEST(test_silu_mul_forward) {
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> d(-2.f, 2.f);
    const int N = 256;
    std::vector<float> g(N), u(N), ref(N);
    for (auto& v : g) v = d(rng);
    for (auto& v : u) v = d(rng);
    for (int i = 0; i < N; ++i) ref[i] = silu_host(g[i]) * u[i];

    Tensor g_h({N}, DType::FP32, Device::Host);
    Tensor u_h({N}, DType::FP32, Device::Host);
    std::memcpy(g_h.data(), g.data(), N * sizeof(float));
    std::memcpy(u_h.data(), u.data(), N * sizeof(float));
    Tensor g_d = g_h.to(Device::Cuda);
    Tensor u_d = u_h.to(Device::Cuda);
    Tensor o_d({N}, DType::FP32, Device::Cuda);
    modernllm::silu_mul_forward(g_d, u_d, o_d);
    Tensor o_h = o_d.to(Device::Host);
    auto* o = o_h.data_as<float>();
    float max = 0.f;
    for (int i = 0; i < N; ++i) max = std::fmax(max, std::fabs(o[i] - ref[i]));
    std::printf("    max=%.3e\n", max);
    MLLM_EXPECT(max < 1e-5f);
}

MLLM_TEST(test_silu_mul_backward_finite_diff) {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> d(-1.5f, 1.5f);
    const int N = 32;
    std::vector<float> g(N), u(N), ups(N);
    for (auto& v : g) v = d(rng);
    for (auto& v : u) v = d(rng);
    for (auto& v : ups) v = d(rng);

    auto loss = [&]() {
        double s = 0.0;
        for (int i = 0; i < N; ++i) s += silu_host(g[i]) * u[i] * ups[i];
        return static_cast<float>(s);
    };

    Tensor g_h({N}, DType::FP32, Device::Host);
    Tensor u_h({N}, DType::FP32, Device::Host);
    Tensor up_h({N}, DType::FP32, Device::Host);
    std::memcpy(g_h.data(), g.data(), N * sizeof(float));
    std::memcpy(u_h.data(), u.data(), N * sizeof(float));
    std::memcpy(up_h.data(), ups.data(), N * sizeof(float));
    Tensor g_d = g_h.to(Device::Cuda);
    Tensor u_d = u_h.to(Device::Cuda);
    Tensor up_d = up_h.to(Device::Cuda);
    Tensor dg_d({N}, DType::FP32, Device::Cuda);
    Tensor du_d({N}, DType::FP32, Device::Cuda);
    modernllm::silu_mul_backward(g_d, u_d, up_d, dg_d, du_d);
    Tensor dg_h = dg_d.to(Device::Host);
    Tensor du_h = du_d.to(Device::Host);
    auto* dga = dg_h.data_as<float>();
    auto* dua = du_h.data_as<float>();

    const float feps = 1e-3f;
    auto perturb = [&](std::vector<float>& v, int i) {
        float saved = v[i];
        v[i] = saved + feps; float lp = loss();
        v[i] = saved - feps; float lm = loss();
        v[i] = saved;
        return (lp - lm) / (2.f * feps);
    };
    float max_g = 0.f, max_u = 0.f;
    for (int i = 0; i < N; ++i) {
        max_g = std::fmax(max_g, std::fabs(perturb(g, i) - dga[i]));
        max_u = std::fmax(max_u, std::fabs(perturb(u, i) - dua[i]));
    }
    std::printf("    dgate=%.3e dup=%.3e\n", max_g, max_u);
    MLLM_EXPECT(max_g < 1e-3f);
    MLLM_EXPECT(max_u < 1e-3f);
}

int main() {
    MLLM_RUN_TEST(test_gelu_forward);
    MLLM_RUN_TEST(test_gelu_backward_finite_diff);
    MLLM_RUN_TEST(test_silu_mul_forward);
    MLLM_RUN_TEST(test_silu_mul_backward_finite_diff);
    std::printf("\nAll activation tests passed.\n");
    return 0;
}
