// T9 Phase A — verify the BF16 sister kernels match their FP32 originals
// within BF16 numerical precision (~7 mantissa bits → ~1/128 relative error).

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "core/cast.h"
#include "core/tensor.h"
#include "model/activations.h"
#include "model/rmsnorm.h"
#include "model/rope.h"
#include "tests/test_util.h"

using modernllm::Device;
using modernllm::DType;
using modernllm::Tensor;

namespace {

float max_abs_diff(const float* a, const float* b, int n) {
    float m = 0.f;
    for (int i = 0; i < n; ++i) m = std::fmax(m, std::fabs(a[i] - b[i]));
    return m;
}

Tensor cast_to_bf16(const Tensor& fp32) {
    Tensor bf16(fp32.shape(), DType::BF16, Device::Cuda);
    modernllm::cast_fp32_to_bf16(fp32, bf16);
    return bf16;
}

Tensor cast_to_fp32(const Tensor& bf16) {
    Tensor fp32(bf16.shape(), DType::FP32, Device::Cuda);
    modernllm::cast_bf16_to_fp32(bf16, fp32);
    return fp32;
}

}  // namespace

MLLM_TEST(test_add_inplace_bf16_matches_fp32) {
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-2.f, 2.f);
    const int N = 1024;

    std::vector<float> a_init(N), b_init(N);
    for (auto& x : a_init) x = dist(rng);
    for (auto& x : b_init) x = dist(rng);

    auto upload = [&](const std::vector<float>& v) {
        Tensor h({N}, DType::FP32, Device::Host);
        std::memcpy(h.data(), v.data(), v.size() * sizeof(float));
        return h.to(Device::Cuda);
    };
    Tensor a_fp32 = upload(a_init);
    Tensor b_fp32 = upload(b_init);
    modernllm::add_inplace(a_fp32, b_fp32);

    Tensor a_bf16 = cast_to_bf16(upload(a_init));
    Tensor b_bf16 = cast_to_bf16(upload(b_init));
    modernllm::add_inplace(a_bf16, b_bf16);

    Tensor a_back = cast_to_fp32(a_bf16);
    Tensor a_h = a_fp32.to(Device::Host);
    Tensor a_b_h = a_back.to(Device::Host);
    float d = max_abs_diff(a_h.data_as<float>(), a_b_h.data_as<float>(), N);
    std::printf("    add_inplace BF16 vs FP32 max=%.3e\n", d);
    MLLM_EXPECT(d < 0.05f);  // BF16 RTNE on input + output
}

MLLM_TEST(test_silu_mul_bf16_matches_fp32) {
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-1.5f, 1.5f);
    const int N = 1024;

    std::vector<float> g(N), u(N);
    for (auto& x : g) x = dist(rng);
    for (auto& x : u) x = dist(rng);

    auto upload = [&](const std::vector<float>& v) {
        Tensor h({N}, DType::FP32, Device::Host);
        std::memcpy(h.data(), v.data(), v.size() * sizeof(float));
        return h.to(Device::Cuda);
    };
    Tensor g_fp32 = upload(g);
    Tensor u_fp32 = upload(u);
    Tensor o_fp32({N}, DType::FP32, Device::Cuda);
    modernllm::silu_mul_forward(g_fp32, u_fp32, o_fp32);

    Tensor g_bf16 = cast_to_bf16(g_fp32);
    Tensor u_bf16 = cast_to_bf16(u_fp32);
    Tensor o_bf16({N}, DType::BF16, Device::Cuda);
    modernllm::silu_mul_forward(g_bf16, u_bf16, o_bf16);

    Tensor o_back = cast_to_fp32(o_bf16);
    Tensor o_h = o_fp32.to(Device::Host);
    Tensor o_bb = o_back.to(Device::Host);
    float d = max_abs_diff(o_h.data_as<float>(), o_bb.data_as<float>(), N);
    std::printf("    silu_mul fwd BF16 vs FP32 max=%.3e\n", d);
    MLLM_EXPECT(d < 0.05f);
}

MLLM_TEST(test_silu_mul_bwd_bf16_matches_fp32) {
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-1.5f, 1.5f);
    const int N = 256;

    std::vector<float> g(N), u(N), dy(N);
    for (auto& x : g) x = dist(rng);
    for (auto& x : u) x = dist(rng);
    for (auto& x : dy) x = dist(rng);

    auto upload = [&](const std::vector<float>& v) {
        Tensor h({N}, DType::FP32, Device::Host);
        std::memcpy(h.data(), v.data(), v.size() * sizeof(float));
        return h.to(Device::Cuda);
    };
    Tensor g_fp32 = upload(g), u_fp32 = upload(u), dy_fp32 = upload(dy);
    Tensor dg_fp32({N}, DType::FP32, Device::Cuda);
    Tensor du_fp32({N}, DType::FP32, Device::Cuda);
    modernllm::silu_mul_backward(g_fp32, u_fp32, dy_fp32, dg_fp32, du_fp32);

    Tensor g_bf = cast_to_bf16(g_fp32);
    Tensor u_bf = cast_to_bf16(u_fp32);
    Tensor dy_bf = cast_to_bf16(dy_fp32);
    Tensor dg_bf({N}, DType::BF16, Device::Cuda);
    Tensor du_bf({N}, DType::BF16, Device::Cuda);
    modernllm::silu_mul_backward(g_bf, u_bf, dy_bf, dg_bf, du_bf);

    Tensor dg_back = cast_to_fp32(dg_bf);
    Tensor du_back = cast_to_fp32(du_bf);
    Tensor dgh = dg_fp32.to(Device::Host);
    Tensor dgb = dg_back.to(Device::Host);
    Tensor duh = du_fp32.to(Device::Host);
    Tensor dub = du_back.to(Device::Host);
    float dg_d = max_abs_diff(dgh.data_as<float>(), dgb.data_as<float>(), N);
    float du_d = max_abs_diff(duh.data_as<float>(), dub.data_as<float>(), N);
    std::printf("    silu_mul bwd BF16 vs FP32 dgate=%.3e dup=%.3e\n",
                 dg_d, du_d);
    MLLM_EXPECT(dg_d < 0.05f);
    MLLM_EXPECT(du_d < 0.05f);
}

MLLM_TEST(test_rmsnorm_bf16_matches_fp32) {
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    const int N = 8, D = 32;

    std::vector<float> x(N * D), gamma(D);
    for (auto& v : x) v = dist(rng);
    for (auto& v : gamma) v = dist(rng) * 0.5f + 1.f;

    auto upload = [&](const std::vector<float>& v,
                       std::initializer_list<std::int64_t> shape) {
        Tensor h(shape, DType::FP32, Device::Host);
        std::memcpy(h.data(), v.data(), v.size() * sizeof(float));
        return h.to(Device::Cuda);
    };
    Tensor x_fp32 = upload(x, {N, D});
    Tensor g_fp32 = upload(gamma, {D});
    Tensor y_fp32({N, D}, DType::FP32, Device::Cuda);
    Tensor rstd_fp32({N}, DType::FP32, Device::Cuda);
    modernllm::rmsnorm_forward(x_fp32, g_fp32, 1e-5f, y_fp32, rstd_fp32);

    Tensor x_bf16 = cast_to_bf16(x_fp32);
    Tensor y_bf16({N, D}, DType::BF16, Device::Cuda);
    Tensor rstd_bf16({N}, DType::FP32, Device::Cuda);
    modernllm::rmsnorm_forward(x_bf16, g_fp32, 1e-5f, y_bf16, rstd_bf16);

    Tensor y_back = cast_to_fp32(y_bf16);
    Tensor yh = y_fp32.to(Device::Host);
    Tensor yb = y_back.to(Device::Host);
    float d = max_abs_diff(yh.data_as<float>(), yb.data_as<float>(), N * D);
    std::printf("    rmsnorm fwd BF16 vs FP32 max=%.3e\n", d);
    MLLM_EXPECT(d < 0.05f);
}

MLLM_TEST(test_rope_bf16_matches_fp32) {
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    const int Nb = 4, T = 8, d_h = 16;

    std::vector<float> x(Nb * T * d_h);
    for (auto& v : x) v = dist(rng);

    Tensor x_h({Nb, T, d_h}, DType::FP32, Device::Host);
    std::memcpy(x_h.data(), x.data(), x.size() * sizeof(float));

    auto cs = modernllm::make_rope_cache(T, d_h, 10000.f, Device::Cuda);

    Tensor x_fp32 = x_h.to(Device::Cuda);
    modernllm::rope_apply_inplace(x_fp32, cs.first, cs.second, Nb, T, d_h);

    Tensor x_b_orig = x_h.to(Device::Cuda);
    Tensor x_bf16 = cast_to_bf16(x_b_orig);
    modernllm::rope_apply_inplace(x_bf16, cs.first, cs.second, Nb, T, d_h);

    Tensor x_back = cast_to_fp32(x_bf16);
    Tensor xh = x_fp32.to(Device::Host);
    Tensor xb = x_back.to(Device::Host);
    float d = max_abs_diff(xh.data_as<float>(), xb.data_as<float>(),
                            Nb * T * d_h);
    std::printf("    rope fwd BF16 vs FP32 max=%.3e\n", d);
    MLLM_EXPECT(d < 0.05f);
}

int main() {
    MLLM_RUN_TEST(test_add_inplace_bf16_matches_fp32);
    MLLM_RUN_TEST(test_silu_mul_bf16_matches_fp32);
    MLLM_RUN_TEST(test_silu_mul_bwd_bf16_matches_fp32);
    MLLM_RUN_TEST(test_rmsnorm_bf16_matches_fp32);
    MLLM_RUN_TEST(test_rope_bf16_matches_fp32);
    std::printf("\nAll BF16 kernel tests passed.\n");
    return 0;
}
