// Sanity test: AdamW should drive a 2-D quadratic to its minimum.
// f(x) = 0.5 * ||x - target||^2  ⇒ grad(x) = x - target.
//
// We run AdamW for 2000 steps and check that x converges to target.

#include <cmath>
#include <cstdio>
#include <vector>

#include "core/tensor.h"
#include "tests/test_util.h"
#include "train/adamw.h"

using modernllm::AdamW;
using modernllm::AdamWConfig;
using modernllm::Device;
using modernllm::DType;
using modernllm::Tensor;

namespace {

// Compute grad = param - target  (host side, then copy to device grad tensor).
void compute_grad(const Tensor& param_d, const std::vector<float>& target,
                   Tensor& grad_d) {
    Tensor p_h = param_d.to(Device::Host);
    Tensor g_h(param_d.shape(), DType::FP32, Device::Host);
    auto* p = p_h.data_as<float>();
    auto* g = g_h.data_as<float>();
    for (std::int64_t i = 0; i < p_h.numel(); ++i) g[i] = p[i] - target[i];
    grad_d.copy_from(g_h);
}

}  // namespace

MLLM_TEST(test_adamw_converges_quadratic) {
    const int n = 8;
    const std::vector<float> target = {1.0f, -2.0f, 3.5f, 0.5f,
                                        -1.5f, 4.0f, 0.0f, 2.25f};

    Tensor param({n}, DType::FP32, Device::Cuda);
    Tensor grad({n}, DType::FP32, Device::Cuda);
    param.zero();  // start at origin
    grad.zero();

    AdamWConfig cfg;
    cfg.lr = 0.05f;          // larger LR than typical, since target is fixed
    cfg.weight_decay = 0.0f; // pure quadratic, no L2 pull-to-zero
    AdamW opt(cfg);
    opt.add_param(&param, &grad, /*weight_decay=*/0.0f);

    for (int step = 0; step < 2000; ++step) {
        compute_grad(param, target, grad);
        opt.step();
    }

    Tensor out_h = param.to(Device::Host);
    auto* p = out_h.data_as<float>();
    float max_err = 0.f;
    for (int i = 0; i < n; ++i) {
        max_err = std::fmax(max_err, std::fabs(p[i] - target[i]));
    }
    std::printf("    final max_err=%.3e (target tol=5e-3)\n", max_err);
    MLLM_EXPECT(max_err < 5e-3f);
}

MLLM_TEST(test_adamw_weight_decay_pulls_to_zero) {
    const int n = 4;
    Tensor param({n}, DType::FP32, Device::Cuda);
    Tensor grad({n}, DType::FP32, Device::Cuda);

    // Initialize param to ones, gradient stays zero.
    Tensor ones({n}, DType::FP32, Device::Host);
    ones.fill(1.0f);
    param.copy_from(ones);
    grad.zero();

    AdamWConfig cfg;
    cfg.lr = 0.01f;
    cfg.weight_decay = 0.1f;
    AdamW opt(cfg);
    opt.add_param(&param, &grad);

    for (int step = 0; step < 500; ++step) {
        opt.step();
    }

    Tensor out_h = param.to(Device::Host);
    auto* p = out_h.data_as<float>();
    std::printf("    after 500 steps, p[0] = %.4f (started at 1.0)\n", p[0]);
    // With wd=0.1 and lr=0.01, decay factor per step ≈ (1 - 0.001) = 0.999,
    // so 500 steps ≈ 0.999^500 ≈ 0.606. Just check we moved meaningfully.
    MLLM_EXPECT(p[0] < 0.9f);
    MLLM_EXPECT(p[0] > 0.0f);
}

// Same convergence test but with BF16 states. BF16 has 7 mantissa bits, so
// stored m/v values are quantized — convergence is slightly noisier than FP32
// but still reliably reaches the optimum.
MLLM_TEST(test_adamw_converges_quadratic_bf16_states) {
    const int n = 8;
    const std::vector<float> target = {1.0f, -2.0f, 3.5f, 0.5f,
                                        -1.5f, 4.0f, 0.0f, 2.25f};

    Tensor param({n}, DType::FP32, Device::Cuda);
    Tensor grad({n}, DType::FP32, Device::Cuda);
    param.zero();
    grad.zero();

    AdamWConfig cfg;
    cfg.lr = 0.05f;
    cfg.weight_decay = 0.0f;
    cfg.bf16_states = true;
    AdamW opt(cfg);
    opt.add_param(&param, &grad, /*weight_decay=*/0.0f);

    for (int step = 0; step < 2000; ++step) {
        compute_grad(param, target, grad);
        opt.step();
    }

    Tensor out_h = param.to(Device::Host);
    auto* p = out_h.data_as<float>();
    float max_err = 0.f;
    for (int i = 0; i < n; ++i) {
        max_err = std::fmax(max_err, std::fabs(p[i] - target[i]));
    }
    std::printf("    BF16 states final max_err=%.3e (tol 2e-2)\n", max_err);
    // Looser tolerance than FP32 because BF16 m/v noise accumulates.
    MLLM_EXPECT(max_err < 2e-2f);
}

int main() {
    MLLM_RUN_TEST(test_adamw_converges_quadratic);
    MLLM_RUN_TEST(test_adamw_weight_decay_pulls_to_zero);
    MLLM_RUN_TEST(test_adamw_converges_quadratic_bf16_states);
    std::printf("\nAll AdamW tests passed.\n");
    return 0;
}
