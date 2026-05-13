#pragma once

#include <cstdint>
#include <vector>

#include "core/tensor.h"

namespace modernllm {

struct AdamWConfig {
    float lr = 3e-4f;
    float beta1 = 0.9f;
    float beta2 = 0.95f;
    float eps = 1e-8f;
    float weight_decay = 0.1f;
    // T8 — store first/second moments as BF16 instead of FP32 (2× smaller).
    // Computation in the step kernel is still FP32; the cast on read/write is
    // ~free vs the math. BF16 quantization adds a tiny convergence noise but
    // it's well below the AdamW tolerances for normal training.
    bool bf16_states = false;
};

// One AdamW parameter slot: parameter, gradient, and per-element m/v state.
// All four tensors must be FP32, same shape, on the same CUDA device.
struct AdamWParam {
    Tensor* param;  // updated in place
    Tensor* grad;   // read, NOT zeroed by step() — caller decides when to zero
    Tensor m;       // first moment, owned by optimizer
    Tensor v;       // second moment, owned by optimizer
    float weight_decay = 0.1f;  // can be overridden per-param (e.g., 0 for biases)
};

class AdamW {
   public:
    explicit AdamW(AdamWConfig cfg);

    // Register a parameter / gradient pair. The optimizer allocates m, v.
    // `param` and `grad` must outlive the optimizer.
    void add_param(Tensor* param, Tensor* grad, float weight_decay = -1.f);

    // One optimizer step: applies the update to every registered param.
    // step_idx is the 1-based step counter (used for bias correction).
    void step();

    // Set all gradient tensors to zero.
    void zero_grad();

    int step_count() const noexcept { return step_; }
    void set_step_count(int s) noexcept { step_ = s; }
    const AdamWConfig& config() const noexcept { return cfg_; }
    void set_lr(float lr) noexcept { cfg_.lr = lr; }

    // Direct access for checkpoint serialization. Each slot owns its m, v.
    std::vector<AdamWParam>& params() noexcept { return params_; }
    const std::vector<AdamWParam>& params() const noexcept { return params_; }

   private:
    AdamWConfig cfg_;
    std::vector<AdamWParam> params_;
    int step_ = 0;
};

}  // namespace modernllm
