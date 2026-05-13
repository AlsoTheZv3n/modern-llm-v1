#pragma once

#include <cmath>
#include <vector>

#include "core/tensor.h"

namespace modernllm {

// Compute the global L2 norm sqrt(sum_i ||grad_i||^2).
// Synchronous: returns a host float.
float compute_grad_norm(const std::vector<Tensor*>& grads);

// Scale every tensor in `grads` in place by `scale`.
void scale_grads(std::vector<Tensor*>& grads, float scale);

// Clip the global gradient norm to `max_norm`. Returns the *pre-clip* norm.
// If the norm was below `max_norm`, no scaling is applied.
float clip_grad_norm(std::vector<Tensor*>& grads, float max_norm);

// Cosine learning-rate schedule with linear warmup.
//   step in [1, total_steps]
//   step <= warmup_steps        -> linear warmup from 0 to max_lr
//   step >  warmup_steps        -> cosine decay from max_lr to min_lr
inline float cosine_lr_with_warmup(int step, int warmup_steps,
                                    int total_steps, float max_lr,
                                    float min_lr) {
    if (step <= warmup_steps) {
        return max_lr * static_cast<float>(step) /
               static_cast<float>(warmup_steps);
    }
    float t = static_cast<float>(step - warmup_steps) /
              static_cast<float>(total_steps - warmup_steps);
    if (t > 1.f) t = 1.f;
    constexpr float kPi = 3.14159265358979323846f;
    return min_lr + 0.5f * (max_lr - min_lr) * (1.f + std::cos(kPi * t));
}

}  // namespace modernllm
