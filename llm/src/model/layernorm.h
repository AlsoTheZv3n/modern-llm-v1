#pragma once

#include "core/tensor.h"

namespace modernllm {

// LayerNorm over the last dimension.
//
// Forward:
//   mean[n]  = (1/D) sum_d x[n, d]
//   var[n]   = (1/D) sum_d (x[n, d] - mean[n])^2
//   rstd[n]  = 1 / sqrt(var[n] + eps)
//   y[n, d]  = ((x[n, d] - mean[n]) * rstd[n]) * gamma[d] + beta[d]
//
// Shapes (FP32 / CUDA):
//   x      [N, D]
//   gamma  [D]
//   beta   [D]
//   y      [N, D]
//   rstd   [N]    — saved for backward
//   mean   [N]    — saved for backward
void layernorm_forward(const Tensor& x, const Tensor& gamma,
                        const Tensor& beta, float eps,
                        Tensor& y, Tensor& mean, Tensor& rstd);

// d_gamma and d_beta are accumulated into. d_x is overwritten.
void layernorm_backward(const Tensor& d_y, const Tensor& x,
                         const Tensor& gamma,
                         const Tensor& mean, const Tensor& rstd,
                         Tensor& d_x, Tensor& d_gamma, Tensor& d_beta);

}  // namespace modernllm
