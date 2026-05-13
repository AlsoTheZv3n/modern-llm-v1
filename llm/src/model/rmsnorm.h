#pragma once

#include "core/tensor.h"

namespace modernllm {

// RMSNorm over the last dim (no mean centering, no beta).
//   rms[n]  = sqrt( (1/D) * sum_d x[n, d]^2 + eps )
//   rstd[n] = 1 / rms[n]
//   y[n, d] = x[n, d] * rstd[n] * gamma[d]
//
// Shapes:  x, y [N, D] FP32 CUDA;  gamma [D];  rstd [N]
void rmsnorm_forward(const Tensor& x, const Tensor& gamma, float eps,
                      Tensor& y, Tensor& rstd);

// d_gamma is accumulated into. d_x is overwritten.
void rmsnorm_backward(const Tensor& d_y, const Tensor& x, const Tensor& gamma,
                       const Tensor& rstd,
                       Tensor& d_x, Tensor& d_gamma);

}  // namespace modernllm
