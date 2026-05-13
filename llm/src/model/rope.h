#pragma once

#include <utility>

#include "core/tensor.h"

namespace modernllm {

// Precompute cos/sin tables for RoPE.
//   Returns (cos, sin), each shape [T_max, d_h / 2], on the requested device.
//
// Convention: pair (2i, 2i+1) of the head dim is rotated by angle
//   theta_i(t) = t * base^(-2i / d_h)
std::pair<Tensor, Tensor> make_rope_cache(int T_max, int d_h, float base,
                                            Device device);

// In-place rotary embedding on a tensor shaped [N, T, d_h].
//   x[n, t, 2i]   <- x[n, t, 2i] * cos - x[n, t, 2i+1] * sin
//   x[n, t, 2i+1] <- x[n, t, 2i] * sin + x[n, t, 2i+1] * cos
//
// d_h must be even. Caller selects the (T x d_h/2) slice of cos/sin tables.
void rope_apply_inplace(Tensor& x, const Tensor& cos, const Tensor& sin,
                         int N, int T, int d_h);

// Backward through RoPE (rotate by -theta = same matrix transposed).
//   dx[n, t, 2i]   <- dx[n, t, 2i] * cos + dx[n, t, 2i+1] * sin
//   dx[n, t, 2i+1] <- -dx[n, t, 2i] * sin + dx[n, t, 2i+1] * cos
void rope_apply_backward_inplace(Tensor& dx, const Tensor& cos,
                                   const Tensor& sin,
                                   int N, int T, int d_h);

}  // namespace modernllm
