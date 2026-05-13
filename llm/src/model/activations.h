#pragma once

#include "core/tensor.h"

namespace modernllm {

// GELU activation, tanh-approximation form (matches GPT-2 / nanoGPT).
//
//   y = 0.5 * x * (1 + tanh(c * (x + 0.044715 * x^3)))
//   c = sqrt(2 / pi) ≈ 0.7978845608
//
// Forward and backward are elementwise.
void gelu_forward(const Tensor& x, Tensor& y);
void gelu_backward(const Tensor& x, const Tensor& d_y, Tensor& d_x);

// Elementwise tensor add: c[i] = a[i] + b[i].
void add_inplace(Tensor& a, const Tensor& b);

// Fused SwiGLU gate: out[i] = silu(gate[i]) * up[i]
//   silu(x) = x * sigmoid(x)
void silu_mul_forward(const Tensor& gate, const Tensor& up, Tensor& out);

// Backward for fused silu*up:
//   d_up[i]   = d_out[i] * silu(gate[i])
//   d_gate[i] = d_out[i] * up[i] * silu'(gate[i])
//     where silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
void silu_mul_backward(const Tensor& gate, const Tensor& up,
                        const Tensor& d_out,
                        Tensor& d_gate, Tensor& d_up);

}  // namespace modernllm
