#pragma once

#include "core/tensor.h"

namespace modernllm {

// Token embedding lookup.
//
// Forward:  out[n, d]  = weight[ids[n], d]
// Backward: d_weight[ids[n], d] += d_out[n, d]   (atomicAdd; resets if zero=true)
//
// Shapes:
//   weight   [V, D]  FP32
//   ids      [N]     INT32  (values in [0, V))
//   out      [N, D]  FP32
//   d_out    [N, D]  FP32
//   d_weight [V, D]  FP32
//
// Caller is responsible for flattening (B, T) → N if needed.
void embedding_forward(const Tensor& weight, const Tensor& ids, Tensor& out);

// Accumulates into d_weight. Caller must zero d_weight first if a fresh
// gradient pass is wanted (training loop typically zeros all grads at top).
void embedding_backward(const Tensor& d_out, const Tensor& ids,
                         Tensor& d_weight);

}  // namespace modernllm
