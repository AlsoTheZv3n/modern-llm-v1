#pragma once

#include "core/tensor.h"

namespace modernllm {

// Combined softmax + cross-entropy loss with mean reduction.
//
// Inputs:
//   logits   [B, V] FP32 device tensor
//   targets  [B]    INT32 device tensor (values in [0, V))
//
// Outputs:
//   loss_per_row [B]    FP32 device tensor — caller averages for scalar loss
//   dlogits      [B, V] FP32 device tensor — already scaled by 1/B (mean grad)
//
// Equivalent (in PyTorch):
//   probs   = softmax(logits, dim=-1)
//   loss    = mean(-log(probs[range(B), targets]))
//   dlogits = (probs - one_hot(targets)) / B
void softmax_ce_forward_backward(const Tensor& logits,
                                  const Tensor& targets,
                                  Tensor& loss_per_row,
                                  Tensor& dlogits);

// Convenience: compute the scalar mean loss on host from loss_per_row.
float reduce_mean_to_scalar(const Tensor& loss_per_row);

}  // namespace modernllm
