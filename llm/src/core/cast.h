#pragma once

#include "core/tensor.h"

namespace modernllm {

// Elementwise cast between FP32 and BF16. Both tensors must:
//   - have the same shape / numel
//   - be on CUDA device
//
// Round-to-nearest-even on the FP32 -> BF16 path (matches PyTorch defaults
// and the host-side helper in dtype.h).
void cast_fp32_to_bf16(const Tensor& src_fp32, Tensor& dst_bf16);
void cast_bf16_to_fp32(const Tensor& src_bf16, Tensor& dst_fp32);

}  // namespace modernllm
