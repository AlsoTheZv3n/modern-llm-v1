#pragma once

#include "core/tensor.h"

namespace modernllm {

// Fill an FP32 tensor with normal(mean, stddev) using std::mt19937.
// Works for Host and Cuda tensors (Cuda path goes via host->device copy).
void normal_(Tensor& t, float mean, float stddev, unsigned long long seed);

// Fill with uniform(lo, hi).
void uniform_(Tensor& t, float lo, float hi, unsigned long long seed);

}  // namespace modernllm
