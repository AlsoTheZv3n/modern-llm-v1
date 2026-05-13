#include "train/adamw.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "core/cuda_check.h"

namespace modernllm {

namespace {

__global__ void adamw_step_kernel(float* __restrict__ param,
                                  const float* __restrict__ grad,
                                  float* __restrict__ m,
                                  float* __restrict__ v,
                                  std::int64_t numel,
                                  float lr, float beta1, float beta2,
                                  float eps, float weight_decay,
                                  float bc1, float bc2) {
    std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (idx >= numel) return;

    float g = grad[idx];
    float p = param[idx];
    float mi = beta1 * m[idx] + (1.f - beta1) * g;
    float vi = beta2 * v[idx] + (1.f - beta2) * g * g;
    m[idx] = mi;
    v[idx] = vi;
    float m_hat = mi / bc1;
    float v_hat = vi / bc2;
    p -= lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * p);
    param[idx] = p;
}

// BF16-state variant: m and v are stored as 2-byte BF16. Compute in FP32,
// cast on read/write. Identical math to the FP32 kernel modulo BF16
// quantization in the m/v stored values.
__device__ __forceinline__ float bf16_to_f32_dev(unsigned short b) {
    unsigned int bits = static_cast<unsigned int>(b) << 16;
    return __uint_as_float(bits);
}
__device__ __forceinline__ unsigned short f32_to_bf16_rne(float f) {
    unsigned int bits = __float_as_uint(f);
    unsigned int lsb = (bits >> 16) & 1u;
    unsigned int rounding_bias = 0x7FFFu + lsb;
    bits += rounding_bias;
    return static_cast<unsigned short>(bits >> 16);
}

__global__ void adamw_step_kernel_bf16(float* __restrict__ param,
                                        const float* __restrict__ grad,
                                        unsigned short* __restrict__ m,
                                        unsigned short* __restrict__ v,
                                        std::int64_t numel,
                                        float lr, float beta1, float beta2,
                                        float eps, float weight_decay,
                                        float bc1, float bc2) {
    std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (idx >= numel) return;

    float g = grad[idx];
    float p = param[idx];
    float m_old = bf16_to_f32_dev(m[idx]);
    float v_old = bf16_to_f32_dev(v[idx]);
    float mi = beta1 * m_old + (1.f - beta1) * g;
    float vi = beta2 * v_old + (1.f - beta2) * g * g;
    m[idx] = f32_to_bf16_rne(mi);
    v[idx] = f32_to_bf16_rne(vi);
    float m_hat = mi / bc1;
    float v_hat = vi / bc2;
    p -= lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * p);
    param[idx] = p;
}

}  // namespace

AdamW::AdamW(AdamWConfig cfg) : cfg_(cfg) {}

void AdamW::add_param(Tensor* param, Tensor* grad, float weight_decay) {
    if (!param || !grad) {
        throw std::invalid_argument("AdamW::add_param: null tensor");
    }
    if (param->numel() != grad->numel() ||
        param->shape() != grad->shape()) {
        throw std::invalid_argument("AdamW::add_param: shape mismatch");
    }
    if (param->dtype() != DType::FP32 || grad->dtype() != DType::FP32) {
        throw std::invalid_argument("AdamW::add_param: only FP32 supported");
    }
    if (param->device() != grad->device()) {
        throw std::invalid_argument("AdamW::add_param: device mismatch");
    }

    AdamWParam slot;
    slot.param = param;
    slot.grad = grad;
    DType state_dtype = cfg_.bf16_states ? DType::BF16 : DType::FP32;
    slot.m = Tensor::zeros(param->shape(), state_dtype, param->device());
    slot.v = Tensor::zeros(param->shape(), state_dtype, param->device());
    slot.weight_decay = (weight_decay < 0.f) ? cfg_.weight_decay
                                              : weight_decay;
    params_.push_back(std::move(slot));
}

void AdamW::step() {
    ++step_;
    float bc1 = 1.f - std::pow(cfg_.beta1, static_cast<float>(step_));
    float bc2 = 1.f - std::pow(cfg_.beta2, static_cast<float>(step_));

    const int block = 256;
    for (auto& slot : params_) {
        std::int64_t n = slot.param->numel();
        if (n == 0) continue;
        unsigned grid = static_cast<unsigned>((n + block - 1) / block);

        if (slot.param->device() == Device::Cuda) {
            if (slot.m.dtype() == DType::FP32) {
                adamw_step_kernel<<<grid, block>>>(
                    slot.param->data_as<float>(),
                    slot.grad->data_as<float>(),
                    slot.m.data_as<float>(),
                    slot.v.data_as<float>(),
                    n, cfg_.lr, cfg_.beta1, cfg_.beta2,
                    cfg_.eps, slot.weight_decay, bc1, bc2);
            } else {
                adamw_step_kernel_bf16<<<grid, block>>>(
                    slot.param->data_as<float>(),
                    slot.grad->data_as<float>(),
                    static_cast<unsigned short*>(slot.m.data()),
                    static_cast<unsigned short*>(slot.v.data()),
                    n, cfg_.lr, cfg_.beta1, cfg_.beta2,
                    cfg_.eps, slot.weight_decay, bc1, bc2);
            }
            MLLM_CUDA_CHECK(cudaGetLastError());
        } else {
            // Host fallback (mostly for testing convenience).
            float* p = slot.param->data_as<float>();
            const float* g = slot.grad->data_as<float>();
            float* m = slot.m.data_as<float>();
            float* v = slot.v.data_as<float>();
            for (std::int64_t i = 0; i < n; ++i) {
                float gi = g[i];
                m[i] = cfg_.beta1 * m[i] + (1.f - cfg_.beta1) * gi;
                v[i] = cfg_.beta2 * v[i] + (1.f - cfg_.beta2) * gi * gi;
                float m_hat = m[i] / bc1;
                float v_hat = v[i] / bc2;
                p[i] -= cfg_.lr * (m_hat / (std::sqrt(v_hat) + cfg_.eps) +
                                    slot.weight_decay * p[i]);
            }
        }
    }
}

void AdamW::zero_grad() {
    for (auto& slot : params_) {
        slot.grad->zero();
    }
}

}  // namespace modernllm
