#include "core/random.h"

#include <random>
#include <stdexcept>
#include <vector>

#include "core/cuda_check.h"
#include "core/dtype.h"

namespace modernllm {

namespace {

template <typename Dist>
void fill_with_dist(Tensor& t, Dist& dist, std::mt19937_64& rng) {
    if (t.dtype() != DType::FP32) {
        throw std::invalid_argument(
            "random init: only FP32 supported for now");
    }
    if (t.numel() == 0) return;

    if (t.device() == Device::Host) {
        float* p = t.data_as<float>();
        for (std::int64_t i = 0; i < t.numel(); ++i) p[i] = dist(rng);
        return;
    }

    // Device path: fill on host, copy over.
    std::vector<float> buf(t.numel());
    for (std::int64_t i = 0; i < t.numel(); ++i) buf[i] = dist(rng);
    MLLM_CUDA_CHECK(cudaMemcpy(t.data(), buf.data(), t.bytes(),
                                cudaMemcpyHostToDevice));
}

}  // namespace

void normal_(Tensor& t, float mean, float stddev, unsigned long long seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(mean, stddev);
    fill_with_dist(t, dist, rng);
}

void uniform_(Tensor& t, float lo, float hi, unsigned long long seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    fill_with_dist(t, dist, rng);
}

}  // namespace modernllm
