// Tests softmax + cross-entropy:
//   1) Forward matches a host reference
//   2) Backward matches finite differences (numerical gradient check)

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "core/tensor.h"
#include "tests/test_util.h"
#include "train/loss.h"

using modernllm::Device;
using modernllm::DType;
using modernllm::Tensor;

namespace {

// Host reference: combined softmax + CE with mean reduction.
// Returns scalar loss; fills d_logits if non-null with (probs - one_hot)/B.
float ce_host(const float* logits, const int* targets,
              int B, int V, float* d_logits = nullptr) {
    double total = 0.0;
    for (int b = 0; b < B; ++b) {
        const float* row = logits + b * V;
        // numerically stable softmax + CE
        float row_max = -INFINITY;
        for (int v = 0; v < V; ++v) row_max = std::fmax(row_max, row[v]);
        double sum = 0.0;
        for (int v = 0; v < V; ++v) sum += std::exp(row[v] - row_max);
        double row_loss = row_max + std::log(sum) - row[targets[b]];
        total += row_loss;

        if (d_logits) {
            float* drow = d_logits + b * V;
            float scale = 1.f / static_cast<float>(B);
            for (int v = 0; v < V; ++v) {
                float prob = static_cast<float>(std::exp(row[v] - row_max) /
                                                 sum);
                float ind = (v == targets[b]) ? 1.f : 0.f;
                drow[v] = (prob - ind) * scale;
            }
        }
    }
    return static_cast<float>(total / B);
}

}  // namespace

MLLM_TEST(test_ce_forward_matches_host) {
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-2.f, 2.f);
    std::uniform_int_distribution<int> tgt_dist;

    const int B = 7;
    const int V = 65;  // not power of two, like char vocab

    Tensor logits_h({B, V}, DType::FP32, Device::Host);
    Tensor targets_h({B}, DType::INT32, Device::Host);

    std::vector<float> logits_data(B * V);
    std::vector<int> tgt_data(B);
    for (auto& x : logits_data) x = dist(rng);
    for (auto& t : tgt_data) {
        t = std::uniform_int_distribution<int>(0, V - 1)(rng);
    }
    auto* lh = logits_h.data_as<float>();
    for (int i = 0; i < B * V; ++i) lh[i] = logits_data[i];
    auto* th = targets_h.data_as<int>();
    for (int i = 0; i < B; ++i) th[i] = tgt_data[i];

    Tensor logits_d = logits_h.to(Device::Cuda);
    Tensor targets_d = targets_h.to(Device::Cuda);

    Tensor loss_pr({B}, DType::FP32, Device::Cuda);
    Tensor dlogits({B, V}, DType::FP32, Device::Cuda);

    modernllm::softmax_ce_forward_backward(logits_d, targets_d, loss_pr,
                                            dlogits);
    float gpu_mean = modernllm::reduce_mean_to_scalar(loss_pr);

    std::vector<float> dlogits_ref(B * V);
    float ref_mean = ce_host(logits_data.data(), tgt_data.data(),
                              B, V, dlogits_ref.data());

    std::printf("    gpu_loss=%.6f ref_loss=%.6f diff=%.3e\n",
                 gpu_mean, ref_mean, std::fabs(gpu_mean - ref_mean));
    MLLM_EXPECT_NEAR(gpu_mean, ref_mean, 1e-4);

    Tensor dlog_h = dlogits.to(Device::Host);
    auto* dh = dlog_h.data_as<float>();
    float max_d = 0.f;
    for (int i = 0; i < B * V; ++i) {
        max_d = std::fmax(max_d, std::fabs(dh[i] - dlogits_ref[i]));
    }
    std::printf("    dlogits max_diff=%.3e\n", max_d);
    MLLM_EXPECT(max_d < 1e-5f);
}

MLLM_TEST(test_ce_gradient_check_finite_diff) {
    // Verify that analytical dlogits matches numerical (logits +/- eps) gradient.
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.5f, 1.5f);

    const int B = 3;
    const int V = 5;
    std::vector<float> logits(B * V);
    std::vector<int> tgt(B);
    for (auto& x : logits) x = dist(rng);
    for (auto& t : tgt) t = std::uniform_int_distribution<int>(0, V - 1)(rng);

    // Analytical via host reference
    std::vector<float> dlog_anal(B * V);
    ce_host(logits.data(), tgt.data(), B, V, dlog_anal.data());

    // Numerical via central differences
    const float eps = 1e-3f;
    float max_d = 0.f;
    for (int i = 0; i < B * V; ++i) {
        float saved = logits[i];
        logits[i] = saved + eps;
        float lp = ce_host(logits.data(), tgt.data(), B, V);
        logits[i] = saved - eps;
        float lm = ce_host(logits.data(), tgt.data(), B, V);
        logits[i] = saved;
        float num = (lp - lm) / (2.f * eps);
        max_d = std::fmax(max_d, std::fabs(num - dlog_anal[i]));
    }
    std::printf("    finite-diff max_diff=%.3e\n", max_d);
    MLLM_EXPECT(max_d < 1e-3f);
}

int main() {
    MLLM_RUN_TEST(test_ce_forward_matches_host);
    MLLM_RUN_TEST(test_ce_gradient_check_finite_diff);
    std::printf("\nAll loss tests passed.\n");
    return 0;
}
