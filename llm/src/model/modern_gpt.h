#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/gemm.h"
#include "core/scratch.h"
#include "core/tensor.h"
#include "train/adamw.h"

namespace modernllm {

// Single LLaMA-style pre-norm transformer block: RMSNorm → MHA(+RoPE) →
// residual → RMSNorm → SwiGLU FFN → residual. All linear layers run without
// bias (LLaMA convention).
struct ModernBlock {
    // Parameters. Wk and Wv are sized for n_kv_heads (smaller than Wq, Wo).
    Tensor ln1_g, Wq, Wk, Wv, Wo, ln2_g, Wgate, Wup, Wdown;
    // QK-Norm: per-head RMSNorm gamma applied after RoPE on Q and K
    // (Olmo-2 / DeepSeek style). Shape [d_h].
    Tensor q_norm_g, k_norm_g;
    // Gradients
    Tensor d_ln1_g, d_Wq, d_Wk, d_Wv, d_Wo, d_ln2_g, d_Wgate, d_Wup, d_Wdown;
    Tensor d_q_norm_g, d_k_norm_g;
    // BF16 mirrors of the linear weights (only allocated if use_bf16_).
    // Refreshed once per opt.step() instead of cast every forward.
    Tensor Wq_bf16, Wk_bf16, Wv_bf16, Wo_bf16;
    Tensor Wgate_bf16, Wup_bf16, Wdown_bf16;
    // Forward cache (saved for backward)
    Tensor inp;
    Tensor ln1_out, ln1_rstd;
    Tensor q_proj, k_proj, v_proj;       // q: [N, n_q*d_h], k/v: [N, n_kv*d_h]
    Tensor q_split, k_split, v_split;     // q: [B*n_q, T, d_h], k/v: [B*n_kv, T, d_h]
    Tensor q_normed, k_normed;            // post-QK-Norm copies (same shapes)
    Tensor q_norm_rstd, k_norm_rstd;      // 1D per-row rstd from RMSNorm
    Tensor k_full, v_full;                // GQA broadcast: [B*n_q, T, d_h]
    Tensor ctx_split;
    Tensor probs;
    Tensor ctx;
    Tensor attn_out;
    Tensor h_residual1;
    Tensor ln2_out, ln2_rstd;
    Tensor ffn_gate, ffn_up;
    Tensor ffn_silu_up;
    Tensor ffn_out;
    Tensor h_block_out;
};

// One named parameter (with its gradient) for checkpointing & optimizer.
struct NamedParam {
    std::string name;
    Tensor* param;
    Tensor* grad;
};

class ModernGPT {
   public:
    struct Config {
        int B = 0;
        int T = 0;
        int V = 0;
        int D = 0;          // d_model
        int H = 0;          // n_q_heads
        int n_kv_heads = 0; // GQA: n_kv heads (default = H = MHA)
        int Dff = 0;        // d_ffn
        int n_layers = 0;
        float rope_base = 10000.f;
    };

    void allocate(Config cfg);
    void init_random(unsigned long long seed);

    // Pick FP32 (default) vs BF16 mixed-precision matmul. Must be called
    // BEFORE allocate() so the BF16 mirrors and scratch arena get sized.
    void set_use_bf16(bool b) noexcept { use_bf16_ = b; }
    bool use_bf16() const noexcept { return use_bf16_; }

    // Refresh BF16 mirrors from FP32 masters. Call once per training step,
    // after `opt.step()`, when in BF16 mode.
    void refresh_bf16_mirrors();

    // T7 helper: re-run the forward of block L from `blocks_[L].inp` into
    // the shared activation buffers. Used inside backward to repopulate the
    // intermediate activations of an earlier layer that were overwritten by
    // the forward of a later layer.
    void recompute_block_forward(int L, cublasHandle_t handle);

    // Returns ordered list of (name, param, grad) for every learnable tensor.
    std::vector<NamedParam> named_params();

    // Wires every parameter into the optimizer (linear weights get `wd`,
    // norms / embeddings get the value passed for them).
    void register_with_optimizer(AdamW& opt, float wd);

    // Returns raw grad pointers (for clip_grad_norm).
    std::vector<Tensor*> all_grads();

    // Forward pass. If target_ids is non-null, computes loss + dlogits.
    // Returns scalar mean loss when target_ids is provided, else 0.0f.
    float forward(const Tensor& input_ids, const Tensor* target_ids,
                  cublasHandle_t handle);

    // Backward pass. Must be called only if forward computed loss.
    void backward(const Tensor& input_ids, cublasHandle_t handle);

    // Logits cache (filled by forward), shape [B*T, V].
    const Tensor& logits() const noexcept { return logits_; }

    // Config / dims
    const Config& config() const noexcept { return cfg_; }
    int B() const noexcept { return cfg_.B; }
    int T() const noexcept { return cfg_.T; }
    int V() const noexcept { return cfg_.V; }
    int D() const noexcept { return cfg_.D; }
    int H() const noexcept { return cfg_.H; }
    int d_h() const noexcept { return d_h_; }
    int n_layers() const noexcept { return cfg_.n_layers; }

    // Total trainable parameter count (sum of numel across all tensors
    // returned by named_params()).
    long long parameter_count() const;

   private:
    Config cfg_{};
    int d_h_ = 0;
    int n_kv_ = 0;          // resolved n_kv_heads (defaults to cfg_.H if 0)
    bool use_bf16_ = false;

    // RoPE cache
    Tensor rope_cos_, rope_sin_;

    // Top-level params + grads. The LM head is TIED to tok_emb (output
    // projection uses tok_emb^T) — no separate lm_head tensor.
    Tensor tok_emb_, lnf_g_;
    Tensor d_tok_emb_, d_lnf_g_;

    // Top-level activations
    Tensor h_input_;
    Tensor lnf_out_, lnf_rstd_;
    Tensor logits_, loss_pr_, dlogits_;

    // Stage T7 — gradient checkpointing: per-block activations are SHARED
    // across all layers. Each ModernBlock's `ln1_out`, `q_proj`, ... are
    // non-owning views into these. Backward must call recompute_block_forward
    // to repopulate them for each layer (except the last, whose values are
    // still live from the final forward pass).
    Tensor act_ln1_out_, act_ln1_rstd_;
    Tensor act_q_proj_, act_k_proj_, act_v_proj_;
    Tensor act_q_split_, act_k_split_, act_v_split_;
    Tensor act_q_normed_, act_k_normed_;
    Tensor act_q_norm_rstd_, act_k_norm_rstd_;
    Tensor act_k_full_, act_v_full_;
    Tensor act_ctx_split_, act_probs_;
    Tensor act_ctx_;
    Tensor act_attn_out_;
    Tensor act_h_residual1_;
    Tensor act_ln2_out_, act_ln2_rstd_;
    Tensor act_ffn_gate_, act_ffn_up_, act_ffn_silu_up_;
    Tensor act_ffn_out_;
    Tensor act_h_block_out_;

    // Backward scratch
    Tensor d_lnf_out_, d_block_out_, d_h_residual1_, d_ffn_out_;
    Tensor d_ln2_out_, d_ffn_silu_up_, d_ffn_gate_, d_ffn_up_;
    Tensor d_attn_out_, d_ctx_;
    Tensor d_q_split_;                    // [B*n_q, T, d_h]
    Tensor d_k_full_, d_v_full_;          // [B*n_q, T, d_h] (post-broadcast grads)
    Tensor d_k_split_, d_v_split_;        // [B*n_kv, T, d_h] (after accumulate)
    Tensor d_q_proj_, d_k_proj_, d_v_proj_, d_ln1_out_, d_branch_;

    // Per-block storage
    std::vector<ModernBlock> blocks_;

    // Scratch arena for BF16 activation casts (only sized if use_bf16_).
    ScratchArena bf16_arena_;
};

}  // namespace modernllm
