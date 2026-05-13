#include "model/modern_gpt.h"

#include <stdexcept>
#include <utility>

#include "core/cast.h"
#include "core/cuda_check.h"
#include "core/random.h"
#include "model/activations.h"
#include "model/attention.h"
#include "model/embedding.h"
#include "model/linear.h"
#include "model/rmsnorm.h"
#include "model/rope.h"
#include "train/loss.h"

namespace modernllm {

namespace {

void alloc(Tensor& t, std::vector<std::int64_t> shape) {
    t = Tensor(std::move(shape), DType::FP32, Device::Cuda);
}
void zalloc(Tensor& t, std::vector<std::int64_t> shape) {
    t = Tensor::zeros(std::move(shape), DType::FP32, Device::Cuda);
}

}  // namespace

void ModernGPT::allocate(Config cfg) {
    cfg_ = cfg;
    if (cfg_.D % cfg_.H != 0) throw std::runtime_error("D must divide H");
    d_h_ = cfg_.D / cfg_.H;
    // GQA: n_kv_heads defaults to n_q_heads when not set (== plain MHA).
    n_kv_ = (cfg_.n_kv_heads <= 0) ? cfg_.H : cfg_.n_kv_heads;
    if (cfg_.H % n_kv_ != 0) {
        throw std::runtime_error("n_heads must be a multiple of n_kv_heads");
    }

    const int N = cfg_.B * cfg_.T;
    const int BH_q = cfg_.B * cfg_.H;       // for queries / merged ctx
    const int BH_kv = cfg_.B * n_kv_;        // for keys / values
    const int D = cfg_.D, V = cfg_.V, Dff = cfg_.Dff, H = cfg_.H;
    const int Dkv = n_kv_ * d_h_;            // K/V projection dim

    // RoPE cache (host-precomputed, on device)
    auto cs = make_rope_cache(cfg_.T, d_h_, cfg_.rope_base, Device::Cuda);
    rope_cos_ = std::move(cs.first);
    rope_sin_ = std::move(cs.second);

    // Top-level params + grads. lm_head is tied to tok_emb (uses tok_emb^T).
    alloc(tok_emb_, {V, D});      zalloc(d_tok_emb_, {V, D});
    alloc(lnf_g_, {D});           zalloc(d_lnf_g_, {D});

    // Top-level activations
    alloc(h_input_, {N, D});
    alloc(lnf_out_, {N, D});
    alloc(lnf_rstd_, {N});
    alloc(logits_, {N, V});
    alloc(loss_pr_, {N});
    alloc(dlogits_, {N, V});

    // Backward scratch
    alloc(d_lnf_out_, {N, D});
    alloc(d_block_out_, {N, D});
    alloc(d_h_residual1_, {N, D});
    alloc(d_ffn_out_, {N, D});
    alloc(d_ln2_out_, {N, D});
    alloc(d_ffn_silu_up_, {N, Dff});
    alloc(d_ffn_gate_, {N, Dff});
    alloc(d_ffn_up_, {N, Dff});
    alloc(d_attn_out_, {N, D});
    alloc(d_ctx_, {cfg_.B, cfg_.T, D});
    alloc(d_q_split_, {BH_q, cfg_.T, d_h_});
    alloc(d_k_full_, {BH_q, cfg_.T, d_h_});
    alloc(d_v_full_, {BH_q, cfg_.T, d_h_});
    alloc(d_k_split_, {BH_kv, cfg_.T, d_h_});
    alloc(d_v_split_, {BH_kv, cfg_.T, d_h_});
    alloc(d_q_proj_, {N, D});
    alloc(d_k_proj_, {N, Dkv});
    alloc(d_v_proj_, {N, Dkv});
    alloc(d_ln1_out_, {N, D});
    alloc(d_branch_, {N, D});

    // T7 — gradient checkpointing: allocate the per-block activation buffers
    // ONCE at the top level. Each block's bl.<act> is a non-owning view into
    // these. After forward, only the last block's values remain; backward
    // recomputes earlier blocks' forwards into the same buffers.
    alloc(act_ln1_out_, {N, D});
    alloc(act_ln1_rstd_, {N});
    alloc(act_q_proj_, {N, D});
    alloc(act_k_proj_, {N, Dkv});
    alloc(act_v_proj_, {N, Dkv});
    alloc(act_q_split_, {BH_q, cfg_.T, d_h_});
    alloc(act_k_split_, {BH_kv, cfg_.T, d_h_});
    alloc(act_v_split_, {BH_kv, cfg_.T, d_h_});
    alloc(act_q_normed_, {BH_q, cfg_.T, d_h_});
    alloc(act_k_normed_, {BH_kv, cfg_.T, d_h_});
    alloc(act_q_norm_rstd_, {BH_q * cfg_.T});
    alloc(act_k_norm_rstd_, {BH_kv * cfg_.T});
    alloc(act_k_full_, {BH_q, cfg_.T, d_h_});
    alloc(act_v_full_, {BH_q, cfg_.T, d_h_});
    alloc(act_ctx_split_, {BH_q, cfg_.T, d_h_});
    alloc(act_probs_, {BH_q, cfg_.T, cfg_.T});
    alloc(act_ctx_, {cfg_.B, cfg_.T, D});
    alloc(act_attn_out_, {N, D});
    alloc(act_h_residual1_, {N, D});
    alloc(act_ln2_out_, {N, D});
    alloc(act_ln2_rstd_, {N});
    alloc(act_ffn_gate_, {N, Dff});
    alloc(act_ffn_up_, {N, Dff});
    alloc(act_ffn_silu_up_, {N, Dff});
    alloc(act_ffn_out_, {N, D});
    alloc(act_h_block_out_, {N, D});

    // Blocks
    blocks_.resize(cfg_.n_layers);
    for (auto& bl : blocks_) {
        alloc(bl.ln1_g, {D});         zalloc(bl.d_ln1_g, {D});
        alloc(bl.Wq, {D, D});         zalloc(bl.d_Wq, {D, D});
        alloc(bl.Wk, {D, Dkv});       zalloc(bl.d_Wk, {D, Dkv});
        alloc(bl.Wv, {D, Dkv});       zalloc(bl.d_Wv, {D, Dkv});
        alloc(bl.Wo, {D, D});         zalloc(bl.d_Wo, {D, D});
        alloc(bl.q_norm_g, {d_h_});   zalloc(bl.d_q_norm_g, {d_h_});
        alloc(bl.k_norm_g, {d_h_});   zalloc(bl.d_k_norm_g, {d_h_});
        alloc(bl.ln2_g, {D});         zalloc(bl.d_ln2_g, {D});
        alloc(bl.Wgate, {D, Dff});    zalloc(bl.d_Wgate, {D, Dff});
        alloc(bl.Wup, {D, Dff});      zalloc(bl.d_Wup, {D, Dff});
        alloc(bl.Wdown, {Dff, D});    zalloc(bl.d_Wdown, {Dff, D});

        alloc(bl.inp, {N, D});  // per-block input — kept for grad-ckpt recompute

        // Views into the shared activation buffers.
        bl.ln1_out      = Tensor::from_blob(act_ln1_out_.data(),
                                              act_ln1_out_.shape(),
                                              DType::FP32, Device::Cuda);
        bl.ln1_rstd     = Tensor::from_blob(act_ln1_rstd_.data(),
                                              act_ln1_rstd_.shape(),
                                              DType::FP32, Device::Cuda);
        bl.q_proj       = Tensor::from_blob(act_q_proj_.data(),
                                              act_q_proj_.shape(),
                                              DType::FP32, Device::Cuda);
        bl.k_proj       = Tensor::from_blob(act_k_proj_.data(),
                                              act_k_proj_.shape(),
                                              DType::FP32, Device::Cuda);
        bl.v_proj       = Tensor::from_blob(act_v_proj_.data(),
                                              act_v_proj_.shape(),
                                              DType::FP32, Device::Cuda);
        bl.q_split      = Tensor::from_blob(act_q_split_.data(),
                                              act_q_split_.shape(),
                                              DType::FP32, Device::Cuda);
        bl.k_split      = Tensor::from_blob(act_k_split_.data(),
                                              act_k_split_.shape(),
                                              DType::FP32, Device::Cuda);
        bl.v_split      = Tensor::from_blob(act_v_split_.data(),
                                              act_v_split_.shape(),
                                              DType::FP32, Device::Cuda);
        bl.q_normed     = Tensor::from_blob(act_q_normed_.data(),
                                              act_q_normed_.shape(),
                                              DType::FP32, Device::Cuda);
        bl.k_normed     = Tensor::from_blob(act_k_normed_.data(),
                                              act_k_normed_.shape(),
                                              DType::FP32, Device::Cuda);
        bl.q_norm_rstd  = Tensor::from_blob(act_q_norm_rstd_.data(),
                                              act_q_norm_rstd_.shape(),
                                              DType::FP32, Device::Cuda);
        bl.k_norm_rstd  = Tensor::from_blob(act_k_norm_rstd_.data(),
                                              act_k_norm_rstd_.shape(),
                                              DType::FP32, Device::Cuda);
        bl.k_full       = Tensor::from_blob(act_k_full_.data(),
                                              act_k_full_.shape(),
                                              DType::FP32, Device::Cuda);
        bl.v_full       = Tensor::from_blob(act_v_full_.data(),
                                              act_v_full_.shape(),
                                              DType::FP32, Device::Cuda);
        bl.ctx_split    = Tensor::from_blob(act_ctx_split_.data(),
                                              act_ctx_split_.shape(),
                                              DType::FP32, Device::Cuda);
        bl.probs        = Tensor::from_blob(act_probs_.data(),
                                              act_probs_.shape(),
                                              DType::FP32, Device::Cuda);
        bl.ctx          = Tensor::from_blob(act_ctx_.data(),
                                              act_ctx_.shape(),
                                              DType::FP32, Device::Cuda);
        bl.attn_out     = Tensor::from_blob(act_attn_out_.data(),
                                              act_attn_out_.shape(),
                                              DType::FP32, Device::Cuda);
        bl.h_residual1  = Tensor::from_blob(act_h_residual1_.data(),
                                              act_h_residual1_.shape(),
                                              DType::FP32, Device::Cuda);
        bl.ln2_out      = Tensor::from_blob(act_ln2_out_.data(),
                                              act_ln2_out_.shape(),
                                              DType::FP32, Device::Cuda);
        bl.ln2_rstd     = Tensor::from_blob(act_ln2_rstd_.data(),
                                              act_ln2_rstd_.shape(),
                                              DType::FP32, Device::Cuda);
        bl.ffn_gate     = Tensor::from_blob(act_ffn_gate_.data(),
                                              act_ffn_gate_.shape(),
                                              DType::FP32, Device::Cuda);
        bl.ffn_up       = Tensor::from_blob(act_ffn_up_.data(),
                                              act_ffn_up_.shape(),
                                              DType::FP32, Device::Cuda);
        bl.ffn_silu_up  = Tensor::from_blob(act_ffn_silu_up_.data(),
                                              act_ffn_silu_up_.shape(),
                                              DType::FP32, Device::Cuda);
        bl.ffn_out      = Tensor::from_blob(act_ffn_out_.data(),
                                              act_ffn_out_.shape(),
                                              DType::FP32, Device::Cuda);
        bl.h_block_out  = Tensor::from_blob(act_h_block_out_.data(),
                                              act_h_block_out_.shape(),
                                              DType::FP32, Device::Cuda);

        // BF16 weight mirrors (only allocated in bf16 mode).
        if (use_bf16_) {
            bl.Wq_bf16    = Tensor({D, D}, DType::BF16, Device::Cuda);
            bl.Wk_bf16    = Tensor({D, Dkv}, DType::BF16, Device::Cuda);
            bl.Wv_bf16    = Tensor({D, Dkv}, DType::BF16, Device::Cuda);
            bl.Wo_bf16    = Tensor({D, D}, DType::BF16, Device::Cuda);
            bl.Wgate_bf16 = Tensor({D, Dff}, DType::BF16, Device::Cuda);
            bl.Wup_bf16   = Tensor({D, Dff}, DType::BF16, Device::Cuda);
            bl.Wdown_bf16 = Tensor({Dff, D}, DType::BF16, Device::Cuda);
        }
    }

    // Scratch arena for BF16 X / dY casts during linear_*_bf16_arena calls.
    // Sized for the worst-case backward of either Wdown or Wgate, which need
    // X_bf16 [N, max(D, Dff)] + dY_bf16 [N, max(D, Dff)] simultaneously, plus
    // a generous safety margin and alignment padding.
    if (use_bf16_) {
        std::size_t worst_in_out = static_cast<std::size_t>(D) +
                                    static_cast<std::size_t>(Dff);
        std::size_t bytes = static_cast<std::size_t>(N) * worst_in_out *
                             2 /*BF16*/ * 2 /*safety*/;
        bytes += 1024 * 1024;  // 1 MB alignment padding
        bf16_arena_.initialize(bytes);
    }
}

void ModernGPT::refresh_bf16_mirrors() {
    if (!use_bf16_) return;
    for (auto& bl : blocks_) {
        cast_fp32_to_bf16(bl.Wq, bl.Wq_bf16);
        cast_fp32_to_bf16(bl.Wk, bl.Wk_bf16);
        cast_fp32_to_bf16(bl.Wv, bl.Wv_bf16);
        cast_fp32_to_bf16(bl.Wo, bl.Wo_bf16);
        cast_fp32_to_bf16(bl.Wgate, bl.Wgate_bf16);
        cast_fp32_to_bf16(bl.Wup, bl.Wup_bf16);
        cast_fp32_to_bf16(bl.Wdown, bl.Wdown_bf16);
    }
}

void ModernGPT::init_random(unsigned long long seed) {
    normal_(tok_emb_, 0.f, 0.02f, seed + 0);
    lnf_g_.fill(1.0f);
    // lm_head shares storage with tok_emb (tied embeddings).

    unsigned long long s = seed + 100;
    for (auto& bl : blocks_) {
        bl.ln1_g.fill(1.0f);
        bl.ln2_g.fill(1.0f);
        bl.q_norm_g.fill(1.0f);  // QK-Norm starts as identity
        bl.k_norm_g.fill(1.0f);
        normal_(bl.Wq, 0.f, 0.02f, s++);
        normal_(bl.Wk, 0.f, 0.02f, s++);
        normal_(bl.Wv, 0.f, 0.02f, s++);
        normal_(bl.Wo, 0.f, 0.02f, s++);
        normal_(bl.Wgate, 0.f, 0.02f, s++);
        normal_(bl.Wup, 0.f, 0.02f, s++);
        normal_(bl.Wdown, 0.f, 0.02f, s++);
    }
    refresh_bf16_mirrors();  // populate initial mirrors if in bf16 mode
}

std::vector<NamedParam> ModernGPT::named_params() {
    std::vector<NamedParam> v;
    v.push_back({"tok_emb", &tok_emb_, &d_tok_emb_});
    v.push_back({"lnf_g", &lnf_g_, &d_lnf_g_});
    // lm_head is tied to tok_emb — no separate parameter.
    for (int l = 0; l < cfg_.n_layers; ++l) {
        auto& bl = blocks_[l];
        const std::string p = "blocks." + std::to_string(l) + ".";
        v.push_back({p + "ln1_g", &bl.ln1_g, &bl.d_ln1_g});
        v.push_back({p + "Wq", &bl.Wq, &bl.d_Wq});
        v.push_back({p + "Wk", &bl.Wk, &bl.d_Wk});
        v.push_back({p + "Wv", &bl.Wv, &bl.d_Wv});
        v.push_back({p + "Wo", &bl.Wo, &bl.d_Wo});
        v.push_back({p + "q_norm_g", &bl.q_norm_g, &bl.d_q_norm_g});
        v.push_back({p + "k_norm_g", &bl.k_norm_g, &bl.d_k_norm_g});
        v.push_back({p + "ln2_g", &bl.ln2_g, &bl.d_ln2_g});
        v.push_back({p + "Wgate", &bl.Wgate, &bl.d_Wgate});
        v.push_back({p + "Wup", &bl.Wup, &bl.d_Wup});
        v.push_back({p + "Wdown", &bl.Wdown, &bl.d_Wdown});
    }
    return v;
}

void ModernGPT::register_with_optimizer(AdamW& opt, float wd) {
    auto params = named_params();
    for (auto& np : params) {
        // Norm gammas (LN + QK-Norm) use no weight decay; everything else uses `wd`.
        bool is_norm = np.name == "lnf_g" ||
                        np.name.find("ln1_g") != std::string::npos ||
                        np.name.find("ln2_g") != std::string::npos ||
                        np.name.find("q_norm_g") != std::string::npos ||
                        np.name.find("k_norm_g") != std::string::npos;
        opt.add_param(np.param, np.grad, is_norm ? 0.f : wd);
    }
}

std::vector<Tensor*> ModernGPT::all_grads() {
    auto params = named_params();
    std::vector<Tensor*> v;
    v.reserve(params.size());
    for (auto& np : params) v.push_back(np.grad);
    return v;
}

long long ModernGPT::parameter_count() const {
    long long n = 0;
    n += tok_emb_.numel() + lnf_g_.numel();  // lm_head tied to tok_emb
    for (auto& bl : blocks_) {
        n += bl.ln1_g.numel() + bl.Wq.numel() + bl.Wk.numel() + bl.Wv.numel()
           + bl.Wo.numel() + bl.q_norm_g.numel() + bl.k_norm_g.numel()
           + bl.ln2_g.numel() + bl.Wgate.numel()
           + bl.Wup.numel() + bl.Wdown.numel();
    }
    return n;
}

void ModernGPT::recompute_block_forward(int L, cublasHandle_t handle) {
    auto& bl = blocks_[L];
    const int B = cfg_.B, T = cfg_.T, D = cfg_.D, H = cfg_.H;
    const int Dkv = n_kv_ * d_h_;

    auto lin_fwd = [&](const Tensor& X, const Tensor& W_fp32,
                        const Tensor& W_bf16, const Tensor* bias,
                        Tensor& Y) {
        if (use_bf16_) {
            bf16_arena_.reset();
            linear_forward_bf16_arena(handle, X, W_bf16, bias, Y, bf16_arena_);
        } else {
            linear_forward(handle, X, W_fp32, bias, Y);
        }
    };

    rmsnorm_forward(bl.inp, bl.ln1_g, 1e-5f, bl.ln1_out, bl.ln1_rstd);
    lin_fwd(bl.ln1_out, bl.Wq, bl.Wq_bf16, nullptr, bl.q_proj);
    lin_fwd(bl.ln1_out, bl.Wk, bl.Wk_bf16, nullptr, bl.k_proj);
    lin_fwd(bl.ln1_out, bl.Wv, bl.Wv_bf16, nullptr, bl.v_proj);

    Tensor q3 = bl.q_proj.view({B, T, D});
    Tensor k3 = bl.k_proj.view({B, T, Dkv});
    Tensor v3 = bl.v_proj.view({B, T, Dkv});
    split_heads(q3, bl.q_split, B, T, H, d_h_);
    split_heads(k3, bl.k_split, B, T, n_kv_, d_h_);
    split_heads(v3, bl.v_split, B, T, n_kv_, d_h_);

    rope_apply_inplace(bl.q_split, rope_cos_, rope_sin_, B * H, T, d_h_);
    rope_apply_inplace(bl.k_split, rope_cos_, rope_sin_, B * n_kv_, T, d_h_);

    {
        Tensor q_flat = bl.q_split.view({B * H * T, d_h_});
        Tensor q_norm_flat = bl.q_normed.view({B * H * T, d_h_});
        rmsnorm_forward(q_flat, bl.q_norm_g, 1e-5f, q_norm_flat, bl.q_norm_rstd);
    }
    {
        Tensor k_flat = bl.k_split.view({B * n_kv_ * T, d_h_});
        Tensor k_norm_flat = bl.k_normed.view({B * n_kv_ * T, d_h_});
        rmsnorm_forward(k_flat, bl.k_norm_g, 1e-5f, k_norm_flat, bl.k_norm_rstd);
    }

    repeat_kv_heads(bl.k_normed, bl.k_full, B, n_kv_, H, T, d_h_);
    repeat_kv_heads(bl.v_split, bl.v_full, B, n_kv_, H, T, d_h_);

    scaled_dot_attention_forward(handle, bl.q_normed, bl.k_full, bl.v_full,
                                   bl.ctx_split, bl.probs, B * H, T, d_h_);

    merge_heads(bl.ctx_split, bl.ctx, B, T, H, d_h_);
    Tensor ctx_flat = bl.ctx.view({B * T, D});
    lin_fwd(ctx_flat, bl.Wo, bl.Wo_bf16, nullptr, bl.attn_out);

    bl.h_residual1.copy_from(bl.inp);
    add_inplace(bl.h_residual1, bl.attn_out);

    rmsnorm_forward(bl.h_residual1, bl.ln2_g, 1e-5f, bl.ln2_out, bl.ln2_rstd);
    lin_fwd(bl.ln2_out, bl.Wgate, bl.Wgate_bf16, nullptr, bl.ffn_gate);
    lin_fwd(bl.ln2_out, bl.Wup, bl.Wup_bf16, nullptr, bl.ffn_up);
    silu_mul_forward(bl.ffn_gate, bl.ffn_up, bl.ffn_silu_up);
    lin_fwd(bl.ffn_silu_up, bl.Wdown, bl.Wdown_bf16, nullptr, bl.ffn_out);

    bl.h_block_out.copy_from(bl.h_residual1);
    add_inplace(bl.h_block_out, bl.ffn_out);
}

float ModernGPT::forward(const Tensor& input_ids, const Tensor* target_ids,
                          cublasHandle_t handle) {
    embedding_forward(tok_emb_, input_ids, h_input_);

    Tensor* prev = &h_input_;
    for (int l = 0; l < cfg_.n_layers; ++l) {
        auto& bl = blocks_[l];
        bl.inp.copy_from(*prev);
        recompute_block_forward(l, handle);
        prev = &bl.h_block_out;
    }

    rmsnorm_forward(*prev, lnf_g_, 1e-5f, lnf_out_, lnf_rstd_);
    // Tied LM head: logits [N, V] = lnf_out [N, D] @ tok_emb^T [D, V].
    // Stays FP32 (cublasSgemm uses TF32 tensor cores on Ampere; with V ~ 100k
    // the cast to BF16 would dominate the GEMM speedup).
    {
        const int N = cfg_.B * cfg_.T;
        gemm_fp32_rowmajor(handle,
                            logits_.data_as<float>(),
                            lnf_out_.data_as<float>(),
                            tok_emb_.data_as<float>(),
                            N, cfg_.V, cfg_.D,
                            /*trans_a=*/false, /*trans_b=*/true);
    }

    if (target_ids) {
        softmax_ce_forward_backward(logits_, *target_ids, loss_pr_, dlogits_);
        return reduce_mean_to_scalar(loss_pr_);
    }
    return 0.0f;
}

void ModernGPT::backward(const Tensor& input_ids, cublasHandle_t handle) {
    const int B = cfg_.B, T = cfg_.T, D = cfg_.D, H = cfg_.H;

    auto block_lin_bwd = [&](const Tensor& X, const Tensor& W_fp32,
                              const Tensor& W_bf16, const Tensor& d_Y,
                              Tensor& d_X, Tensor& d_W, Tensor* d_b) {
        if (use_bf16_) {
            bf16_arena_.reset();
            linear_backward_bf16_arena(handle, X, W_bf16, d_Y, d_X, d_W,
                                         d_b, bf16_arena_);
        } else {
            linear_backward(handle, X, W_fp32, d_Y, d_X, d_W, d_b);
        }
    };

    // Tied LM head backward: forward was logits = lnf_out @ tok_emb^T.
    //   d_lnf_out [N, D] = dlogits [N, V] @ tok_emb [V, D]   (no transpose)
    //   d_tok_emb [V, D] += dlogits^T [V, N] @ lnf_out [N, D]  (accumulate)
    {
        const int N = cfg_.B * cfg_.T;
        gemm_fp32_rowmajor(handle,
                            d_lnf_out_.data_as<float>(),
                            dlogits_.data_as<float>(),
                            tok_emb_.data_as<float>(),
                            N, cfg_.D, cfg_.V,
                            /*trans_a=*/false, /*trans_b=*/false);
        // d_tok_emb is zeroed by opt.zero_grad at top of step. The embedding
        // backward later in this function adds its own contribution; using
        // beta=1 here means we accumulate (with zero start, this is just a
        // write).
        gemm_fp32_rowmajor(handle,
                            d_tok_emb_.data_as<float>(),
                            dlogits_.data_as<float>(),
                            lnf_out_.data_as<float>(),
                            cfg_.V, cfg_.D, N,
                            /*trans_a=*/true, /*trans_b=*/false,
                            /*alpha=*/1.0f, /*beta=*/1.0f);
    }

    Tensor& last_h = blocks_[cfg_.n_layers - 1].h_block_out;
    rmsnorm_backward(d_lnf_out_, last_h, lnf_g_, lnf_rstd_,
                       d_block_out_, d_lnf_g_);

    for (int l = cfg_.n_layers - 1; l >= 0; --l) {
        auto& bl = blocks_[l];

        // T7 grad checkpointing: per-block activations (q_split, ctx, etc.)
        // are SHARED across layers. Forward only leaves the LAST block's
        // values in those buffers. For every earlier layer, we must recompute
        // the forward of THIS layer from its saved input before backward.
        if (l < cfg_.n_layers - 1) {
            recompute_block_forward(l, handle);
        }

        d_h_residual1_.copy_from(d_block_out_);
        d_ffn_out_.copy_from(d_block_out_);

        block_lin_bwd(bl.ffn_silu_up, bl.Wdown, bl.Wdown_bf16, d_ffn_out_,
                         d_ffn_silu_up_, bl.d_Wdown, nullptr);
        silu_mul_backward(bl.ffn_gate, bl.ffn_up, d_ffn_silu_up_,
                            d_ffn_gate_, d_ffn_up_);
        block_lin_bwd(bl.ln2_out, bl.Wup, bl.Wup_bf16, d_ffn_up_,
                         d_branch_, bl.d_Wup, nullptr);
        block_lin_bwd(bl.ln2_out, bl.Wgate, bl.Wgate_bf16, d_ffn_gate_,
                         d_ln2_out_, bl.d_Wgate, nullptr);
        add_inplace(d_ln2_out_, d_branch_);

        rmsnorm_backward(d_ln2_out_, bl.h_residual1, bl.ln2_g, bl.ln2_rstd,
                           d_branch_, bl.d_ln2_g);
        add_inplace(d_h_residual1_, d_branch_);

        d_attn_out_.copy_from(d_h_residual1_);

        Tensor d_ctx_flat = d_ctx_.view({B * T, D});
        block_lin_bwd(bl.ctx.view({B * T, D}), bl.Wo, bl.Wo_bf16, d_attn_out_,
                         d_ctx_flat, bl.d_Wo, nullptr);

        Tensor d_ctx_split({B * H, T, d_h_}, DType::FP32, Device::Cuda);
        split_heads(d_ctx_, d_ctx_split, B, T, H, d_h_);

        // Attention backward sees the post-norm Q and the post-broadcast K, V
        // (n_q heads). It writes d_q_split / d_k_full / d_v_full where
        // d_q_split currently holds the gradient at the Q-NORM-OUTPUT level
        // (will be transformed by qk-norm bwd into the gradient at q_split's
        // RoPE level).
        scaled_dot_attention_backward(handle, d_ctx_split,
                                        bl.q_normed, bl.k_full, bl.v_full,
                                        bl.probs,
                                        d_q_split_, d_k_full_, d_v_full_,
                                        B * H, T, d_h_);

        d_k_split_.zero();
        d_v_split_.zero();
        accumulate_kv_grads(d_k_full_, d_k_split_, B, n_kv_, H, T, d_h_);
        accumulate_kv_grads(d_v_full_, d_v_split_, B, n_kv_, H, T, d_h_);

        // QK-Norm backward (in-place on d_q_split / d_k_split). The kernel
        // reads d_y / x for each row independently before writing d_x; running
        // it with d_y == d_x is safe.
        {
            Tensor q_flat = bl.q_split.view({B * H * T, d_h_});
            Tensor dq_flat = d_q_split_.view({B * H * T, d_h_});
            rmsnorm_backward(dq_flat, q_flat, bl.q_norm_g, bl.q_norm_rstd,
                              dq_flat, bl.d_q_norm_g);
        }
        {
            Tensor k_flat = bl.k_split.view({B * n_kv_ * T, d_h_});
            Tensor dk_flat = d_k_split_.view({B * n_kv_ * T, d_h_});
            rmsnorm_backward(dk_flat, k_flat, bl.k_norm_g, bl.k_norm_rstd,
                              dk_flat, bl.d_k_norm_g);
        }

        // RoPE backward on Q (n_q heads) and K (n_kv heads). V has no RoPE.
        rope_apply_backward_inplace(d_q_split_, rope_cos_, rope_sin_,
                                       B * H, T, d_h_);
        rope_apply_backward_inplace(d_k_split_, rope_cos_, rope_sin_,
                                       B * n_kv_, T, d_h_);

        const int Dkv = n_kv_ * d_h_;
        Tensor dqp3 = d_q_proj_.view({B, T, D});
        Tensor dkp3 = d_k_proj_.view({B, T, Dkv});
        Tensor dvp3 = d_v_proj_.view({B, T, Dkv});
        merge_heads(d_q_split_, dqp3, B, T, H, d_h_);
        merge_heads(d_k_split_, dkp3, B, T, n_kv_, d_h_);
        merge_heads(d_v_split_, dvp3, B, T, n_kv_, d_h_);

        block_lin_bwd(bl.ln1_out, bl.Wq, bl.Wq_bf16, d_q_proj_,
                         d_ln1_out_, bl.d_Wq, nullptr);
        block_lin_bwd(bl.ln1_out, bl.Wk, bl.Wk_bf16, d_k_proj_,
                         d_branch_, bl.d_Wk, nullptr);
        add_inplace(d_ln1_out_, d_branch_);
        block_lin_bwd(bl.ln1_out, bl.Wv, bl.Wv_bf16, d_v_proj_,
                         d_branch_, bl.d_Wv, nullptr);
        add_inplace(d_ln1_out_, d_branch_);

        rmsnorm_backward(d_ln1_out_, bl.inp, bl.ln1_g, bl.ln1_rstd,
                           d_branch_, bl.d_ln1_g);
        add_inplace(d_h_residual1_, d_branch_);

        d_block_out_.copy_from(d_h_residual1_);
    }

    embedding_backward(d_block_out_, input_ids, d_tok_emb_);
}

}  // namespace modernllm
