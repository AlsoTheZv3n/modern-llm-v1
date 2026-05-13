// Stage B.2 — 1-block, single-head pre-norm transformer on TinyShakespeare.
//
// Architecture (per nanoGPT, simplified):
//   x_ids [B, T]
//   h0   = tok_embed[x_ids] + pos_embed[0..T-1]
//   h0p  = LN1(h0)
//   q,k,v = Linear(h0p) for each
//   ctx  = causal_attn(q, k, v)
//   h1   = h0 + Linear(ctx)
//   h1p  = LN2(h1)
//   h2   = h1 + Linear( GELU( Linear(h1p) ) )
//   logits = Linear( LN_f(h2) )

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/gemm.h"
#include "core/random.h"
#include "core/tensor.h"
#include "data/dataloader.h"
#include "model/activations.h"
#include "model/attention.h"
#include "model/embedding.h"
#include "model/layernorm.h"
#include "model/linear.h"
#include "tokenizer/char_tokenizer.h"
#include "train/adamw.h"
#include "train/loss.h"

using modernllm::AdamW;
using modernllm::AdamWConfig;
using modernllm::CharTokenizer;
using modernllm::CublasHandle;
using modernllm::Device;
using modernllm::DType;
using modernllm::Tensor;
using modernllm::TextDataset;

namespace {

struct Args {
    std::string data_path = "data/tinyshakespeare.txt";
    std::string log_path = "runs/minigpt_log.jsonl";
    int batch_size = 32;
    int seq_len = 32;
    int d_model = 64;
    int d_ffn = 256;
    int max_steps = 1000;
    int log_every = 25;
    float lr = 3e-3f;
    float weight_decay = 0.0f;
    unsigned long long seed = 1337ULL;
};

void parse_args(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + k);
            return argv[++i];
        };
        if      (k == "--data")       a.data_path = next();
        else if (k == "--log")        a.log_path = next();
        else if (k == "--batch")      a.batch_size = std::stoi(next());
        else if (k == "--seq-len")    a.seq_len = std::stoi(next());
        else if (k == "--d-model")    a.d_model = std::stoi(next());
        else if (k == "--d-ffn")      a.d_ffn = std::stoi(next());
        else if (k == "--steps")      a.max_steps = std::stoi(next());
        else if (k == "--log-every")  a.log_every = std::stoi(next());
        else if (k == "--lr")         a.lr = std::stof(next());
        else if (k == "--seed")       a.seed = std::stoull(next());
        else throw std::runtime_error("unknown arg: " + k);
    }
}

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::ostringstream ss; ss << in.rdbuf(); return ss.str();
}

void ensure_parent_dir(const std::string& path) {
    auto pos = path.find_last_of("/\\");
    if (pos == std::string::npos) return;
    std::string dir = path.substr(0, pos);
    if (dir.empty()) return;
    std::filesystem::create_directories(dir);
}

// All model parameters and gradient buffers, plus the activations needed for
// backward. Allocated once at start, reused every step.
struct Model {
    int B, T, V, D, Dff, T_max;

    // Parameters
    Tensor tok_emb, pos_emb;
    Tensor ln1_g, ln1_b;
    Tensor Wq, Wk, Wv, Wo;
    Tensor bq, bk, bv, bo;
    Tensor ln2_g, ln2_b;
    Tensor W1, b1, W2, b2;
    Tensor lnf_g, lnf_b;
    Tensor lm_head;  // [D, V], no bias

    // Gradients (same shape, prefix d_)
    Tensor d_tok_emb, d_pos_emb;
    Tensor d_ln1_g, d_ln1_b;
    Tensor d_Wq, d_Wk, d_Wv, d_Wo;
    Tensor d_bq, d_bk, d_bv, d_bo;
    Tensor d_ln2_g, d_ln2_b;
    Tensor d_W1, d_b1, d_W2, d_b2;
    Tensor d_lnf_g, d_lnf_b;
    Tensor d_lm_head;

    // Activations (saved for backward)
    Tensor pos_ids;            // [B*T] INT32 — values 0..T-1 repeated B times
    Tensor h0;                 // [B*T, D]
    Tensor h1, h2;             // [B*T, D]
    Tensor ln1_out, ln1_mean, ln1_rstd;
    Tensor ln2_out, ln2_mean, ln2_rstd;
    Tensor lnf_out, lnf_mean, lnf_rstd;
    Tensor q_buf, k_buf, v_buf;     // [B*T, D]
    Tensor probs;                    // [B, T, T]
    Tensor ctx_buf;                  // [B*T, D]
    Tensor attn_out;                 // [B*T, D]
    Tensor ffn_h;                    // [B*T, Dff]
    Tensor ffn_h_act;                // [B*T, Dff]
    Tensor ffn_out;                  // [B*T, D]
    Tensor logits;                   // [B*T, V]
    Tensor loss_pr;                  // [B*T]

    // Backward scratch (allocate once, reuse)
    Tensor dlogits;                  // [B*T, V]
    Tensor d_lnf_out;                // [B*T, D]
    Tensor d_h2, d_h1, d_h0;
    Tensor d_h_branch;               // temp for residual addition
    Tensor d_ffn_out, d_ffn_h_act, d_ffn_h, d_ln2_out;
    Tensor d_attn_out, d_ctx, d_q, d_k, d_v, d_ln1_out;

    void allocate(int B_, int T_, int V_, int D_, int Dff_, int T_max_) {
        B = B_; T = T_; V = V_; D = D_; Dff = Dff_; T_max = T_max_;

        auto C = [](std::initializer_list<std::int64_t> s, DType dt = DType::FP32) {
            return Tensor(s, dt, Device::Cuda);
        };
        auto Z = [](std::initializer_list<std::int64_t> s) {
            return Tensor::zeros(s, DType::FP32, Device::Cuda);
        };

        // Params
        tok_emb = C({V, D}); pos_emb = C({T_max, D});
        ln1_g = C({D}); ln1_b = C({D});
        Wq = C({D, D}); Wk = C({D, D}); Wv = C({D, D}); Wo = C({D, D});
        bq = C({D}); bk = C({D}); bv = C({D}); bo = C({D});
        ln2_g = C({D}); ln2_b = C({D});
        W1 = C({D, Dff}); b1 = C({Dff});
        W2 = C({Dff, D}); b2 = C({D});
        lnf_g = C({D}); lnf_b = C({D});
        lm_head = C({D, V});

        // Gradients (zero-init)
        d_tok_emb = Z({V, D}); d_pos_emb = Z({T_max, D});
        d_ln1_g = Z({D}); d_ln1_b = Z({D});
        d_Wq = Z({D, D}); d_Wk = Z({D, D}); d_Wv = Z({D, D}); d_Wo = Z({D, D});
        d_bq = Z({D}); d_bk = Z({D}); d_bv = Z({D}); d_bo = Z({D});
        d_ln2_g = Z({D}); d_ln2_b = Z({D});
        d_W1 = Z({D, Dff}); d_b1 = Z({Dff});
        d_W2 = Z({Dff, D}); d_b2 = Z({D});
        d_lnf_g = Z({D}); d_lnf_b = Z({D});
        d_lm_head = Z({D, V});

        // Position ids: [B*T] with values 0..T-1 repeated B times
        pos_ids = Tensor({B * T}, DType::INT32, Device::Cuda);
        Tensor pos_h({B * T}, DType::INT32, Device::Host);
        auto* p = pos_h.data_as<int>();
        for (int b = 0; b < B; ++b)
            for (int t = 0; t < T; ++t)
                p[b * T + t] = t;
        pos_ids.copy_from(pos_h);

        // Activations
        const int N = B * T;
        h0 = C({N, D}); h1 = C({N, D}); h2 = C({N, D});
        ln1_out = C({N, D}); ln1_mean = C({N}); ln1_rstd = C({N});
        ln2_out = C({N, D}); ln2_mean = C({N}); ln2_rstd = C({N});
        lnf_out = C({N, D}); lnf_mean = C({N}); lnf_rstd = C({N});
        q_buf = C({N, D}); k_buf = C({N, D}); v_buf = C({N, D});
        probs = C({B, T, T});
        ctx_buf = C({N, D});
        attn_out = C({N, D});
        ffn_h = C({N, Dff});
        ffn_h_act = C({N, Dff});
        ffn_out = C({N, D});
        logits = C({N, V});
        loss_pr = C({N});

        dlogits = C({N, V});
        d_lnf_out = C({N, D});
        d_h2 = C({N, D}); d_h1 = C({N, D}); d_h0 = C({N, D});
        d_h_branch = C({N, D});
        d_ffn_out = C({N, D});
        d_ffn_h_act = C({N, Dff});
        d_ffn_h = C({N, Dff});
        d_ln2_out = C({N, D});
        d_attn_out = C({N, D});
        d_ctx = C({B, T, D});
        d_q = C({N, D}); d_k = C({N, D}); d_v = C({N, D});
        d_ln1_out = C({N, D});
    }

    void init_random(unsigned long long seed) {
        // Standard nanoGPT-ish init: small Gaussian for embeddings/linear,
        // ones/zeros for layernorm.
        modernllm::normal_(tok_emb, 0.f, 0.02f, seed + 0);
        modernllm::normal_(pos_emb, 0.f, 0.02f, seed + 1);
        ln1_g.fill(1.0f); ln1_b.fill(0.0f);
        ln2_g.fill(1.0f); ln2_b.fill(0.0f);
        lnf_g.fill(1.0f); lnf_b.fill(0.0f);
        modernllm::normal_(Wq, 0.f, 0.02f, seed + 2);
        modernllm::normal_(Wk, 0.f, 0.02f, seed + 3);
        modernllm::normal_(Wv, 0.f, 0.02f, seed + 4);
        modernllm::normal_(Wo, 0.f, 0.02f, seed + 5);
        bq.fill(0.f); bk.fill(0.f); bv.fill(0.f); bo.fill(0.f);
        modernllm::normal_(W1, 0.f, 0.02f, seed + 6);
        modernllm::normal_(W2, 0.f, 0.02f, seed + 7);
        b1.fill(0.f); b2.fill(0.f);
        modernllm::normal_(lm_head, 0.f, 0.02f, seed + 8);
    }

    // Register every (param, grad) pair with the optimizer. Biases and LN
    // params get weight_decay = 0 (standard practice).
    void register_with_optimizer(AdamW& opt, float wd) {
        opt.add_param(&tok_emb, &d_tok_emb, wd);
        opt.add_param(&pos_emb, &d_pos_emb, wd);
        opt.add_param(&ln1_g, &d_ln1_g, 0.f);
        opt.add_param(&ln1_b, &d_ln1_b, 0.f);
        opt.add_param(&Wq, &d_Wq, wd);
        opt.add_param(&Wk, &d_Wk, wd);
        opt.add_param(&Wv, &d_Wv, wd);
        opt.add_param(&Wo, &d_Wo, wd);
        opt.add_param(&bq, &d_bq, 0.f);
        opt.add_param(&bk, &d_bk, 0.f);
        opt.add_param(&bv, &d_bv, 0.f);
        opt.add_param(&bo, &d_bo, 0.f);
        opt.add_param(&ln2_g, &d_ln2_g, 0.f);
        opt.add_param(&ln2_b, &d_ln2_b, 0.f);
        opt.add_param(&W1, &d_W1, wd);
        opt.add_param(&b1, &d_b1, 0.f);
        opt.add_param(&W2, &d_W2, wd);
        opt.add_param(&b2, &d_b2, 0.f);
        opt.add_param(&lnf_g, &d_lnf_g, 0.f);
        opt.add_param(&lnf_b, &d_lnf_b, 0.f);
        opt.add_param(&lm_head, &d_lm_head, wd);
    }
};

float forward(Model& M, const Tensor& input_ids, const Tensor& target_ids,
              cublasHandle_t handle) {
    using modernllm::add_inplace;
    using modernllm::embedding_forward;
    using modernllm::gelu_forward;
    using modernllm::layernorm_forward;
    using modernllm::linear_forward;
    using modernllm::scaled_dot_attention_forward;
    using modernllm::softmax_ce_forward_backward;

    const int B = M.B, T = M.T, D = M.D, Dff = M.Dff;

    // 1. Token + position embedding sum -> h0
    embedding_forward(M.tok_emb, input_ids, M.h0);
    Tensor pe_buf({B * T, D}, DType::FP32, Device::Cuda);
    embedding_forward(M.pos_emb, M.pos_ids, pe_buf);
    add_inplace(M.h0, pe_buf);

    // 2. LN1
    layernorm_forward(M.h0, M.ln1_g, M.ln1_b, 1e-5f,
                       M.ln1_out, M.ln1_mean, M.ln1_rstd);

    // 3. Q, K, V projections
    linear_forward(handle, M.ln1_out, M.Wq, &M.bq, M.q_buf);
    linear_forward(handle, M.ln1_out, M.Wk, &M.bk, M.k_buf);
    linear_forward(handle, M.ln1_out, M.Wv, &M.bv, M.v_buf);

    // 4. Causal attention (view 2D as 3D)
    Tensor q3 = M.q_buf.view({B, T, D});
    Tensor k3 = M.k_buf.view({B, T, D});
    Tensor v3 = M.v_buf.view({B, T, D});
    Tensor ctx3 = M.ctx_buf.view({B, T, D});
    scaled_dot_attention_forward(handle, q3, k3, v3, ctx3, M.probs, B, T, D);

    // 5. Output projection
    linear_forward(handle, M.ctx_buf, M.Wo, &M.bo, M.attn_out);

    // 6. Residual: h1 = h0 + attn_out
    M.h1.copy_from(M.h0);
    add_inplace(M.h1, M.attn_out);

    // 7. LN2 + FFN
    layernorm_forward(M.h1, M.ln2_g, M.ln2_b, 1e-5f,
                       M.ln2_out, M.ln2_mean, M.ln2_rstd);
    linear_forward(handle, M.ln2_out, M.W1, &M.b1, M.ffn_h);
    gelu_forward(M.ffn_h, M.ffn_h_act);
    linear_forward(handle, M.ffn_h_act, M.W2, &M.b2, M.ffn_out);

    // 8. Residual: h2 = h1 + ffn_out
    M.h2.copy_from(M.h1);
    add_inplace(M.h2, M.ffn_out);

    // 9. LN_f + LM head
    layernorm_forward(M.h2, M.lnf_g, M.lnf_b, 1e-5f,
                       M.lnf_out, M.lnf_mean, M.lnf_rstd);
    linear_forward(handle, M.lnf_out, M.lm_head, /*bias=*/nullptr, M.logits);

    // 10. Loss (also produces dlogits)
    softmax_ce_forward_backward(M.logits, target_ids, M.loss_pr, M.dlogits);
    return modernllm::reduce_mean_to_scalar(M.loss_pr);
}

void backward(Model& M, const Tensor& input_ids, cublasHandle_t handle) {
    using modernllm::add_inplace;
    using modernllm::embedding_backward;
    using modernllm::gelu_backward;
    using modernllm::layernorm_backward;
    using modernllm::linear_backward;
    using modernllm::scaled_dot_attention_backward;

    const int B = M.B, T = M.T, D = M.D;

    // Reverse of step 9: linear_backward through lm_head.
    // We zeroed all grads at top of step, so accumulation into d_lm_head is OK.
    linear_backward(handle, M.lnf_out, M.lm_head, M.dlogits,
                     M.d_lnf_out, M.d_lm_head, /*d_b=*/nullptr);

    // Reverse of LN_f
    layernorm_backward(M.d_lnf_out, M.h2, M.lnf_g, M.lnf_mean, M.lnf_rstd,
                        M.d_h2, M.d_lnf_g, M.d_lnf_b);

    // h2 = h1 + ffn_out → d_h1 = d_h2,  d_ffn_out = d_h2
    M.d_ffn_out.copy_from(M.d_h2);
    M.d_h1.copy_from(M.d_h2);

    // Reverse FFN: linear → gelu → linear
    linear_backward(handle, M.ffn_h_act, M.W2, M.d_ffn_out,
                     M.d_ffn_h_act, M.d_W2, &M.d_b2);
    gelu_backward(M.ffn_h, M.d_ffn_h_act, M.d_ffn_h);
    linear_backward(handle, M.ln2_out, M.W1, M.d_ffn_h,
                     M.d_ln2_out, M.d_W1, &M.d_b1);

    // Reverse LN2 → adds to d_h1 (via d_h_branch)
    layernorm_backward(M.d_ln2_out, M.h1, M.ln2_g, M.ln2_mean, M.ln2_rstd,
                        M.d_h_branch, M.d_ln2_g, M.d_ln2_b);
    add_inplace(M.d_h1, M.d_h_branch);

    // h1 = h0 + attn_out → d_h0 = d_h1, d_attn_out = d_h1
    M.d_attn_out.copy_from(M.d_h1);
    M.d_h0.copy_from(M.d_h1);

    // Reverse output projection of attention
    Tensor d_ctx_flat = M.d_ctx.view({B * T, D});
    linear_backward(handle, M.ctx_buf, M.Wo, M.d_attn_out,
                     d_ctx_flat, M.d_Wo, &M.d_bo);

    // Reverse scaled-dot attention → d_q, d_k, d_v (each [B, T, D])
    Tensor q3 = M.q_buf.view({B, T, D});
    Tensor k3 = M.k_buf.view({B, T, D});
    Tensor v3 = M.v_buf.view({B, T, D});
    Tensor dq3 = M.d_q.view({B, T, D});
    Tensor dk3 = M.d_k.view({B, T, D});
    Tensor dv3 = M.d_v.view({B, T, D});
    scaled_dot_attention_backward(handle, M.d_ctx, q3, k3, v3, M.probs,
                                   dq3, dk3, dv3, B, T, D);

    // Reverse Q, K, V linear projections.
    // Each contributes to d_ln1_out → we sum them. Use d_ln1_out as
    // accumulator: Q first (overwrites), then K and V add into it.
    linear_backward(handle, M.ln1_out, M.Wq, M.d_q,
                     M.d_ln1_out, M.d_Wq, &M.d_bq);
    linear_backward(handle, M.ln1_out, M.Wk, M.d_k,
                     M.d_h_branch, M.d_Wk, &M.d_bk);
    add_inplace(M.d_ln1_out, M.d_h_branch);
    linear_backward(handle, M.ln1_out, M.Wv, M.d_v,
                     M.d_h_branch, M.d_Wv, &M.d_bv);
    add_inplace(M.d_ln1_out, M.d_h_branch);

    // Reverse LN1 → adds to d_h0
    layernorm_backward(M.d_ln1_out, M.h0, M.ln1_g, M.ln1_mean, M.ln1_rstd,
                        M.d_h_branch, M.d_ln1_g, M.d_ln1_b);
    add_inplace(M.d_h0, M.d_h_branch);

    // Reverse the embedding sum: d_h0 splits to d_tok_emb and d_pos_emb
    embedding_backward(M.d_h0, input_ids, M.d_tok_emb);
    embedding_backward(M.d_h0, M.pos_ids, M.d_pos_emb);
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    try {
        parse_args(argc, argv, args);
    } catch (std::exception& e) {
        std::fprintf(stderr, "args: %s\n", e.what());
        return 2;
    }

    // Tokenizer + data
    std::string text = read_file(args.data_path);
    CharTokenizer tok;
    tok.fit(text);
    int V = tok.vocab_size();
    TextDataset ds;
    ds.load(args.data_path, tok);
    std::printf("data: %lld tokens, vocab=%d\n",
                 static_cast<long long>(ds.num_tokens()), V);
    std::printf("model: B=%d T=%d D=%d D_ffn=%d V=%d  (%.2fK params)\n",
                 args.batch_size, args.seq_len, args.d_model, args.d_ffn, V,
                 (V + args.seq_len) * args.d_model * 1e-3 +
                     5 * args.d_model * args.d_model * 1e-3 +
                     2 * args.d_model * args.d_ffn * 1e-3);

    Model M;
    M.allocate(args.batch_size, args.seq_len, V,
                args.d_model, args.d_ffn, args.seq_len);
    M.init_random(args.seed);

    AdamWConfig cfg;
    cfg.lr = args.lr;
    cfg.weight_decay = args.weight_decay;
    AdamW opt(cfg);
    M.register_with_optimizer(opt, args.weight_decay);

    CublasHandle handle;

    int N = args.batch_size * args.seq_len;
    Tensor inputs_h({N}, DType::INT32, Device::Host);
    Tensor targets_h({N}, DType::INT32, Device::Host);
    Tensor inputs_d({N}, DType::INT32, Device::Cuda);
    Tensor targets_d({N}, DType::INT32, Device::Cuda);

    std::mt19937_64 rng(args.seed);
    std::vector<std::int32_t> in_buf, tgt_buf;

    ensure_parent_dir(args.log_path);
    std::ofstream log(args.log_path, std::ios::out);

    float first_loss = -1.f, last_loss = -1.f;
    auto run_start = std::chrono::steady_clock::now();

    for (int step = 1; step <= args.max_steps; ++step) {
        ds.sample_batch(args.batch_size, args.seq_len, rng, in_buf, tgt_buf);
        std::memcpy(inputs_h.data(), in_buf.data(),
                     static_cast<std::size_t>(N) * sizeof(int));
        std::memcpy(targets_h.data(), tgt_buf.data(),
                     static_cast<std::size_t>(N) * sizeof(int));
        inputs_d.copy_from(inputs_h);
        targets_d.copy_from(targets_h);

        opt.zero_grad();
        float loss = forward(M, inputs_d, targets_d, handle);
        backward(M, inputs_d, handle);
        opt.step();

        if (first_loss < 0.f) first_loss = loss;
        last_loss = loss;

        if (step % args.log_every == 0 || step == 1 || step == args.max_steps) {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - run_start).count();
            std::printf("  step %5d  loss %.4f  elapsed %.2fs\n",
                         step, loss, elapsed);
            log << "{\"step\":" << step
                << ",\"loss\":" << loss
                << ",\"lr\":" << cfg.lr
                << ",\"elapsed_sec\":" << elapsed << "}\n";
            log.flush();
        }
    }

    auto end = std::chrono::steady_clock::now();
    double total = std::chrono::duration<double>(end - run_start).count();
    std::printf("\ndone: %d steps in %.2fs (%.1f steps/s)\n",
                 args.max_steps, total, args.max_steps / std::max(total, 1e-6));
    std::printf("loss: %.4f -> %.4f (drop %.4f)\n",
                 first_loss, last_loss, first_loss - last_loss);

    if (last_loss >= first_loss) {
        std::fprintf(stderr, "WARNING: loss did not decrease.\n");
        return 1;
    }
    return 0;
}
