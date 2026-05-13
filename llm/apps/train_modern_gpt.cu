// Stage 1 — train_modern_gpt with checkpoint save/load + validation loss.
//
// Architecture is owned by model/modern_gpt.{h,cu}. This binary just wires
// the dataset, optimizer, and the train loop.

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
#include "core/tensor.h"
#include "data/dataloader.h"
#include "data/token_meta.h"
#include "model/modern_gpt.h"
#include "tokenizer/char_tokenizer.h"
#include "train/adamw.h"
#include "train/checkpoint.h"
#include "train/grad_utils.h"
#include "train/loss.h"

using modernllm::AdamW;
using modernllm::AdamWConfig;
using modernllm::CharTokenizer;
using modernllm::CheckpointInfo;
using modernllm::CublasHandle;
using modernllm::Device;
using modernllm::DType;
using modernllm::ModernGPT;
using modernllm::Tensor;
using modernllm::TextDataset;

namespace {

struct Args {
    std::string data_path = "data/tinyshakespeare.txt";
    std::string log_path = "runs/modern_gpt_log.jsonl";
    std::string save_path = "runs/modern_gpt.ckpt";
    std::string resume_path = "";

    int batch_size = 32;
    int seq_len = 64;
    int d_model = 128;
    int n_heads = 4;
    int n_kv_heads = 0;     // 0 = MHA (n_kv = n_heads)
    int n_layers = 4;
    int d_ffn = 512;
    int max_steps = 1500;
    int warmup_steps = 50;
    int log_every = 25;
    int save_every = 0;        // 0 disables
    int val_every = 0;         // 0 disables
    int val_batches = 4;
    float val_frac = 0.1f;     // last 10% of corpus held out

    float max_lr = 3e-3f;
    float min_lr = 3e-4f;
    float weight_decay = 0.01f;
    float grad_clip = 1.0f;
    float rope_base = 10000.f;
    unsigned long long seed = 1337ULL;
    bool bf16 = false;
    bool opt_bf16 = false;  // T8: store AdamW m, v as BF16 (2x memory savings)
};

void parse_args(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + k);
            return argv[++i];
        };
        if      (k == "--data")        a.data_path = next();
        else if (k == "--log")         a.log_path = next();
        else if (k == "--save-path")   a.save_path = next();
        else if (k == "--resume")      a.resume_path = next();
        else if (k == "--batch")       a.batch_size = std::stoi(next());
        else if (k == "--seq-len")     a.seq_len = std::stoi(next());
        else if (k == "--d-model")     a.d_model = std::stoi(next());
        else if (k == "--n-heads")     a.n_heads = std::stoi(next());
        else if (k == "--n-kv-heads")  a.n_kv_heads = std::stoi(next());
        else if (k == "--n-layers")    a.n_layers = std::stoi(next());
        else if (k == "--d-ffn")       a.d_ffn = std::stoi(next());
        else if (k == "--steps")       a.max_steps = std::stoi(next());
        else if (k == "--warmup")      a.warmup_steps = std::stoi(next());
        else if (k == "--log-every")   a.log_every = std::stoi(next());
        else if (k == "--save-every")  a.save_every = std::stoi(next());
        else if (k == "--val-every")   a.val_every = std::stoi(next());
        else if (k == "--val-batches") a.val_batches = std::stoi(next());
        else if (k == "--val-frac")    a.val_frac = std::stof(next());
        else if (k == "--max-lr")      a.max_lr = std::stof(next());
        else if (k == "--min-lr")      a.min_lr = std::stof(next());
        else if (k == "--wd")          a.weight_decay = std::stof(next());
        else if (k == "--grad-clip")   a.grad_clip = std::stof(next());
        else if (k == "--seed")        a.seed = std::stoull(next());
        else if (k == "--bf16")        a.bf16 = true;
        else if (k == "--opt-bf16")    a.opt_bf16 = true;
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

bool ends_with(const std::string& s, const std::string& suffix) {
    if (s.size() < suffix.size()) return false;
    return std::equal(suffix.rbegin(), suffix.rend(), s.rbegin());
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    try { parse_args(argc, argv, args); }
    catch (std::exception& e) {
        std::fprintf(stderr, "args: %s\n", e.what());
        return 2;
    }

    // Data: char-level (.txt) or pre-tokenized binary (.bin) mode
    int V = 0;
    TextDataset ds;
    bool binary_mode = ends_with(args.data_path, ".bin");
    if (binary_mode) {
        modernllm::TokenMeta meta =
            modernllm::read_token_meta(args.data_path + ".meta");
        V = meta.vocab_size;
        ds.load_binary(args.data_path);
        std::printf("data: binary tokens (%s, %lld tokens, vocab=%d)\n",
                     meta.encoding.c_str(),
                     static_cast<long long>(meta.num_tokens), V);
    } else {
        std::string text = read_file(args.data_path);
        CharTokenizer tok;
        tok.fit(text);
        V = tok.vocab_size();
        ds.load(args.data_path, tok);
        std::printf("data: char-level tokens (vocab=%d)\n", V);
    }
    std::int64_t total_tokens = ds.num_tokens();
    std::int64_t val_tokens = static_cast<std::int64_t>(
        total_tokens * static_cast<double>(args.val_frac));
    std::int64_t train_end = total_tokens - val_tokens;
    std::printf("split: train=%lld, val=%lld\n",
                 static_cast<long long>(train_end),
                 static_cast<long long>(val_tokens));

    // Model
    ModernGPT::Config cfg;
    cfg.B = args.batch_size; cfg.T = args.seq_len; cfg.V = V;
    cfg.D = args.d_model; cfg.H = args.n_heads;
    cfg.n_kv_heads = args.n_kv_heads;  // 0 = default to MHA
    cfg.Dff = args.d_ffn; cfg.n_layers = args.n_layers;
    cfg.rope_base = args.rope_base;

    ModernGPT model;
    model.set_use_bf16(args.bf16);  // must be set BEFORE allocate
    model.allocate(cfg);
    model.init_random(args.seed);
    if (args.bf16) {
        std::printf("compute: BF16 mixed precision (FP32 master + persistent BF16 mirrors)\n");
    }

    AdamWConfig optcfg;
    optcfg.lr = args.max_lr;
    optcfg.weight_decay = args.weight_decay;
    optcfg.bf16_states = args.opt_bf16;
    AdamW opt(optcfg);
    if (args.opt_bf16) {
        std::printf("optimizer: AdamW with BF16 m/v states (2x memory savings)\n");
    }
    model.register_with_optimizer(opt, args.weight_decay);
    auto grads = model.all_grads();
    auto named = model.named_params();

    int eff_kv = args.n_kv_heads > 0 ? args.n_kv_heads : args.n_heads;
    std::printf("model: B=%d T=%d D=%d H=%d (Hkv=%d) Dff=%d L=%d  (%.2fM params)\n",
                 args.batch_size, args.seq_len, args.d_model, args.n_heads,
                 eff_kv, args.d_ffn, args.n_layers,
                 model.parameter_count() * 1e-6);

    // Resume
    int start_step = 0;
    long long tokens_seen = 0;
    if (!args.resume_path.empty()) {
        CheckpointInfo info;
        if (modernllm::load_checkpoint(args.resume_path, named, opt, info)) {
            start_step = info.step;
            tokens_seen = info.tokens_seen;
            std::printf("resumed from %s (step=%d, loss=%.4f, tokens_seen=%lld)\n",
                         args.resume_path.c_str(), info.step, info.loss,
                         static_cast<long long>(info.tokens_seen));
            // Repopulate BF16 mirrors from the freshly-loaded FP32 weights.
            model.refresh_bf16_mirrors();
        } else {
            std::fprintf(stderr, "resume file not found: %s\n",
                          args.resume_path.c_str());
            return 3;
        }
    }

    CublasHandle handle;

    int N = args.batch_size * args.seq_len;
    Tensor inputs_h({N}, DType::INT32, Device::Host);
    Tensor targets_h({N}, DType::INT32, Device::Host);
    Tensor inputs_d({N}, DType::INT32, Device::Cuda);
    Tensor targets_d({N}, DType::INT32, Device::Cuda);

    std::mt19937_64 rng(args.seed + start_step);
    std::vector<std::int32_t> in_buf, tgt_buf;

    ensure_parent_dir(args.log_path);
    std::ofstream log(args.log_path,
                      args.resume_path.empty() ? std::ios::out
                                                : std::ios::app);

    auto upload_batch = [&](const std::vector<std::int32_t>& in_v,
                             const std::vector<std::int32_t>& tgt_v) {
        std::memcpy(inputs_h.data(), in_v.data(), N * sizeof(int));
        std::memcpy(targets_h.data(), tgt_v.data(), N * sizeof(int));
        inputs_d.copy_from(inputs_h);
        targets_d.copy_from(targets_h);
    };

    auto compute_val_loss = [&]() {
        if (val_tokens < args.seq_len + 1) return 0.f;
        std::mt19937_64 vrng(0xCAFEBABEULL);  // deterministic across calls
        std::vector<std::int32_t> vin, vtgt;
        double total = 0.0;
        for (int k = 0; k < args.val_batches; ++k) {
            ds.sample_batch_range(args.batch_size, args.seq_len, vrng,
                                    train_end, total_tokens, vin, vtgt);
            upload_batch(vin, vtgt);
            total += model.forward(inputs_d, &targets_d, handle);
        }
        return static_cast<float>(total / args.val_batches);
    };

    float first_loss = -1.f, last_loss = -1.f;
    auto run_start = std::chrono::steady_clock::now();

    int step = start_step;
    while (step < args.max_steps) {
        ++step;
        float lr = modernllm::cosine_lr_with_warmup(step, args.warmup_steps,
                                                      args.max_steps,
                                                      args.max_lr,
                                                      args.min_lr);
        opt.set_lr(lr);

        ds.sample_batch_range(args.batch_size, args.seq_len, rng,
                                0, train_end, in_buf, tgt_buf);
        upload_batch(in_buf, tgt_buf);

        opt.zero_grad();
        float loss = model.forward(inputs_d, &targets_d, handle);
        model.backward(inputs_d, handle);
        float gnorm = modernllm::clip_grad_norm(grads, args.grad_clip);
        opt.step();
        // Re-cast updated FP32 master weights into the persistent BF16 mirrors.
        // No-op when use_bf16 is false.
        model.refresh_bf16_mirrors();

        tokens_seen += static_cast<long long>(N);
        if (first_loss < 0.f) first_loss = loss;
        last_loss = loss;

        bool do_val = args.val_every > 0 &&
                      (step % args.val_every == 0 || step == args.max_steps);
        bool do_log = step % args.log_every == 0 || step == 1 ||
                      step == args.max_steps;
        bool do_save = args.save_every > 0 &&
                       (step % args.save_every == 0 || step == args.max_steps);

        if (do_log || do_val) {
            auto now = std::chrono::steady_clock::now();
            double elapsed =
                std::chrono::duration<double>(now - run_start).count();

            float val_loss = -1.f;
            if (do_val) val_loss = compute_val_loss();

            if (val_loss >= 0.f) {
                std::printf("  step %5d  loss %.4f  val %.4f  lr %.2e  gnorm %.3f  %5.2fs\n",
                             step, loss, val_loss, lr, gnorm, elapsed);
                log << "{\"step\":" << step
                    << ",\"loss\":" << loss
                    << ",\"val_loss\":" << val_loss
                    << ",\"lr\":" << lr
                    << ",\"grad_norm\":" << gnorm
                    << ",\"tokens_seen\":" << tokens_seen
                    << ",\"elapsed_sec\":" << elapsed << "}\n";
            } else {
                std::printf("  step %5d  loss %.4f  lr %.2e  gnorm %.3f  %5.2fs\n",
                             step, loss, lr, gnorm, elapsed);
                log << "{\"step\":" << step
                    << ",\"loss\":" << loss
                    << ",\"lr\":" << lr
                    << ",\"grad_norm\":" << gnorm
                    << ",\"tokens_seen\":" << tokens_seen
                    << ",\"elapsed_sec\":" << elapsed << "}\n";
            }
            log.flush();
        }

        if (do_save) {
            CheckpointInfo info;
            info.step = step;
            info.tokens_seen = tokens_seen;
            info.loss = loss;
            info.lr = lr;
            ensure_parent_dir(args.save_path);
            modernllm::save_checkpoint(args.save_path, named, opt, info);
            std::printf("    [checkpoint saved -> %s]\n", args.save_path.c_str());
        }
    }

    auto end = std::chrono::steady_clock::now();
    double total = std::chrono::duration<double>(end - run_start).count();
    std::printf("\ndone: %d steps in %.2fs (%.1f steps/s)\n",
                 step - start_step, total,
                 (step - start_step) / std::max(total, 1e-6));
    if (first_loss > 0.f) {
        std::printf("loss this run: %.4f -> %.4f (drop %.4f)\n",
                     first_loss, last_loss, first_loss - last_loss);
    }
    return 0;
}
