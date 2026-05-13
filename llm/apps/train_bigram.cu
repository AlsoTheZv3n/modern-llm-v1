// Stage B.1 — Bigram language model on TinyShakespeare.
//
// Proves the end-to-end loop: data load → embedding fwd → softmax-CE →
// embedding bwd → AdamW. No attention, no FFN, no normalization yet.
//
// Initial loss should be ≈ log(vocab_size) ≈ 4.17 for the 65-char
// TinyShakespeare vocab. The bigram floor is around 2.4 — anything noticeably
// below the initial random loss is success for this stage.

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

#include "core/random.h"
#include "core/tensor.h"
#include "data/dataloader.h"
#include "model/embedding.h"
#include "tokenizer/char_tokenizer.h"
#include "train/adamw.h"
#include "train/loss.h"

using modernllm::AdamW;
using modernllm::AdamWConfig;
using modernllm::CharTokenizer;
using modernllm::Device;
using modernllm::DType;
using modernllm::Tensor;
using modernllm::TextDataset;

namespace {

struct Args {
    std::string data_path = "data/tinyshakespeare.txt";
    std::string log_path = "runs/bigram_log.jsonl";
    int batch_size = 64;
    int seq_len = 1;     // bigram: previous char predicts next
    int max_steps = 500;
    int log_every = 10;
    float lr = 1e-2f;
    float weight_decay = 0.0f;  // bigram = pure embedding lookup, no L2 needed
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
    // Use std::filesystem for portability.
    std::filesystem::create_directories(dir);
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

    // -----------------------------------------------------------------------
    // Tokenizer + data
    // -----------------------------------------------------------------------
    std::string text = read_file(args.data_path);
    CharTokenizer tok;
    tok.fit(text);
    int V = tok.vocab_size();

    TextDataset ds;
    ds.load(args.data_path, tok);
    std::printf("data: %s (%lld bytes, %lld tokens, vocab=%d)\n",
                args.data_path.c_str(),
                static_cast<long long>(text.size()),
                static_cast<long long>(ds.num_tokens()), V);

    // -----------------------------------------------------------------------
    // Model: a single weight tensor [V, V], used as both embedding table
    // and (transposed view of) LM head. For a bigram, embedding lookup
    // directly produces logits.
    // -----------------------------------------------------------------------
    Tensor weight({V, V}, DType::FP32, Device::Cuda);
    Tensor d_weight = Tensor::zeros({V, V}, DType::FP32, Device::Cuda);
    modernllm::normal_(weight, /*mean=*/0.0f, /*stddev=*/0.02f, args.seed);

    AdamWConfig cfg;
    cfg.lr = args.lr;
    cfg.weight_decay = args.weight_decay;
    AdamW opt(cfg);
    opt.add_param(&weight, &d_weight, args.weight_decay);

    // -----------------------------------------------------------------------
    // Persistent device buffers, sized for B*T
    // -----------------------------------------------------------------------
    int N = args.batch_size * args.seq_len;
    Tensor inputs_d({N}, DType::INT32, Device::Cuda);
    Tensor targets_d({N}, DType::INT32, Device::Cuda);
    Tensor logits_d({N, V}, DType::FP32, Device::Cuda);
    Tensor loss_pr({N}, DType::FP32, Device::Cuda);
    Tensor dlogits({N, V}, DType::FP32, Device::Cuda);

    Tensor inputs_h({N}, DType::INT32, Device::Host);
    Tensor targets_h({N}, DType::INT32, Device::Host);

    std::mt19937_64 rng(args.seed);
    std::vector<std::int32_t> in_buf, tgt_buf;

    // -----------------------------------------------------------------------
    // Logging
    // -----------------------------------------------------------------------
    ensure_parent_dir(args.log_path);
    std::ofstream log(args.log_path, std::ios::out);
    if (!log) {
        std::fprintf(stderr, "could not open log %s\n", args.log_path.c_str());
        return 3;
    }

    float first_loss = -1.f, last_loss = -1.f;
    auto run_start = std::chrono::steady_clock::now();

    // -----------------------------------------------------------------------
    // Training loop
    // -----------------------------------------------------------------------
    for (int step = 1; step <= args.max_steps; ++step) {
        ds.sample_batch(args.batch_size, args.seq_len, rng, in_buf, tgt_buf);
        std::memcpy(inputs_h.data(), in_buf.data(),
                    static_cast<std::size_t>(N) * sizeof(int));
        std::memcpy(targets_h.data(), tgt_buf.data(),
                    static_cast<std::size_t>(N) * sizeof(int));
        inputs_d.copy_from(inputs_h);
        targets_d.copy_from(targets_h);

        opt.zero_grad();

        // Forward: embedding lookup gives logits [N, V] directly.
        modernllm::embedding_forward(weight, inputs_d, logits_d);

        // Loss + dlogits.
        modernllm::softmax_ce_forward_backward(logits_d, targets_d,
                                                loss_pr, dlogits);
        float loss = modernllm::reduce_mean_to_scalar(loss_pr);

        // Backward through the embedding: scatter dlogits → d_weight.
        modernllm::embedding_backward(dlogits, inputs_d, d_weight);

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
                << ",\"elapsed_sec\":" << elapsed
                << "}\n";
            log.flush();
        }
    }

    auto end = std::chrono::steady_clock::now();
    double total = std::chrono::duration<double>(end - run_start).count();
    std::printf("\ndone: %d steps in %.2fs (%.0f steps/s)\n",
                 args.max_steps, total,
                 args.max_steps / std::max(total, 1e-6));
    std::printf("loss: %.4f → %.4f (drop %.4f)\n",
                 first_loss, last_loss, first_loss - last_loss);

    if (last_loss >= first_loss) {
        std::fprintf(stderr, "WARNING: loss did not decrease.\n");
        return 1;
    }
    return 0;
}
