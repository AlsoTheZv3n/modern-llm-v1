// Load a ModernGPT checkpoint and generate text. Two prompt modes:
//
//   --prompt  STR             char-level: tokenize STR with the corpus
//                              (CharTokenizer rebuilt from --data); decode
//                              output the same way.
//   --prompt-tokens "1,2,3"   pre-tokenized: skip the C++ tokenizer entirely.
//                              Vocab size comes from --meta PATH.meta.
//                              When combined with --output-tokens-only, the
//                              app emits space-separated int IDs to stdout
//                              (and only those, plus a trailing newline) so a
//                              Python wrapper can detokenize.
//
// No KV cache yet — full T-context forward per generated token.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/gemm.h"
#include "core/tensor.h"
#include "data/token_meta.h"
#include "model/modern_gpt.h"
#include "tokenizer/char_tokenizer.h"
#include "train/adamw.h"
#include "train/checkpoint.h"

using modernllm::AdamW;
using modernllm::AdamWConfig;
using modernllm::CharTokenizer;
using modernllm::CheckpointInfo;
using modernllm::CublasHandle;
using modernllm::Device;
using modernllm::DType;
using modernllm::ModernGPT;
using modernllm::Tensor;

namespace {

struct Args {
    std::string data_path = "";       // char mode: corpus to fit tokenizer on
    std::string meta_path = "";       // bpe mode: vocab_size source
    std::string ckpt_path = "runs/modern_gpt.ckpt";
    std::string prompt = "";          // char mode: string prompt
    std::string prompt_tokens = "";   // bpe mode: comma-sep int IDs
    int num_tokens = 200;
    int seq_len = 64;
    int d_model = 128;
    int n_heads = 4;
    int n_kv_heads = 0;          // 0 = MHA (n_kv = n_heads)
    int n_layers = 4;
    int d_ffn = 512;
    float rope_base = 10000.f;
    float temperature = 0.8f;
    unsigned long long seed = 42ULL;
    bool output_tokens_only = false;
};

void parse_args(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + k);
            return argv[++i];
        };
        if      (k == "--data")              a.data_path = next();
        else if (k == "--meta")              a.meta_path = next();
        else if (k == "--ckpt")              a.ckpt_path = next();
        else if (k == "--prompt")            a.prompt = next();
        else if (k == "--prompt-tokens")     a.prompt_tokens = next();
        else if (k == "--num")               a.num_tokens = std::stoi(next());
        else if (k == "--seq-len")           a.seq_len = std::stoi(next());
        else if (k == "--d-model")           a.d_model = std::stoi(next());
        else if (k == "--n-heads")           a.n_heads = std::stoi(next());
        else if (k == "--n-kv-heads")        a.n_kv_heads = std::stoi(next());
        else if (k == "--n-layers")          a.n_layers = std::stoi(next());
        else if (k == "--d-ffn")             a.d_ffn = std::stoi(next());
        else if (k == "--temp")              a.temperature = std::stof(next());
        else if (k == "--seed")              a.seed = std::stoull(next());
        else if (k == "--output-tokens-only") a.output_tokens_only = true;
        else throw std::runtime_error("unknown arg: " + k);
    }
}

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::ostringstream ss; ss << in.rdbuf(); return ss.str();
}

std::vector<std::int32_t> parse_csv_ints(const std::string& s) {
    std::vector<std::int32_t> out;
    std::string cur;
    for (char c : s) {
        if (c == ',' || c == ' ' || c == '\t') {
            if (!cur.empty()) { out.push_back(std::stoi(cur)); cur.clear(); }
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) out.push_back(std::stoi(cur));
    return out;
}

int sample_next_token(const float* row, int V, float temperature,
                       std::mt19937_64& rng) {
    if (temperature <= 0.f) {
        int best = 0;
        float best_v = row[0];
        for (int v = 1; v < V; ++v) {
            if (row[v] > best_v) { best_v = row[v]; best = v; }
        }
        return best;
    }
    float row_max = row[0];
    for (int v = 1; v < V; ++v) row_max = std::fmax(row_max, row[v]);
    std::vector<double> probs(V);
    double sum = 0.0;
    for (int v = 0; v < V; ++v) {
        probs[v] = std::exp((row[v] - row_max) / temperature);
        sum += probs[v];
    }
    for (auto& p : probs) p /= sum;

    std::uniform_real_distribution<double> u(0.0, 1.0);
    double r = u(rng);
    double acc = 0.0;
    for (int v = 0; v < V; ++v) {
        acc += probs[v];
        if (r <= acc) return v;
    }
    return V - 1;
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    try { parse_args(argc, argv, args); }
    catch (std::exception& e) {
        std::fprintf(stderr, "args: %s\n", e.what());
        return 2;
    }

    bool bpe_mode = !args.meta_path.empty() || !args.prompt_tokens.empty();

    int V = 0;
    CharTokenizer char_tok;
    std::vector<std::int32_t> tokens;

    if (bpe_mode) {
        if (args.meta_path.empty()) {
            std::fprintf(stderr,
                "BPE mode requires --meta PATH.meta for vocab_size\n");
            return 2;
        }
        modernllm::TokenMeta meta = modernllm::read_token_meta(args.meta_path);
        V = meta.vocab_size;

        if (!args.prompt_tokens.empty()) {
            tokens = parse_csv_ints(args.prompt_tokens);
        } else {
            std::fprintf(stderr,
                "BPE mode requires --prompt-tokens (use scripts/sample.py "
                "for the friendly Python wrapper)\n");
            return 2;
        }
    } else {
        if (args.data_path.empty()) args.data_path = "data/tinyshakespeare.txt";
        std::string text = read_file(args.data_path);
        char_tok.fit(text);
        V = char_tok.vocab_size();
        if (args.prompt.empty()) args.prompt = "ROMEO:";
        try {
            tokens = char_tok.encode(args.prompt);
        } catch (std::exception& e) {
            std::fprintf(stderr, "encode failed: %s\n", e.what());
            return 3;
        }
    }

    if (tokens.empty()) tokens.push_back(0);

    // Build the model with B=1 and T from args; T must match training.
    ModernGPT::Config cfg;
    cfg.B = 1;
    cfg.T = args.seq_len;
    cfg.V = V;
    cfg.D = args.d_model;
    cfg.H = args.n_heads;
    cfg.n_kv_heads = args.n_kv_heads;
    cfg.Dff = args.d_ffn;
    cfg.n_layers = args.n_layers;
    cfg.rope_base = args.rope_base;

    ModernGPT model;
    model.allocate(cfg);
    model.init_random(0);  // overwritten by checkpoint

    AdamWConfig opt_cfg; opt_cfg.lr = 0.f;
    AdamW opt(opt_cfg);
    model.register_with_optimizer(opt, /*wd=*/0.f);
    auto named = model.named_params();

    CheckpointInfo info;
    if (!modernllm::load_checkpoint(args.ckpt_path, named, opt, info)) {
        std::fprintf(stderr, "checkpoint not found: %s\n",
                      args.ckpt_path.c_str());
        return 4;
    }
    if (!args.output_tokens_only) {
        std::fprintf(stderr, "loaded %s (step=%d, loss=%.4f, vocab=%d)\n",
                     args.ckpt_path.c_str(), info.step, info.loss, V);
    }

    CublasHandle handle;
    int T = cfg.T;

    std::vector<std::int32_t> window(T, tokens[0]);
    int valid_len = 0;
    int prompt_taken = static_cast<int>(std::min<size_t>(tokens.size(), T));
    int prompt_offset = static_cast<int>(tokens.size()) - prompt_taken;
    for (int i = 0; i < prompt_taken; ++i) {
        window[i] = tokens[prompt_offset + i];
    }
    valid_len = prompt_taken;

    Tensor in_h({T}, DType::INT32, Device::Host);
    Tensor in_d({T}, DType::INT32, Device::Cuda);

    auto upload_window = [&]() {
        std::memcpy(in_h.data(), window.data(), T * sizeof(int));
        in_d.copy_from(in_h);
    };

    std::mt19937_64 rng(args.seed);
    std::vector<std::int32_t> generated;
    generated.reserve(args.num_tokens);

    // Char mode echoes the prompt + decoded tokens to stdout immediately.
    // Token-only mode emits IDs at the very end (one line, space-separated).
    if (!args.output_tokens_only && !bpe_mode) {
        std::printf("%s", args.prompt.c_str());
        std::fflush(stdout);
    }

    for (int step = 0; step < args.num_tokens; ++step) {
        upload_window();
        model.forward(in_d, /*target_ids=*/nullptr, handle);

        int pos = std::min(valid_len - 1, T - 1);
        Tensor logits_h = model.logits().to(Device::Host);
        const float* row = logits_h.data_as<float>() +
                            static_cast<long long>(pos) * V;
        int next = sample_next_token(row, V, args.temperature, rng);

        if (!args.output_tokens_only && !bpe_mode) {
            std::string c = char_tok.decode({next});
            std::printf("%s", c.c_str());
            std::fflush(stdout);
        }
        generated.push_back(next);

        if (valid_len < T) {
            window[valid_len] = next;
            ++valid_len;
        } else {
            std::memmove(window.data(), window.data() + 1,
                          (T - 1) * sizeof(std::int32_t));
            window[T - 1] = next;
        }
    }

    if (args.output_tokens_only) {
        // One line, space-separated, easy to parse from Python.
        for (size_t i = 0; i < generated.size(); ++i) {
            std::printf("%s%d", (i == 0 ? "" : " "), generated[i]);
        }
        std::printf("\n");
    } else if (!bpe_mode) {
        std::printf("\n");
    } else {
        // bpe_mode without output-tokens-only: also dump IDs to stderr so
        // the user sees something meaningful even without the wrapper.
        std::fprintf(stderr, "\ngenerated %d token IDs (use sample.py to decode)\n",
                      (int)generated.size());
        for (size_t i = 0; i < generated.size(); ++i) {
            std::printf("%s%d", (i == 0 ? "" : " "), generated[i]);
        }
        std::printf("\n");
    }
    return 0;
}
