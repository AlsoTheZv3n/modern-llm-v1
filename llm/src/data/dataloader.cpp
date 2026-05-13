#include "data/dataloader.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace modernllm {

void TextDataset::load(const std::string& path, const CharTokenizer& tok) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("TextDataset::load: cannot open " + path);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    std::string text = ss.str();
    tokens_ = tok.encode(text);
    if (tokens_.size() < 2) {
        throw std::runtime_error("TextDataset::load: corpus too small");
    }
}

void TextDataset::load_binary(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error(
            "TextDataset::load_binary: cannot open " + path);
    }
    std::streamsize sz = in.tellg();
    if (sz < 0 || sz % static_cast<std::streamsize>(sizeof(std::int32_t)) != 0) {
        throw std::runtime_error(
            "TextDataset::load_binary: bad size for " + path);
    }
    in.seekg(0, std::ios::beg);
    std::size_t n = static_cast<std::size_t>(sz / sizeof(std::int32_t));
    tokens_.resize(n);
    in.read(reinterpret_cast<char*>(tokens_.data()), sz);
    if (!in) {
        throw std::runtime_error(
            "TextDataset::load_binary: short read on " + path);
    }
    if (tokens_.size() < 2) {
        throw std::runtime_error(
            "TextDataset::load_binary: corpus too small");
    }
}

void TextDataset::sample_batch(int B, int T, std::mt19937_64& rng,
                                std::vector<std::int32_t>& inputs,
                                std::vector<std::int32_t>& targets) const {
    sample_batch_range(B, T, rng, 0, num_tokens(), inputs, targets);
}

void TextDataset::sample_batch_range(int B, int T, std::mt19937_64& rng,
                                      std::int64_t begin, std::int64_t end,
                                      std::vector<std::int32_t>& inputs,
                                      std::vector<std::int32_t>& targets) const {
    if (B <= 0 || T <= 0) {
        throw std::invalid_argument("sample_batch: B and T must be positive");
    }
    if (begin < 0 || end > num_tokens() || begin >= end) {
        throw std::invalid_argument("sample_batch: invalid range");
    }
    std::int64_t span = end - begin;
    if (static_cast<std::int64_t>(T) + 1 > span) {
        throw std::runtime_error("sample_batch: T+1 > range size");
    }
    inputs.assign(static_cast<std::size_t>(B) * T, 0);
    targets.assign(static_cast<std::size_t>(B) * T, 0);

    std::int64_t max_off = span - T - 1;
    std::uniform_int_distribution<std::int64_t> off_dist(0, max_off);

    for (int b = 0; b < B; ++b) {
        std::int64_t off = begin + off_dist(rng);
        for (int t = 0; t < T; ++t) {
            inputs[static_cast<std::size_t>(b) * T + t] =
                tokens_[off + t];
            targets[static_cast<std::size_t>(b) * T + t] =
                tokens_[off + t + 1];
        }
    }
}

}  // namespace modernllm
