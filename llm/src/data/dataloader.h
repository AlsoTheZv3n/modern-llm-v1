#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "tokenizer/char_tokenizer.h"

namespace modernllm {

// Reads a flat text file, tokenizes it once with the supplied CharTokenizer,
// and serves random (B, T) batches for next-token prediction.
//
//   inputs[b, t]  = tokens[off_b + t]
//   targets[b, t] = tokens[off_b + t + 1]
//
// where off_b is sampled uniformly per batch element.
class TextDataset {
   public:
    void load(const std::string& path, const CharTokenizer& tok);

    // Load a flat int32-token binary file produced by
    // scripts/prepare_tokens.py. The corresponding `.meta` file is read
    // separately by callers (see data/token_meta.h).
    void load_binary(const std::string& path);

    // Sample one batch from the entire corpus.
    void sample_batch(int B, int T, std::mt19937_64& rng,
                       std::vector<std::int32_t>& inputs,
                       std::vector<std::int32_t>& targets) const;

    // Sample one batch from the half-open token range [begin, end).
    // Used for train/val splits without copying the underlying tokens.
    void sample_batch_range(int B, int T, std::mt19937_64& rng,
                              std::int64_t begin, std::int64_t end,
                              std::vector<std::int32_t>& inputs,
                              std::vector<std::int32_t>& targets) const;

    std::int64_t num_tokens() const noexcept {
        return static_cast<std::int64_t>(tokens_.size());
    }

   private:
    std::vector<std::int32_t> tokens_;
};

}  // namespace modernllm
