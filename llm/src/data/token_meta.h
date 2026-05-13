#pragma once

#include <cstdint>
#include <string>

namespace modernllm {

// Companion .meta file written by scripts/prepare_tokens.py:
//   encoding=cl100k_base
//   vocab_size=100277
//   num_tokens=301832
//   source=...
//   max_tokens=...
struct TokenMeta {
    std::string encoding;
    int vocab_size = 0;
    long long num_tokens = 0;
    std::string source;
};

// Reads the .meta file companion to a .bin file. Throws on missing/malformed.
TokenMeta read_token_meta(const std::string& meta_path);

}  // namespace modernllm
