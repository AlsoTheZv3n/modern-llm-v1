#include "tokenizer/char_tokenizer.h"

#include <algorithm>
#include <stdexcept>

namespace modernllm {

void CharTokenizer::fit(const std::string& text) {
    std::array<bool, 256> seen{};
    seen.fill(false);
    for (unsigned char c : text) seen[c] = true;

    id_to_byte_.clear();
    byte_to_id_.fill(-1);
    for (int b = 0; b < 256; ++b) {
        if (seen[b]) {
            byte_to_id_[b] = static_cast<std::int32_t>(id_to_byte_.size());
            id_to_byte_.push_back(static_cast<std::uint8_t>(b));
        }
    }
}

std::vector<std::int32_t> CharTokenizer::encode(const std::string& s) const {
    std::vector<std::int32_t> out;
    out.reserve(s.size());
    for (unsigned char c : s) {
        std::int32_t id = byte_to_id_[c];
        if (id < 0) {
            throw std::runtime_error(
                "CharTokenizer::encode: byte not in vocab");
        }
        out.push_back(id);
    }
    return out;
}

std::string CharTokenizer::decode(const std::vector<std::int32_t>& ids) const {
    std::string out;
    out.reserve(ids.size());
    for (auto id : ids) {
        if (id < 0 || id >= static_cast<std::int32_t>(id_to_byte_.size())) {
            throw std::runtime_error(
                "CharTokenizer::decode: id out of range");
        }
        out.push_back(static_cast<char>(id_to_byte_[id]));
    }
    return out;
}

}  // namespace modernllm
