#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace modernllm {

// Character-level tokenizer. Vocab is the set of unique bytes in the
// fitting corpus. Used for proof-of-life training before BPE.
class CharTokenizer {
   public:
    CharTokenizer() = default;

    // Build vocab from the unique bytes in `text`. Sorted for determinism.
    void fit(const std::string& text);

    int vocab_size() const noexcept {
        return static_cast<int>(id_to_byte_.size());
    }

    std::vector<std::int32_t> encode(const std::string& s) const;
    std::string decode(const std::vector<std::int32_t>& ids) const;

   private:
    // 256-entry table, byte -> token id (-1 for unseen bytes)
    std::array<std::int32_t, 256> byte_to_id_{};
    std::vector<std::uint8_t> id_to_byte_;
};

}  // namespace modernllm
