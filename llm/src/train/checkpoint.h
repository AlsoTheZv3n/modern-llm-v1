#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "model/modern_gpt.h"
#include "train/adamw.h"

namespace modernllm {

// Per-checkpoint metadata recorded alongside model + optimizer state.
struct CheckpointInfo {
    int step = 0;
    long long tokens_seen = 0;
    float loss = 0.f;
    float lr = 0.f;
};

// Magic number "MLLM" little-endian.
constexpr std::uint32_t kCkptMagic = 0x4D4C4C4D;
constexpr std::uint32_t kCkptVersion = 1;

// Save weights + AdamW (m, v, step) state to a single binary file.
// The order of `params` is the wire-order; load matches by name.
void save_checkpoint(const std::string& path,
                      const std::vector<NamedParam>& params,
                      const AdamW& opt,
                      const CheckpointInfo& info);

// Load a checkpoint into an already-allocated model and optimizer.
// Both must have been initialized with the SAME shapes/registration order
// as when the checkpoint was saved (we additionally verify per-tensor name
// + shape on load).
//
// Returns false if file does not exist; throws on format / shape errors.
bool load_checkpoint(const std::string& path,
                      const std::vector<NamedParam>& params,
                      AdamW& opt,
                      CheckpointInfo& info);

}  // namespace modernllm
