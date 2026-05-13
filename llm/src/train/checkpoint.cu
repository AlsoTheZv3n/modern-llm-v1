#include "train/checkpoint.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace modernllm {

namespace {

// Tiny binary writer / reader. Little-endian assumed (we're on x86-64).
template <typename T>
void write_pod(std::ofstream& out, const T& v) {
    out.write(reinterpret_cast<const char*>(&v), sizeof(T));
}
template <typename T>
T read_pod(std::ifstream& in) {
    T v;
    in.read(reinterpret_cast<char*>(&v), sizeof(T));
    return v;
}

void write_str(std::ofstream& out, const std::string& s) {
    std::uint32_t n = static_cast<std::uint32_t>(s.size());
    write_pod(out, n);
    out.write(s.data(), n);
}
std::string read_str(std::ifstream& in) {
    std::uint32_t n = read_pod<std::uint32_t>(in);
    std::string s(n, '\0');
    in.read(s.data(), n);
    return s;
}

void write_tensor(std::ofstream& out, const Tensor& t) {
    std::uint32_t ndim = static_cast<std::uint32_t>(t.shape().size());
    write_pod(out, ndim);
    for (auto d : t.shape()) {
        std::int64_t dim = d;
        write_pod(out, dim);
    }
    std::uint8_t dtype_byte = static_cast<std::uint8_t>(t.dtype());
    write_pod(out, dtype_byte);

    // Bring to host and write raw bytes.
    Tensor h = t.to(Device::Host);
    out.write(reinterpret_cast<const char*>(h.data()), t.bytes());
}

void read_tensor_into(std::ifstream& in, Tensor& dst,
                       const std::string& name) {
    std::uint32_t ndim = read_pod<std::uint32_t>(in);
    std::vector<std::int64_t> shape(ndim);
    for (auto& d : shape) d = read_pod<std::int64_t>(in);
    std::uint8_t dtype_byte = read_pod<std::uint8_t>(in);
    DType dtype = static_cast<DType>(dtype_byte);

    if (shape != dst.shape()) {
        throw std::runtime_error("checkpoint: shape mismatch for '" + name +
                                  "'");
    }
    if (dtype != dst.dtype()) {
        throw std::runtime_error("checkpoint: dtype mismatch for '" + name +
                                  "'");
    }

    Tensor h(shape, dtype, Device::Host);
    in.read(reinterpret_cast<char*>(h.data()), h.bytes());
    dst.copy_from(h);
}

}  // namespace

void save_checkpoint(const std::string& path,
                      const std::vector<NamedParam>& params,
                      const AdamW& opt,
                      const CheckpointInfo& info) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("save_checkpoint: cannot open " + path);

    // Header
    write_pod(out, kCkptMagic);
    write_pod(out, kCkptVersion);
    write_pod<std::int32_t>(out, info.step);
    write_pod<std::int64_t>(out, info.tokens_seen);
    write_pod<float>(out, info.loss);
    write_pod<float>(out, info.lr);
    write_pod<std::int32_t>(out, opt.step_count());

    // Build pointer→AdamWParam lookup for matching m/v.
    std::unordered_map<const Tensor*, const AdamWParam*> opt_lookup;
    for (auto& slot : opt.params()) {
        opt_lookup[slot.param] = &slot;
    }

    std::uint32_t num = static_cast<std::uint32_t>(params.size());
    write_pod(out, num);

    for (auto& np : params) {
        write_str(out, np.name);

        auto it = opt_lookup.find(np.param);
        std::uint8_t has_opt = (it != opt_lookup.end()) ? 1 : 0;
        write_pod(out, has_opt);

        write_tensor(out, *np.param);
        if (has_opt) {
            write_tensor(out, it->second->m);
            write_tensor(out, it->second->v);
        }
    }
}

bool load_checkpoint(const std::string& path,
                      const std::vector<NamedParam>& params,
                      AdamW& opt,
                      CheckpointInfo& info) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;

    auto magic = read_pod<std::uint32_t>(in);
    if (magic != kCkptMagic) {
        throw std::runtime_error("load_checkpoint: bad magic in " + path);
    }
    auto ver = read_pod<std::uint32_t>(in);
    if (ver != kCkptVersion) {
        throw std::runtime_error("load_checkpoint: unsupported version");
    }
    info.step = read_pod<std::int32_t>(in);
    info.tokens_seen = read_pod<std::int64_t>(in);
    info.loss = read_pod<float>(in);
    info.lr = read_pod<float>(in);
    auto opt_step = read_pod<std::int32_t>(in);
    opt.set_step_count(opt_step);

    auto num = read_pod<std::uint32_t>(in);

    // Build lookups: name -> param for the live model and pointer -> slot
    // for the live optimizer, so we can match by name and route m/v correctly.
    std::unordered_map<std::string, const NamedParam*> by_name;
    for (auto& np : params) by_name[np.name] = &np;

    std::unordered_map<const Tensor*, AdamWParam*> opt_lookup;
    for (auto& slot : opt.params()) opt_lookup[slot.param] = &slot;

    for (std::uint32_t i = 0; i < num; ++i) {
        std::string name = read_str(in);
        auto has_opt = read_pod<std::uint8_t>(in);

        auto it = by_name.find(name);
        if (it == by_name.end()) {
            throw std::runtime_error(
                "load_checkpoint: unknown tensor '" + name + "'");
        }
        const NamedParam& np = *it->second;
        read_tensor_into(in, *np.param, name);

        if (has_opt) {
            auto opt_it = opt_lookup.find(np.param);
            if (opt_it == opt_lookup.end()) {
                throw std::runtime_error(
                    "load_checkpoint: '" + name +
                    "' has saved optimizer state but is not registered");
            }
            read_tensor_into(in, opt_it->second->m, name + ".m");
            read_tensor_into(in, opt_it->second->v, name + ".v");
        }
    }

    if (!in) throw std::runtime_error("load_checkpoint: read error in " + path);
    return true;
}

}  // namespace modernllm
