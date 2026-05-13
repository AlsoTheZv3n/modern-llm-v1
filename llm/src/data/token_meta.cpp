#include "data/token_meta.h"

#include <fstream>
#include <stdexcept>

namespace modernllm {

namespace {

std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\t' ||
                      s[a] == '\r' || s[a] == '\n')) ++a;
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' ||
                      s[b - 1] == '\r' || s[b - 1] == '\n')) --b;
    return s.substr(a, b - a);
}

}  // namespace

TokenMeta read_token_meta(const std::string& meta_path) {
    std::ifstream in(meta_path);
    if (!in) {
        throw std::runtime_error("read_token_meta: cannot open " + meta_path);
    }
    TokenMeta m;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') continue;
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = trim(line.substr(0, eq));
        std::string val = trim(line.substr(eq + 1));
        if (key == "encoding") m.encoding = val;
        else if (key == "vocab_size") m.vocab_size = std::stoi(val);
        else if (key == "num_tokens") m.num_tokens = std::stoll(val);
        else if (key == "source") m.source = val;
    }
    if (m.vocab_size <= 0) {
        throw std::runtime_error(
            "read_token_meta: missing/invalid vocab_size in " + meta_path);
    }
    return m;
}

}  // namespace modernllm
