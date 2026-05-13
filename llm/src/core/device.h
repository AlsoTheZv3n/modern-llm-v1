#pragma once

namespace modernllm {

enum class Device {
    Host,
    Cuda,
};

inline const char* device_name(Device d) {
    switch (d) {
        case Device::Host: return "host";
        case Device::Cuda: return "cuda";
    }
    return "?";
}

}  // namespace modernllm
