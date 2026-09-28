#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace benchmark {

void checkDevice();

// Own the production runtime in the Python process. Protocol frames cross an
// in-memory buffer; inference, scheduling, caches and metrics stay upstream.
class Engine final {
public:
    Engine(
        const std::string& modelRoot,
        const std::string& metallibPath
    );
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    std::vector<uint8_t> receive(std::span<const uint8_t> input);
    std::vector<uint8_t> step(double timeoutSeconds);
    std::string status();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace benchmark
