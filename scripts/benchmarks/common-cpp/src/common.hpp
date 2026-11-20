#ifndef __benchmarks_common_hpp__
#define __benchmarks_common_hpp__

#include <concepts>
#include <cstdint>
#include <cstdio>
#include <glaze/glaze.hpp>
#include <stdexcept>
#include <string>

#include "bench.hpp"
#include "memory_counters.h"

class InferenceEngine {
public:
    virtual ~InferenceEngine() = default;
    virtual ReadyResponse ready() const = 0;
    virtual BenchResponse execute(const BenchRequest& request) const = 0;
};

std::FILE* redirect_stdout_to_stderr();

memory_counters_t collect_memory_counters();
double collect_gpu_energy();
void validate_sampling(const BenchSampling& sampling);

// `speculative` is whether the engine decodes speculatively as configured.
void check_specdec(
    std::optional<bool> specdec,
    bool speculative
);

// What every engine's model.json has; engines' configurations add their own fields.
template <typename Config>
concept ModelConfig = requires(Config config) {
    { config.model } -> std::same_as<std::string&>;
    { config.specdec } -> std::same_as<std::optional<bool>&>;
    { config.sampling } -> std::same_as<std::optional<BenchSampling>&>;
    { config.reasoning_efforts } -> std::same_as<std::optional<ReasoningEfforts>&>;
    { config.model_size } -> std::same_as<std::optional<uint64_t>&>;
};

template <ModelConfig Config>
Config read_model_config(const std::string& path) {
    Config config;
    std::string buffer;
    if (const auto error = glz::read_file_json<glz::opts{.error_on_missing_keys = true}>(config, path, buffer)) {
        throw std::runtime_error(path + ": " + glz::format_error(error, buffer));
    }
    if (config.sampling) {
        validate_sampling(*config.sampling);
    }
    return config;
}

void run_loop(
    const InferenceEngine& engine,
    std::FILE* output
);

#endif  // __benchmarks_common_hpp__
