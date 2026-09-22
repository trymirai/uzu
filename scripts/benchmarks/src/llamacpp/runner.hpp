#ifndef __llama_cpp_benchmarks_runner_hpp__
#define __llama_cpp_benchmarks_runner_hpp__

#include <cstdint>
#include <optional>

#include "content.hpp"
#include "memory_counters.h"

struct SamplingConfig {
    std::optional<int32_t> top_k;
    std::optional<float> top_p;
    std::optional<float> min_p;
    std::optional<float> temp;
};

struct RunRequest {
    std::string model;
    Content input;
    std::optional<size_t> max_tokens = std::nullopt;
    std::optional<size_t> speculative_depth = std::nullopt;
    std::optional<SamplingConfig> sampling = std::nullopt;
};

struct RunResponse {
    std::string text;
    double time_to_first_token;
    double prompt_tps;
    double generation_tps;
    double tokens_per_fp;
    double duration;
    memory_counters_t memory_counters;
};

RunResponse run(const RunRequest& request);

#endif  // __llama_cpp_benchmarks_runner_hpp__
