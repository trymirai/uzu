#ifndef __llama_cpp_benchmarks_runner_hpp__
#define __llama_cpp_benchmarks_runner_hpp__

#include <cstdint>
#include <optional>

#include "content.hpp"
#include "memory_counters.h"

struct SamplingConfig {
    int32_t top_k = 0;
    float top_p = 1.0f;
    float min_p = 0.0f;
    float temp = 1.0f;
};

struct RunRequest {
    std::string model;
    Content input;
    size_t max_tokens;
    std::optional<SamplingConfig> sampling;
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