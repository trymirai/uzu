#ifndef __llama_cpp_benchmarks_bench_hpp__
#define __llama_cpp_benchmarks_bench_hpp__

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

struct ChatMessage {
    std::string message;
    std::string role;
};

struct BenchSampling {
    std::optional<int32_t> top_k;
    std::optional<float> top_p;
    std::optional<float> min_p;
    std::optional<float> temp;
};

struct BenchRequest {
    std::optional<std::string> prompt_text = std::nullopt;
    std::optional<std::vector<ChatMessage>> prompt_chat = std::nullopt;
    std::optional<size_t> max_tokens = std::nullopt;
    std::optional<size_t> speculative_depth = std::nullopt;
    std::optional<BenchSampling> sampling = std::nullopt;
};

struct BenchResponse {
    std::string text;
    double time_to_first_token;
    double prompt_tps;
    double decode_tps;
    double tokens_per_forward_pass;
    double duration;
    uint64_t memory_phys_footprint;
    uint64_t memory_resident_peak;
    uint64_t memory_graphics_total;
};

#endif  // __llama_cpp_benchmarks_bench_hpp__
