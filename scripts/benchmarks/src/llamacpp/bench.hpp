#ifndef __llama_cpp_benchmarks_bench_hpp__
#define __llama_cpp_benchmarks_bench_hpp__

#include <cstdint>
#include <glaze/json/generic.hpp>
#include <optional>
#include <string>
#include <vector>

struct ChatMessage {
    std::string role;
    std::optional<std::string> content = std::nullopt;
    std::optional<std::string> reasoning_content = std::nullopt;
    std::optional<std::vector<glz::generic_u64>> tool_calls = std::nullopt;
    std::optional<std::string> tool_call_id = std::nullopt;
};

struct BenchSampling {
    std::optional<int32_t> top_k = std::nullopt;
    std::optional<float> top_p = std::nullopt;
    std::optional<float> min_p = std::nullopt;
    std::optional<float> temp = std::nullopt;
};

struct BenchRequest {
    std::optional<std::string> prompt_text = std::nullopt;
    std::optional<std::vector<ChatMessage>> prompt_chat = std::nullopt;
    std::optional<std::vector<glz::generic_u64>> tools = std::nullopt;
    std::optional<glz::generic_u64> tool_choice = std::nullopt;

    std::optional<size_t> max_tokens = std::nullopt;
    std::optional<size_t> speculative_depth = std::nullopt;
    std::optional<BenchSampling> sampling = std::nullopt;
    std::optional<size_t> num_runs = std::nullopt;
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
