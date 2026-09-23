#ifndef __llama_cpp_benchmarks_common_hpp__
#define __llama_cpp_benchmarks_common_hpp__

#include <llama-cpp.h>

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "bench.hpp"
#include "memory_counters.h"

bool has_mtp_weights(
    const std::filesystem::path& model_path,
    const llama_model* model
);

memory_counters_t collect_memory_counters();

std::string decode_token(
    const llama_vocab* vocab,
    llama_token token
);

std::filesystem::path get_model_path(const std::string& model);

std::vector<llama_token> get_tokens(
    const BenchRequest& request,
    const llama_model_ptr& model
);

#endif  // __llama_cpp_benchmarks_common_hpp__
