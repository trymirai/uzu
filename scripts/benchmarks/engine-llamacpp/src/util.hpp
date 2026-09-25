#ifndef BENCHMARKS_LLAMACPP_UTIL_HPP
#define BENCHMARKS_LLAMACPP_UTIL_HPP

#include <llama-cpp.h>

#include <filesystem>
#include <string>

#include "memory_counters.h"

memory_counters_t collect_memory_counters();

std::filesystem::path get_model_path(const std::string& model);

#endif  // BENCHMARKS_LLAMACPP_UTIL_HPP
