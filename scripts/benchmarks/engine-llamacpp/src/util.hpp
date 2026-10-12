#ifndef BENCHMARKS_LLAMACPP_UTIL_HPP
#define BENCHMARKS_LLAMACPP_UTIL_HPP

#include <llama-cpp.h>

#include <filesystem>
#include <string>

std::filesystem::path get_model_path(
    const std::string& model,
    bool dflash = false
);

#endif  // BENCHMARKS_LLAMACPP_UTIL_HPP
