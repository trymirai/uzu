#ifndef __llama_cpp_benchmarks_runner_hpp__
#define __llama_cpp_benchmarks_runner_hpp__

#include <string>
#include <vector>

#include "bench.hpp"

std::vector<BenchResponse> run(
    const std::string& input_model_path,
    const BenchRequest& request
);

#endif  // __llama_cpp_benchmarks_runner_hpp__
