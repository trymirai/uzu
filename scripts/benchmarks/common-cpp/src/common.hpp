#ifndef __benchmarks_common_hpp__
#define __benchmarks_common_hpp__

#include <cstdio>
#include <vector>

#include "bench.hpp"

class InferenceEngine {
public:
    virtual ~InferenceEngine() = default;
    virtual std::vector<BenchResponse> execute(const BenchRequest& request) const = 0;
};

std::FILE* redirect_stdout_to_stderr();

void run_loop(
    const InferenceEngine& engine,
    std::FILE* output
);

#endif  // __benchmarks_common_hpp__
