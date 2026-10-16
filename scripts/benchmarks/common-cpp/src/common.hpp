#ifndef __benchmarks_common_hpp__
#define __benchmarks_common_hpp__

#include <cstdio>
#include <vector>

#include "bench.hpp"
#include "memory_counters.h"

class InferenceEngine {
public:
    virtual ~InferenceEngine() = default;
    virtual std::vector<BenchResponse> execute(const BenchRequest& request) const = 0;
};

std::FILE* redirect_stdout_to_stderr();

memory_counters_t collect_memory_counters();

void run_loop(
    const InferenceEngine& engine,
    std::FILE* output
);

#endif  // __benchmarks_common_hpp__
