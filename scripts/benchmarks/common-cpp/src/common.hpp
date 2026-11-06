#ifndef __benchmarks_common_hpp__
#define __benchmarks_common_hpp__

#include <cstdint>
#include <cstdio>
#include <vector>

#include "bench.hpp"

#ifdef __APPLE__
#include "memory_counters.h"
#else
// The counters Linux reads from /proc/self/status; engines report their own GPU memory there.
struct memory_counters_t {
    uint64_t phys_footprint;
    uint64_t resident_size;
    uint64_t graphics_total;
};
#endif

class InferenceEngine {
public:
    virtual ~InferenceEngine() = default;
    virtual std::vector<BenchResponse> execute(const BenchRequest& request) const = 0;

    // Engines that render a request's reasoning_effort opt in; run_loop rejects it for the rest.
    virtual bool supports_reasoning_effort() const {
        return false;
    }
};

std::FILE* redirect_stdout_to_stderr();

memory_counters_t collect_memory_counters();

void run_loop(
    const InferenceEngine& engine,
    std::FILE* output
);

#endif  // __benchmarks_common_hpp__
