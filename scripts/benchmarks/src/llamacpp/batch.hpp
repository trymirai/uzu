#ifndef __llama_cpp_benchmarks_batch_hpp__
#define __llama_cpp_benchmarks_batch_hpp__

#include <llama-cpp.h>

#include <memory>

struct llama_batch_deleter {
    void operator()(llama_batch* batch) {
        llama_batch_free(*batch);
    }
};

typedef std::unique_ptr<llama_batch, llama_batch_deleter> llama_batch_ptr;

#endif  // __llama_cpp_benchmarks_batch_hpp__
