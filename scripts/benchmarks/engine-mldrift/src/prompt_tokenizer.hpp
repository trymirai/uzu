#ifndef __engine_mldrift_prompt_tokenizer_hpp__
#define __engine_mldrift_prompt_tokenizer_hpp__

#include <sys/types.h>

#include <cstdio>
#include <filesystem>
#include <vector>

#include "bench.hpp"

class PromptTokenizer final {
public:
    PromptTokenizer(
        const std::filesystem::path& project,
        const std::filesystem::path& checkpoint
    );
    PromptTokenizer(const PromptTokenizer&) = delete;
    PromptTokenizer& operator=(const PromptTokenizer&) = delete;
    ~PromptTokenizer();

    std::vector<int> tokenize(const BenchRequest& request) const;

private:
    pid_t pid;
    std::FILE* input;
    std::FILE* output;
};

#endif  // __engine_mldrift_prompt_tokenizer_hpp__
