#ifndef __engine_mldrift_engine_hpp__
#define __engine_mldrift_engine_hpp__

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "common.hpp"
#include "ml_drift/samples/llm/llm_config.h"
#include "ml_drift/samples/llm/llm_runner.h"

class MlDriftEngine final : public InferenceEngine {
public:
    MlDriftEngine(
        const std::filesystem::path& checkpoint,
        const std::filesystem::path& weights
    );

    BenchResponse execute(const BenchRequest& request) const override;

private:
    std::string weights;
    ml_drift::ModelInfo model_info;
    size_t context_capacity;
    std::unique_ptr<ml_drift::Tokenizer> tokenizer;
    mutable std::unique_ptr<ml_drift::LlmModelRunner> runner;
    mutable size_t runner_prompt_size = 0;
    mutable size_t runner_context_size = 0;
};

#endif  // __engine_mldrift_engine_hpp__
