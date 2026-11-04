#ifndef __engine_llamacpp_engine_hpp__
#define __engine_llamacpp_engine_hpp__

#include <chat.h>
#include <llama-cpp.h>

#include <string>
#include <vector>

#include "common.hpp"

class LlamaEngine final : public InferenceEngine {
public:
    explicit LlamaEngine(
        const std::string& model,
        const std::optional<BenchSampling>& sampling,
        const std::string& draft_model = "",
        size_t speculative_depth = 0
    );

    ReadyResponse ready() const override;
    BenchResponse execute(const BenchRequest& request) const override;

private:
    struct RunConfig;

    // Sets the effort in the template's inputs, and returns whether the template honors it.
    bool apply_reasoning_effort(
        common_chat_templates_inputs& inputs,
        ReasoningEffort effort
    ) const;

    std::vector<llama_token> tokenize(const BenchRequest& request) const;

    BenchResponse run_single(
        std::vector<llama_token> tokens,
        const RunConfig& config,
        const llama_sampler_ptr& sampler_chain
    ) const;

    llama_model_ptr model;
    llama_model_ptr draft_model;
    size_t dflash_depth = 0;
    size_t speculative_depth;
    const llama_vocab* tokenizer;
    common_chat_templates_ptr chat_templates;
    bool has_mtp;
    BenchSampling sampling;
};

std::string decode_token(
    const llama_vocab* vocab,
    llama_token token
);

bool has_mtp_weights(const std::filesystem::path& model_path);

#endif  // __engine_llamacpp_engine_hpp__
