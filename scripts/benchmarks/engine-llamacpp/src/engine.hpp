#ifndef __engine_llamacpp_engine_hpp__
#define __engine_llamacpp_engine_hpp__

#include <chat.h>
#include <llama-cpp.h>

#include <string>
#include <vector>

#include "common.hpp"

class LlamaEngine final : public InferenceEngine {
public:
    explicit LlamaEngine(const std::string& model);

    std::vector<BenchResponse> execute(const BenchRequest& request) const override;

private:
    struct RunConfig;

    std::vector<llama_token> tokenize(const BenchRequest& request) const;

    BenchResponse run_single(
        std::vector<llama_token> tokens,
        const RunConfig& config,
        const llama_sampler_ptr& sampler_chain
    ) const;

    llama_model_ptr model;
    const llama_vocab* tokenizer;
    common_chat_templates_ptr chat_templates;
    bool has_mtp;
};

std::string decode_token(
    const llama_vocab* vocab,
    llama_token token
);

bool has_mtp_weights(const std::filesystem::path& model_path);

#endif  // __engine_llamacpp_engine_hpp__
