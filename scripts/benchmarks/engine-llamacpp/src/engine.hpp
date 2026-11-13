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
        const std::string& draft_model = "",
        ggml_type kv_cache_type = GGML_TYPE_F16
    );

    std::vector<BenchResponse> execute(const BenchRequest& request) const override;

    bool supports_reasoning_effort() const override {
        return true;
    }

private:
    struct RunConfig;

    std::vector<llama_token> tokenize(const BenchRequest& request) const;

    BenchResponse run_single(
        std::vector<llama_token> tokens,
        const RunConfig& config,
        const llama_sampler_ptr& sampler_chain
    ) const;

    llama_model_ptr model;
    llama_model_ptr draft_model;
    size_t dflash_depth = 0;
    // The element type of the target's and the draft's K and V caches.
    ggml_type kv_cache_type;
    const llama_vocab* tokenizer;
    common_chat_templates_ptr chat_templates;
    bool has_mtp;
    // GPU memory in use before the model loaded; Linux reports usage above it.
    uint64_t graphics_baseline = 0;
};

std::string decode_token(
    const llama_vocab* vocab,
    llama_token token
);

bool has_mtp_weights(const std::filesystem::path& model_path);

#endif  // __engine_llamacpp_engine_hpp__
