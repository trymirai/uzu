#include <arg.h>
#include <common.h>
#include <llama-cpp.h>

#include <exception>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "memory_counters.h"

struct ChatMessage {
    std::string message;
    std::string role;
};

using Content = std::variant<std::string, std::vector<ChatMessage>>;

struct SamplingConfig {
    int32_t top_k = 0;
    float top_p = 1.0f;
    float min_p = 0.0f;
    float temp = 1.0f;
};

struct RunRequest {
    std::string model;
    Content input;
    size_t max_tokens;
    std::optional<SamplingConfig> sampling;
};

struct RunResponse {
    std::string text;
    double time_to_first_token;
    double prompt_tps;
    double generation_tps;
    double tokens_per_fp;
    double duration;
    memory_counters_t memory_counters;
};

static std::filesystem::path get_model_path(const std::string& model) {
    if (std::filesystem::is_regular_file(model)) {
        return model;
    }

    common_params params;
    params.model.hf_repo = model;

    auto handler = common_models_handler_init(params, LLAMA_EXAMPLE_COMMON);
    common_models_handler_apply(handler, params);
    if (params.model.path.empty()) {
        throw std::runtime_error("No model found in " + model);
    }

    return params.model.path;
}

static memory_counters_t collect_memory_counters() {
    memory_counters_t memory_counters{};
    const kern_return_t result = get_memory_counters(&memory_counters, false);
    if (result != KERN_SUCCESS) {
        throw std::runtime_error(
            std::string{"Failed to collect memory counters: "} + memory_counters_error_string(result)
        );
    }
    return memory_counters;
}

static std::string decode_token(const llama_vocab* vocab, llama_token token) {
    std::vector<char> piece(128);
    int32_t piece_len = llama_token_to_piece(vocab, token, piece.data(), piece.size(), 0, true);
    if (piece_len < 0) {
        piece_len = -piece_len;
        piece.resize(piece_len);
        piece_len = llama_token_to_piece(vocab, token, piece.data(), piece.size(), 0, true);
    }
    if (piece_len < 0) {
        throw std::runtime_error("Can not convert token to piece");
    }

    return std::string(piece.data(), piece_len);
}

static std::vector<llama_token> get_tokens(const Content& content_variant, const llama_model_ptr& model) {
    std::vector<llama_token> prompt_tokens;
    if (std::holds_alternative<std::string>(content_variant)) {
        const std::string text = std::get<std::string>(content_variant);
        const llama_vocab* vocab = llama_model_get_vocab(model.get());
        const int32_t prompt_tokens_count = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, true, true);
        if (prompt_tokens_count <= 0) {
            throw std::runtime_error("Failed to determine prompt token count");
        }

        prompt_tokens.resize(prompt_tokens_count);
        const int tokenize_result =
            llama_tokenize(vocab, text.c_str(), text.size(), prompt_tokens.data(), prompt_tokens.size(), true, true);
        if (tokenize_result < 0) {
            throw std::runtime_error("Failed to tokenize prompt");
        }
    } else if (std::holds_alternative<std::vector<ChatMessage>>(content_variant)) {
        const auto& chat = std::get<std::vector<ChatMessage>>(content_variant);
        std::vector<llama_chat_message> messages(chat.size());
        for (const auto& message : chat) {
            messages.push_back({message.role.c_str(), message.message.c_str()});
        }

        const std::string chat_template = llama_model_chat_template(model.get(), nullptr);
        const int32_t prompt_length =
            llama_chat_apply_template(chat_template.c_str(), messages.data(), messages.size(), true, nullptr, 0);
        if (prompt_length < 0) {
            throw std::runtime_error("Failed to determine chat prompt length");
        }

        std::string formatted_prompt(prompt_length, '\0');
        const int32_t format_result = llama_chat_apply_template(
            chat_template.c_str(),
            messages.data(),
            messages.size(),
            true,
            formatted_prompt.data(),
            prompt_length
        );
        if (format_result < 0 || format_result > prompt_length) {
            throw std::runtime_error("Failed to apply chat template");
        }
        formatted_prompt.resize(format_result);

        const llama_vocab* vocab = llama_model_get_vocab(model.get());
        const int32_t prompt_tokens_count =
            -llama_tokenize(vocab, formatted_prompt.c_str(), formatted_prompt.size(), nullptr, 0, true, true);
        if (prompt_tokens_count <= 0) {
            throw std::runtime_error("Failed to determine prompt token count");
        }

        prompt_tokens.resize(prompt_tokens_count);
        const int32_t tokenize_result = llama_tokenize(
            vocab,
            formatted_prompt.c_str(),
            formatted_prompt.size(),
            prompt_tokens.data(),
            prompt_tokens.size(),
            true,
            true
        );
        if (tokenize_result < 0) {
            throw std::runtime_error("Failed to tokenize prompt");
        }

        prompt_tokens.resize(tokenize_result);
    } else {
        throw std::runtime_error("Unknown prompt alternative");
    }

    return prompt_tokens;
}

RunResponse run(const RunRequest& request) {
    const std::filesystem::path model_path = get_model_path(request.model);

    // prepare model
    llama_model_params params = llama_model_default_params();
    llama_model_ptr model{llama_model_load_from_file(model_path.c_str(), params)};
    if (!model) {
        throw std::runtime_error("Failed to load model: " + model_path.string());
    }

    // prepare context
    llama_context_params ctx_params = llama_context_default_params();
    llama_context_ptr ctx{llama_init_from_model(model.get(), ctx_params)};
    if (!ctx) {
        throw std::runtime_error("Failed to create context");
    }

    // prepare sampling
    llama_sampler_chain_params sampler_chain_params = llama_sampler_chain_default_params();
    llama_sampler_ptr sampler_chain{llama_sampler_chain_init(sampler_chain_params)};
    if (request.sampling.has_value()) {
        const SamplingConfig sampling = request.sampling.value();
        llama_sampler_chain_add(sampler_chain.get(), llama_sampler_init_top_k(sampling.top_k));
        llama_sampler_chain_add(sampler_chain.get(), llama_sampler_init_top_p(sampling.top_p, 1));
        llama_sampler_chain_add(sampler_chain.get(), llama_sampler_init_min_p(sampling.min_p, 1));
        llama_sampler_chain_add(sampler_chain.get(), llama_sampler_init_temp(sampling.temp));
        llama_sampler_chain_add(sampler_chain.get(), llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    } else {
        llama_sampler_chain_add(sampler_chain.get(), llama_sampler_init_greedy());
    }

    size_t forward_passes = 0;
    size_t tokens_submitted = 0;
    size_t tokens_generated = 0;
    memory_counters_t memory_counters_max = collect_memory_counters();
    const llama_vocab* vocab = llama_model_get_vocab(model.get());
    std::string output_text;

    // prefill
    std::vector<llama_token> tokens = get_tokens(request.input, model);
    const size_t prompt_tokens_count = tokens.size();
    const int64_t time_start = llama_time_us();
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    if (llama_decode(ctx.get(), batch) != 0) {
        throw std::runtime_error("Prefill failed");
    }
    llama_synchronize(ctx.get());
    const int64_t time_prefill_end = llama_time_us();
    forward_passes++;

    // sample first token
    llama_token token = llama_sampler_sample(sampler_chain.get(), ctx.get(), -1);
    const int64_t time_first_token = llama_time_us();
    tokens_generated++;
    bool stop = llama_vocab_is_eog(vocab, token);
    if (!stop) {
        output_text.append(decode_token(vocab, token));
    }

    // decoding loop
    while (!stop && tokens_generated < request.max_tokens) {
        batch = llama_batch_get_one(&token, 1);
        if (llama_decode(ctx.get(), batch) != 0) {
            throw std::runtime_error("Token decode failed: " + std::to_string(token));
        }
        forward_passes++;
        tokens_submitted += batch.n_tokens;
        llama_synchronize(ctx.get());

        token = llama_sampler_sample(sampler_chain.get(), ctx.get(), -1);
        tokens_generated++;
        stop = llama_vocab_is_eog(vocab, token);
        if (!stop) {
            output_text.append(decode_token(vocab, token));
        }

        const memory_counters_t memory_counters = collect_memory_counters();
        if (memory_counters.graphics_total > memory_counters_max.graphics_total) {
            memory_counters_max = memory_counters;
        }
    }

    const int64_t time_end = llama_time_us();

    // collect metrics
    const double prefill_duration = (time_prefill_end - time_start) / 1e6;
    const double decode_duration = (time_end - time_first_token) / 1e6;
    const double total_duration = (time_end - time_start) / 1e6;
    const double time_to_first_token = (time_first_token - time_start) / 1e6;
    const double prompt_tps = prefill_duration > 0.0 ? (prompt_tokens_count / prefill_duration) : 0.0;
    const double decode_tps = decode_duration > 0.0 ? (tokens_generated - 1) / decode_duration : 0.0;
    const double tokens_per_fp = forward_passes > 0 ? (double)(tokens_generated) / forward_passes : 1.0;
    return RunResponse{
        output_text,
        time_to_first_token,
        prompt_tps,
        decode_tps,
        tokens_per_fp,
        total_duration,
        memory_counters_max
    };
};

int main(int argc, char* argv[]) {
    const auto request = RunRequest{
        "unsloth/Qwen3.6-27B-GGUF",
        std::string{"Tell me about London"},
        256,
    };

    ggml_log_callback log_callback = [](enum ggml_log_level level, const char* text, void*) {
        if (level == GGML_LOG_LEVEL_ERROR) {
            fprintf(stderr, "%s", text);
        }
    };
    llama_log_set(log_callback, nullptr);

    int ret = 0;
    llama_backend_init();
    RunResponse response;
    try {
        response = run(request);
    } catch (const std::exception& error) {
        std::cerr << "Failed: " << error.what() << std::endl;
        ret = 1;
    }
    llama_backend_free();

    if (ret == 0) {
        // std::cout << response.text << std::endl;
        std::cout << "time_to_first_token: " << response.time_to_first_token << '\n'
                  << "prompt_tps: " << response.prompt_tps << '\n'
                  << "generation_tps: " << response.generation_tps << '\n'
                  << "tokens_per_fp: " << response.tokens_per_fp << '\n'
                  << "duration: " << response.duration << '\n'
                  << "phys_footprint: " << response.memory_counters.phys_footprint << '\n'
                  << "resident_size: " << response.memory_counters.resident_size << '\n'
                  << "graphics_total: " << response.memory_counters.graphics_total << '\n';
    }

    return ret;
}
