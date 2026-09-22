#include "common.hpp"

#include <arg.h>
#include <common.h>

memory_counters_t collect_memory_counters() {
    memory_counters_t memory_counters{};
    const kern_return_t result = get_memory_counters(&memory_counters, false);
    if (result != KERN_SUCCESS) {
        throw std::runtime_error(
            std::string{"Failed to collect memory counters: "} + memory_counters_error_string(result)
        );
    }
    return memory_counters;
}

std::string decode_token(const llama_vocab* vocab, llama_token token) {
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

std::filesystem::path get_model_path(const std::string& model) {
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

std::vector<llama_token> get_tokens(const Content& content_variant, const llama_model_ptr& model) {
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