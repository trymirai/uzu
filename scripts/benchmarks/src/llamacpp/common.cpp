#include "common.hpp"

#include <arg.h>
#include <common.h>
#include <ggml-cpp.h>

bool has_mtp_weights(
    const std::filesystem::path& model_path,
    const llama_model* model
) {
    if (llama_model_n_layer_nextn(model) == 0) {
        return false;
    }

    // Some GGUFs retain NextN metadata after removing the heads, and Granite-Switch uses it for a router.
    // Check for actual MTP tensors, including in later shards of a split GGUF.
    const std::string tensor_name = "blk." + std::to_string(llama_model_n_layer(model)) + ".nextn.eh_proj.weight";
    const auto read_header = [](const char* path) {
        gguf_context_ptr header{gguf_init_from_file(path, {true, nullptr})};
        if (!header) {
            throw std::runtime_error(std::string{"Failed to read GGUF metadata: "} + path);
        }
        return header;
    };

    auto header = read_header(model_path.c_str());
    if (gguf_find_tensor(header.get(), tensor_name.c_str()) >= 0) {
        return true;
    }
    const int64_t split_key = gguf_find_key(header.get(), "split.count");
    const uint16_t split_count = split_key < 0 ? 1 : gguf_get_val_u16(header.get(), split_key);
    if (split_count <= 1) {
        return false;
    }

    std::vector<char> prefix(model_path.string().size() + 1);
    if (llama_split_prefix(prefix.data(), prefix.size(), model_path.c_str(), 0, split_count) <= 0) {
        throw std::runtime_error("Invalid split GGUF path: " + model_path.string());
    }
    std::vector<char> split_path(prefix.size());
    for (uint16_t i = 1; i < split_count; ++i) {
        llama_split_path(split_path.data(), split_path.size(), prefix.data(), i, split_count);
        header = read_header(split_path.data());
        if (gguf_find_tensor(header.get(), tensor_name.c_str()) >= 0) {
            return true;
        }
    }
    return false;
}

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

std::string decode_token(
    const llama_vocab* vocab,
    llama_token token
) {
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

std::vector<llama_token> get_tokens(
    const std::optional<std::string>& prompt_text,
    const std::optional<std::vector<ChatMessage>>& prompt_chat,
    const llama_model_ptr& model
) {
    std::vector<llama_token> prompt_tokens;
    if (prompt_text.has_value()) {
        const std::string& text = prompt_text.value();
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
    } else if (prompt_chat.has_value()) {
        const auto& chat = prompt_chat.value();
        std::vector<llama_chat_message> messages;
        messages.reserve(chat.size());
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
        throw std::invalid_argument("prompt_text and prompt_chat are absent");
    }

    return prompt_tokens;
}
