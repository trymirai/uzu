#include "common.hpp"

#include <arg.h>
#include <chat.h>
#include <common.h>
#include <ggml-cpp.h>

#include <glaze/glaze.hpp>

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
    const BenchRequest& request,
    const llama_model_ptr& model
) {
    std::vector<llama_token> prompt_tokens;
    if (request.prompt_text.has_value()) {
        const std::string& text = request.prompt_text.value();
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
    } else if (request.prompt_chat.has_value()) {
        const auto request_json = glz::write_json(request);
        if (!request_json) {
            throw std::runtime_error("Failed to serialize request: " + glz::format_error(request_json.error()));
        }
        const auto json = common_json::parse(request_json.value());

        const auto chat_templates = common_chat_templates_init(model.get(), "");
        common_chat_templates_inputs inputs;
        inputs.messages = common_chat_msgs_parse_oaicompat(json.at("prompt_chat"));
        if (json.contains("tools")) {
            const auto& tools = json.at("tools");
            inputs.tools = common_chat_tools_parse_oaicompat(tools);
            inputs.chat_template_kwargs["tools"] = tools.dump();
        }
        if (json.contains("tool_choice")) {
            const auto& tool_choice = json.at("tool_choice");
            if (tool_choice.is_string()) {
                inputs.tool_choice = common_chat_tool_choice_parse_oaicompat(tool_choice.get<std::string>());
            }
            inputs.chat_template_kwargs["tool_choice"] = tool_choice.dump();
        }
        inputs.use_jinja = true;
        inputs.add_generation_prompt = true;
        const std::string formatted_prompt = common_chat_templates_apply(chat_templates.get(), inputs).prompt;

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
