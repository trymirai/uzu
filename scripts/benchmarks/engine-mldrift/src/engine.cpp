#include "engine.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <glaze/glaze.hpp>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "absl/flags/flag.h"
#include "absl/status/status.h"

ABSL_FLAG(
    std::string,
    model,
    "",
    "ML Drift model preset, detected from the checkpoint configuration"
);

struct HfTextConfig {
    int hidden_size = 0;
    int num_hidden_layers = 0;
    int vocab_size = 0;
    size_t max_position_embeddings = 0;
};

struct HfConfig {
    std::optional<HfTextConfig> text_config = std::nullopt;
};

static constexpr std::array<std::string_view, 7> PRESETS = {
    "gemma3:270m", "gemma3:1b", "gemma4:12b", "qwen3:0.6b", "qwen3:1.7b", "qwen3:8b", "qwen3:14b"
};

static void check(const absl::Status& status) {
    if (!status.ok()) {
        throw std::runtime_error(std::string{status.message()});
    }
}

static HfTextConfig read_text_config(const std::filesystem::path& checkpoint) {
    const auto json = ml_drift::LoadFileToString(checkpoint / "config.json");
    check(json.status());

    constexpr glz::opts options{.error_on_unknown_keys = false};
    HfConfig config;
    if (const auto error = glz::read<options>(config, json.value())) {
        throw std::runtime_error(glz::format_error(error, json.value()));
    }
    if (config.text_config.has_value()) {
        return config.text_config.value();
    }

    HfTextConfig text_config;
    if (const auto error = glz::read<options>(text_config, json.value())) {
        throw std::runtime_error(glz::format_error(error, json.value()));
    }
    return text_config;
}

static std::string render_chat(
    ml_drift::ModelType type,
    const std::vector<ChatMessage>& messages
) {
    const auto [open, close, model_role] = [type]() -> std::array<std::string_view, 3> {
        switch (type) {
            case ml_drift::ModelType::kQwen3:
                return {"<|im_start|>", "<|im_end|>", "assistant"};
            case ml_drift::ModelType::kGemma4:
                return {"<|turn>", "<turn|>", "model"};
            case ml_drift::ModelType::kGemma3:
                return {"<start_of_turn>", "<end_of_turn>", "model"};
        }
        throw std::invalid_argument("Unsupported ML Drift model type");
    }();

    std::string prompt;
    for (const ChatMessage& message : messages) {
        const bool supported_role = message.role == "user" || message.role == "assistant" ||
            (message.role == "system" && type != ml_drift::ModelType::kGemma3);
        if (!supported_role || !message.content.has_value() || message.reasoning_content.has_value() ||
            message.tool_calls.has_value() || message.tool_call_id.has_value()) {
            throw std::invalid_argument("ML Drift chats support plain text turns of the model's roles only");
        }
        const std::string_view role = message.role == "assistant" ? model_role : std::string_view{message.role};
        prompt.append(open).append(role).append("\n").append(message.content.value()).append(close).append("\n");
    }
    return prompt.append(open).append(model_role).append("\n");
}

static std::string_view detect_preset(const HfTextConfig& config) {
    for (const std::string_view preset : PRESETS) {
        const auto info = ml_drift::GetModelInfo(preset);
        if (info.ok() && info->config.model_dimension == config.hidden_size &&
            info->config.stack_size == config.num_hidden_layers && info->config.vocabulary_size == config.vocab_size) {
            return preset;
        }
    }
    throw std::invalid_argument("Checkpoint does not match an ML Drift model preset");
}

MlDriftEngine::MlDriftEngine(
    const std::filesystem::path& checkpoint,
    const std::filesystem::path& weights
)
    : weights((weights / "").string()) {
    const HfTextConfig config = read_text_config(checkpoint);
    const std::string_view preset = detect_preset(config);
    absl::SetFlag(&FLAGS_model, std::string{preset});
    this->model_info = *ml_drift::GetModelInfo(preset);
    this->context_capacity = config.max_position_embeddings;

    const auto tokenizer_path = checkpoint / ml_drift::GetTokenizerFilename(this->model_info.type);
    auto tokenizer = ml_drift::Tokenizer::Create(tokenizer_path, this->model_info.type);
    check(tokenizer.status());
    this->tokenizer = std::move(tokenizer).value();
}

BenchResponse MlDriftEngine::execute(const BenchRequest& request) const {
    if (request.tools.has_value() || request.tool_choice.has_value()) {
        throw std::invalid_argument("ML Drift does not support tools");
    }
    if (request.reasoning_effort != ReasoningEffort::default_effort) {
        throw std::invalid_argument("ML Drift does not support reasoning_effort");
    }
    if (request.seed.has_value()) {
        throw std::invalid_argument("ML Drift does not support seed");
    }

    const std::string prompt = render_chat(this->model_info.type, request.messages);
    const std::vector<int> tokens = this->tokenizer->EncodeInputTokens(prompt, false);
    const size_t max_tokens = request.max_output_tokens.value_or(0);
    if (tokens.size() >= this->context_capacity || max_tokens > this->context_capacity - tokens.size()) {
        throw std::invalid_argument("Prompt and generation length exceed the model's context capacity");
    }
    const size_t context_size = max_tokens > 0
        ? std::min<size_t>(ml_drift::round_up(tokens.size() + max_tokens, 256), this->context_capacity)
        : this->context_capacity;
    const size_t generation_limit = max_tokens > 0 ? max_tokens : context_size - tokens.size();

    if (!this->runner || this->runner_prompt_size != tokens.size() || this->runner_context_size != context_size) {
        this->runner.reset();
        auto runner = ml_drift::CreateLlmModelRunner();
        check(runner->Init(this->weights, tokens.size(), context_size, this->model_info.force_fp32));
        check(runner->InitSliceAndGreedy());
        this->runner = std::move(runner);
        this->runner_prompt_size = tokens.size();
        this->runner_context_size = context_size;
    }

    std::vector<int> output_tokens;
    const double energy_start = collect_gpu_energy();
    const auto time_start = std::chrono::steady_clock::now();
    check(this->runner->PostProcessGreedy(*this->tokenizer, tokens, generation_limit, output_tokens));
    const std::chrono::duration<double> duration = std::chrono::steady_clock::now() - time_start;
    const double energy_gpu = collect_gpu_energy() - energy_start;
    if (max_tokens == 0 && output_tokens.size() == generation_limit) {
        throw std::runtime_error("Context capacity exhausted before EOS (" + std::to_string(context_size) + " tokens)");
    }

    const ml_drift::GenerationStats& stats = this->runner->stats();
    const memory_counters_t memory_counters = collect_memory_counters();
    return BenchResponse{
        this->tokenizer->DecodeOutputTokens(output_tokens),
        output_tokens.size() == generation_limit ? FinishReason::length : FinishReason::stop,
        tokens.size(),
        output_tokens.size(),
        stats.prefill_seconds,
        stats.prefill_seconds > 0.0 ? tokens.size() / stats.prefill_seconds : 0.0,
        stats.decode_seconds > 0.0 ? (output_tokens.size() - 1) / stats.decode_seconds : 0.0,
        std::nullopt,
        duration.count(),
        energy_gpu,
        // ML Drift reports no memory; the kernel's count of its graphics memory is its Metal allocation.
        memory_counters.graphics_total,
        memory_counters.phys_footprint,
        memory_counters.resident_size,
        memory_counters.graphics_total
    };
}
