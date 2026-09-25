#include "engine.hpp"

#include <common.h>
#include <ggml-cpp.h>
#include <speculative.h>

#include <algorithm>
#include <glaze/glaze.hpp>
#include <limits>
#include <stdexcept>

#include "util.hpp"

struct llama_batch_deleter {
    void operator()(llama_batch* batch) {
        llama_batch_free(*batch);
    }
};

typedef std::unique_ptr<llama_batch, llama_batch_deleter> llama_batch_ptr;

struct LlamaEngine::RunConfig {
    size_t max_tokens;
    llama_context_params ctx_params;
    std::optional<llama_context_params> mtp_params;
    common_params_speculative spec_params;
};

LlamaEngine::LlamaEngine(const std::string& model) {
    const std::filesystem::path model_path = get_model_path(model);
    llama_model_params params = llama_model_default_params();
    params.load_mtp = has_mtp_weights(model_path);
    this->model.reset(llama_model_load_from_file(model_path.c_str(), params));
    if (!this->model) {
        throw std::runtime_error("Failed to load model: " + model_path.string());
    }

    this->tokenizer = llama_model_get_vocab(this->model.get());
    if (!this->tokenizer) {
        throw std::runtime_error("Failed to load tokenizer: " + model_path.string());
    }

    this->chat_templates = common_chat_templates_init(this->model.get(), "");
    if (!this->chat_templates) {
        throw std::runtime_error("Failed to load chat templates: " + model_path.string());
    }

    this->has_mtp = llama_model_n_layer_nextn(this->model.get()) > 0;
}

std::vector<llama_token> LlamaEngine::tokenize(const BenchRequest& request) const {
    std::vector<llama_token> prompt_tokens;
    if (request.prompt_text.has_value()) {
        const std::string& text = request.prompt_text.value();
        const int32_t prompt_tokens_count =
            -llama_tokenize(this->tokenizer, text.c_str(), text.size(), nullptr, 0, true, true);
        if (prompt_tokens_count <= 0) {
            throw std::runtime_error("Failed to determine prompt token count");
        }

        prompt_tokens.resize(prompt_tokens_count);
        const int tokenize_result = llama_tokenize(
            this->tokenizer,
            text.c_str(),
            text.size(),
            prompt_tokens.data(),
            prompt_tokens.size(),
            true,
            true
        );
        if (tokenize_result < 0) {
            throw std::runtime_error("Failed to tokenize prompt");
        }
    } else if (request.prompt_chat.has_value()) {
        const auto request_json = glz::write_json(request);
        if (!request_json) {
            throw std::runtime_error("Failed to serialize request: " + glz::format_error(request_json.error()));
        }

        const auto json = common_json::parse(request_json.value());
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

        const std::string formatted_prompt = common_chat_templates_apply(this->chat_templates.get(), inputs).prompt;
        const int32_t prompt_tokens_count =
            -llama_tokenize(this->tokenizer, formatted_prompt.c_str(), formatted_prompt.size(), nullptr, 0, true, true);
        if (prompt_tokens_count <= 0) {
            throw std::runtime_error("Failed to determine prompt token count");
        }

        prompt_tokens.resize(prompt_tokens_count);
        const int32_t tokenize_result = llama_tokenize(
            this->tokenizer,
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

BenchResponse LlamaEngine::run_single(
    std::vector<llama_token> tokens,
    const RunConfig& config,
    const llama_sampler_ptr& sampler_chain
) const {
    const size_t max_tokens = config.max_tokens;
    const size_t prompt_tokens_count = tokens.size();
    const bool use_mtp = config.mtp_params.has_value();

    llama_sampler_reset(sampler_chain.get());
    llama_context_ptr ctx{llama_init_from_model(model.get(), config.ctx_params)};
    if (!ctx) {
        throw std::runtime_error("Failed to create context");
    }

    llama_context_ptr ctx_mtp;
    common_speculative_ptr spec;
    bool checkpoint_target = false;
    if (use_mtp) {
        const common_context_seq_rm_type rm_type = common_context_can_seq_rm(ctx.get());
        if (rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            throw std::runtime_error("Target context does not support speculative rollback");
        }
        checkpoint_target = rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

        llama_context_params mtp_params = config.mtp_params.value();
        mtp_params.ctx_other = ctx.get();
        mtp_params.n_ctx = llama_n_ctx(ctx.get());
        ctx_mtp.reset(llama_init_from_model(model.get(), mtp_params));
        if (!ctx_mtp) {
            throw std::runtime_error("Failed to create MTP context");
        }

        common_params_speculative spec_params = config.spec_params;
        spec_params.draft.ctx_tgt = ctx.get();
        spec_params.draft.ctx_dft = ctx_mtp.get();
        spec.reset(common_speculative_init(spec_params, 1));
        if (!spec) {
            throw std::runtime_error("Failed to initialize MTP speculation");
        }
    }

    size_t forward_passes = 0;
    size_t tokens_generated = 0;
    std::string output_text;

    memory_counters_t memory_counters_max = collect_memory_counters();
    const auto update_memory = [&]() {
        const memory_counters_t memory_counters = collect_memory_counters();
        if (memory_counters.graphics_total > memory_counters_max.graphics_total) {
            memory_counters_max = memory_counters;
        }
    };

    const auto trim_context = [](llama_context* context, llama_pos pos) {
        if (!llama_memory_seq_rm(llama_get_memory(context), 0, pos, -1)) {
            throw std::runtime_error("Failed to roll back speculative tokens at position " + std::to_string(pos));
        }
    };

    // prefill
    llama_batch_ptr batch_storage{new llama_batch(llama_batch_init(llama_n_batch(ctx.get()), 0, 1))};
    llama_batch& batch = *batch_storage;
    const int64_t time_start = llama_time_us();
    const size_t prefill_batch_size = use_mtp ? llama_n_ubatch(ctx.get()) : llama_n_batch(ctx.get());
    for (size_t offset = 0; offset < tokens.size(); offset += prefill_batch_size) {
        common_batch_clear(batch);
        const size_t end = std::min(tokens.size(), offset + prefill_batch_size);
        for (size_t i = offset; i < end; ++i) {
            common_batch_add(batch, tokens[i], i, {0}, i + 1 == tokens.size());
        }
        if (llama_decode(ctx.get(), batch) != 0) {
            throw std::runtime_error("Prefill failed");
        }
        forward_passes++;
        if (!common_speculative_process(spec.get(), batch)) {
            throw std::runtime_error("MTP prefill failed");
        }
    }

    llama_synchronize(ctx.get());
    if (ctx_mtp) {
        llama_synchronize(ctx_mtp.get());
        common_speculative_begin(spec.get(), 0, tokens);
    }

    const int64_t time_prefill_end = llama_time_us();

    update_memory();

    // sample first token
    llama_token token = llama_sampler_sample(sampler_chain.get(), ctx.get(), -1);
    const int64_t time_first_token = llama_time_us();
    tokens_generated++;
    bool stop = llama_vocab_is_eog(this->tokenizer, token);
    if (!stop) {
        output_text.append(decode_token(this->tokenizer, token));
    }

    // decoding loop
    common_prompt_checkpoint checkpoint;
    while (!stop && tokens_generated < max_tokens) {
        const llama_pos n_past = tokens.size();
        std::vector<llama_token> draft;
        // The upstream driver limits the result after drafting, so its full draft must fit the context.
        if (spec && max_tokens - tokens_generated > 1 &&
            llama_n_ctx(ctx_mtp.get()) - n_past >= static_cast<uint32_t>(common_speculative_n_max(spec.get()))) {
            const int32_t n_draft =
                std::min<size_t>(common_speculative_n_max(spec.get()), max_tokens - tokens_generated - 1);
            common_speculative_get_draft_params(spec.get(), 0) = {true, n_draft, n_past, token, &tokens, &draft};
            common_speculative_draft(spec.get());
            trim_context(ctx_mtp.get(), n_past);
        }

        if (checkpoint_target && !draft.empty()) {
            checkpoint.update_tgt(ctx.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        }

        common_batch_clear(batch);
        common_batch_add(batch, token, n_past, {0}, true);
        for (size_t i = 0; i < draft.size(); ++i) {
            common_batch_add(batch, draft[i], n_past + i + 1, {0}, true);
        }

        if (llama_decode(ctx.get(), batch) != 0) {
            throw std::runtime_error("Token decode failed: " + std::to_string(token));
        }
        forward_passes++;
        llama_synchronize(ctx.get());

        // Sample only from target logits; accept a draft token only when the target sampler agrees.
        size_t accepted = 0;
        for (size_t i = 0; i <= draft.size(); ++i) {
            tokens.push_back(token);
            token = llama_sampler_sample(sampler_chain.get(), ctx.get(), i);
            tokens_generated++;
            stop = llama_vocab_is_eog(this->tokenizer, token);
            if (stop) {
                break;
            }
            output_text.append(decode_token(this->tokenizer, token));
            if (i == draft.size() || token != draft[i] || tokens_generated == max_tokens) {
                break;
            }
            accepted++;
        }

        if (spec) {
            // Older recurrent architectures cannot remove a suffix. Restore their state and replay
            // the committed inputs without sampling again, preserving the target sampler's RNG state.
            if (checkpoint_target && tokens.size() - n_past < static_cast<size_t>(batch.n_tokens)) {
                checkpoint.load_tgt(ctx.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                trim_context(ctx.get(), n_past);
                batch.n_tokens = tokens.size() - n_past;
                if (llama_decode(ctx.get(), batch) != 0) {
                    throw std::runtime_error("Failed to replay accepted MTP tokens");
                }
                forward_passes++;
            }
            if (!common_speculative_process(spec.get(), batch)) {
                throw std::runtime_error("MTP verification failed");
            }
            common_speculative_accept(spec.get(), 0, accepted);
            trim_context(ctx.get(), tokens.size());
            trim_context(ctx_mtp.get(), tokens.size());
            llama_synchronize(ctx_mtp.get());
        }

        update_memory();
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
    return BenchResponse{
        output_text,
        time_to_first_token,
        prompt_tps,
        decode_tps,
        tokens_per_fp,
        total_duration,
        memory_counters_max.phys_footprint,
        memory_counters_max.resident_size_peak,
        memory_counters_max.graphics_total
    };
}

std::vector<BenchResponse> LlamaEngine::execute(const BenchRequest& request) const {
    if (!request.prompt_text.has_value() && !request.prompt_chat.has_value()) {
        throw std::invalid_argument("prompt_text and prompt_chat are absent");
    }

    const size_t num_runs = request.num_runs.value_or(1);
    if (num_runs == 0) {
        throw std::invalid_argument("num_runs must be 1 or greater");
    }

    const size_t speculative_depth = request.speculative_depth.value_or(0);

    const std::vector<llama_token> tokens = tokenize(request);
    const size_t max_tokens = request.max_tokens.value_or(256);
    const size_t max_context = std::numeric_limits<llama_pos>::max();
    if (tokens.empty() || tokens.size() > max_context || max_tokens > max_context - tokens.size()) {
        throw std::invalid_argument("Prompt and generation length exceed the supported context size");
    }

    // prepare config to run
    RunConfig config{
        .max_tokens = max_tokens,
        .ctx_params = llama_context_default_params(),
    };
    config.ctx_params.n_ctx = std::max<size_t>(config.ctx_params.n_ctx, tokens.size() + max_tokens);

    const bool use_mtp = speculative_depth > 0 && this->has_mtp;
    if (use_mtp) {
        config.spec_params.types = {COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
        // keep verification in one physical batch so recurrent snapshots cover every draft token.
        config.spec_params.draft.n_max =
            std::min<int32_t>(speculative_depth, std::min(config.ctx_params.n_batch, config.ctx_params.n_ubatch) - 1);
        config.ctx_params.n_rs_seq = config.spec_params.need_n_rs_seq();
        config.ctx_params.n_outputs_max = config.spec_params.draft.n_max + 1;
        config.ctx_params.n_outputs_max_per_seq = config.spec_params.draft.n_max + 1;

        config.mtp_params = config.ctx_params;
        config.mtp_params->ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        config.mtp_params->n_rs_seq = 0;
        config.mtp_params->n_outputs_max = 1;
        config.mtp_params->n_outputs_max_per_seq = 1;
    }

    // build the sampling pipeline once.
    // run_single resets its mutable state before each run.
    llama_sampler_chain_params sampler_chain_params = llama_sampler_chain_default_params();
    llama_sampler_ptr sampler_chain{llama_sampler_chain_init(sampler_chain_params)};
    if (request.sampling.has_value()) {
        const auto& sampling = request.sampling.value();
        if (sampling.top_k.has_value()) {
            llama_sampler_chain_add(sampler_chain.get(), llama_sampler_init_top_k(sampling.top_k.value()));
        }
        if (sampling.top_p.has_value()) {
            llama_sampler_chain_add(sampler_chain.get(), llama_sampler_init_top_p(sampling.top_p.value(), 1));
        }
        if (sampling.min_p.has_value()) {
            llama_sampler_chain_add(sampler_chain.get(), llama_sampler_init_min_p(sampling.min_p.value(), 1));
        }
        if (sampling.temp.has_value()) {
            llama_sampler_chain_add(sampler_chain.get(), llama_sampler_init_temp(sampling.temp.value()));
        }
        llama_sampler_chain_add(sampler_chain.get(), llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    } else {
        llama_sampler_chain_add(sampler_chain.get(), llama_sampler_init_greedy());
    }

    std::vector<BenchResponse> responses;
    for (size_t i = 0; i < num_runs; ++i) {
        BenchResponse response = run_single(tokens, config, sampler_chain);
        responses.push_back(response);
    }
    return responses;
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

bool has_mtp_weights(const std::filesystem::path& model_path) {
    // Some GGUFs retain NextN metadata after removing the heads, and Granite-Switch uses it for a router.
    // Check for actual MTP tensors, including in later shards of a split GGUF.
    const auto contains_mtp_weights = [](const gguf_context* header) {
        for (int64_t i = 0; i < gguf_get_n_tensors(header); ++i) {
            const std::string_view name = gguf_get_tensor_name(header, i);
            if (name.starts_with("blk.") && name.ends_with(".nextn.eh_proj.weight")) {
                return true;
            }
        }
        return false;
    };

    const auto read_header = [](const char* path) {
        gguf_context_ptr header{gguf_init_from_file(path, {true, nullptr})};
        if (!header) {
            throw std::runtime_error(std::string{"Failed to read GGUF metadata: "} + path);
        }
        return header;
    };

    auto header = read_header(model_path.c_str());
    if (contains_mtp_weights(header.get())) {
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
        if (contains_mtp_weights(header.get())) {
            return true;
        }
    }

    return false;
}
