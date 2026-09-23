#include "runner.hpp"

#include <common.h>
#include <speculative.h>

#include <algorithm>
#include <limits>

#include "batch.hpp"
#include "common.hpp"

struct RunConfig {
    size_t max_tokens;
    const llama_vocab* vocab;
    llama_context_params ctx_params;
    std::optional<llama_context_params> mtp_params;
    common_params_speculative spec_params;
};

static BenchResponse run_single(
    const llama_model_ptr& model,
    std::vector<llama_token> tokens,
    const RunConfig& config,
    const llama_sampler_ptr& sampler_chain
) {
    const size_t max_tokens = config.max_tokens;
    const size_t prompt_tokens_count = tokens.size();
    const bool use_mtp = config.mtp_params.has_value();

    // Reuse the sampling pipeline with fresh state and a new random seed for this run.
    llama_sampler_reset(sampler_chain.get());

    llama_context_ptr ctx{llama_init_from_model(model.get(), config.ctx_params)};
    if (!ctx) {
        throw std::runtime_error("Failed to create context");
    }

    // The MTP context uses the same model weights and the upstream hidden-state transfer/acceptance driver.
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
    const llama_vocab* vocab = config.vocab;
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
    bool stop = llama_vocab_is_eog(vocab, token);
    if (!stop) {
        output_text.append(decode_token(vocab, token));
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
            stop = llama_vocab_is_eog(vocab, token);
            if (stop) {
                break;
            }
            output_text.append(decode_token(vocab, token));
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

std::vector<BenchResponse> run(
    const std::string& input_model_path,
    const BenchRequest& request
) {
    if (!request.prompt_text.has_value() && !request.prompt_chat.has_value()) {
        throw std::invalid_argument("prompt_text and prompt_chat are absent");
    }

    const size_t num_runs = request.num_runs.value_or(1);
    if (num_runs == 0) {
        throw std::invalid_argument("num_runs must be 1 or greater");
    }

    const std::filesystem::path model_path = get_model_path(input_model_path);
    const size_t speculative_depth = request.speculative_depth.value_or(0);

    // load model weights and tokenize the prompt once for all runs
    llama_model_params params = llama_model_default_params();
    params.load_mtp = speculative_depth > 0;
    llama_model_ptr model{llama_model_load_from_file(model_path.c_str(), params)};
    if (!model) {
        throw std::runtime_error("Failed to load model: " + model_path.string());
    }

    // get prompt tokens
    const std::vector<llama_token> tokens = get_tokens(request, model);
    const size_t max_tokens = request.max_tokens.value_or(256);
    const size_t max_context = std::numeric_limits<llama_pos>::max();
    if (tokens.empty() || tokens.size() > max_context || max_tokens > max_context - tokens.size()) {
        throw std::invalid_argument("Prompt and generation length exceed the supported context size");
    }

    // prepare config to run
    RunConfig config{
        .max_tokens = max_tokens,
        .vocab = llama_model_get_vocab(model.get()),
        .ctx_params = llama_context_default_params(),
    };
    config.ctx_params.n_ctx = std::max<size_t>(config.ctx_params.n_ctx, tokens.size() + max_tokens);

    const bool use_mtp = speculative_depth > 0 && has_mtp_weights(model_path, model.get());
    if (use_mtp) {
        config.spec_params.types = {COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
        // Keep verification in one physical batch so recurrent snapshots cover every draft token.
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

    // build the sampling pipeline once; run_single resets its mutable state before each run.
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
        BenchResponse response = run_single(model, tokens, config, sampler_chain);
        responses.push_back(response);
    }
    return responses;
}
