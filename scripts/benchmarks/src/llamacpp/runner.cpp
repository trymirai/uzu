#include "runner.hpp"

#include "common.hpp"

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