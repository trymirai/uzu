// Slang compute kernel. ABI declarations are resolved by the WebGPU binding adapter.
#include "abi.slang"
UZU_PARAM(float, repetition_penalty)
UZU_PARAM(uint, sampling_length)
UZU_PARAM(uint, sampling_start)
UZU_PARAM(uint, suffix_repetition_length)
UZU_PARAM(uint, vocab_size)
UZU_BUFFER(context_ring)
UZU_BUFFER(logits_copy)
UZU_BUFFER(original_logits)
UZU_BUFFER(token_ids)

[shader("compute")][numthreads(256, 1, 1)] void main(uint3 gid : SV_DispatchThreadID) {
    let window_index = gid.x;
    let sample = gid.y;
    if (window_index >= param_suffix_repetition_length() || sample >= param_sampling_length()) {
        return;
    }
    let ring_offset = word_context_ring(0u);
    let ring_length = word_context_ring(1u);
    let total_length = ring_length + param_sampling_start() + sample + 1u;
    let window_length = min(total_length, param_suffix_repetition_length());
    if (window_index >= window_length) {
        return;
    }
    let source = total_length - window_length + window_index;
    var token = 0u;
    if (source < ring_length) {
        token = word_context_ring(2u + (ring_offset + source) % param_suffix_repetition_length());
    } else {
        token = word_token_ids(source - ring_length);
    }
    if (token >= param_vocab_size()) {
        return;
    }
    let offset = sample * param_vocab_size() + token;
    let logit = load_original_logits(offset);
    var penalized = logit * param_repetition_penalty();
    if (logit > 0.0) {
        penalized = logit / param_repetition_penalty();
    }
    store_logits_copy(offset, penalized);
}
