// uzu-widths: 32, 64, 128
// Slang compute kernel. ABI declarations are resolved by the WebGPU binding adapter.
#include "abi.slang"
UZU_PARAM(uint, HEAD_DIM)
UZU_PARAM(uint, gqa_factor)
UZU_PARAM(bool, has_sinks)
UZU_PARAM(bool, is_causal)
UZU_PARAM(bool, is_kv_cache_ring)
UZU_PARAM(bool, is_sliding_window)
UZU_PARAM(uint, k_head_stride)
UZU_PARAM(uint, k_seq_stride)
UZU_PARAM(uint, lane_count)
UZU_PARAM(uint, num_heads)
UZU_PARAM(uint, ring_params)
UZU_PARAM(uint, ring_params_1)
UZU_PARAM(float, scale)
UZU_PARAM(uint, sequence_length)
UZU_PARAM(uint, sliding_window_size)
UZU_PARAM(uint, suffix_length)
UZU_PARAM(uint, v_head_stride)
UZU_PARAM(uint, v_seq_stride)
UZU_BUFFER(keys)
UZU_BUFFER(out)
UZU_BUFFER(queries)
UZU_BUFFER(sinks)
UZU_BUFFER(values)

// One workgroup is one hardware subgroup. Each lane retains its query and
// output components in registers; online softmax never materializes scores.
[shader("compute")][numthreads(WIDTH, 1, 1)] void main(uint lane : SV_WaveLaneIndex,
                                                       uint3 group : SV_GroupID) {
    let head = group.x;
    let query = group.y;
    let kvHead = head / param_gqa_factor();
    let prefix = param_sequence_length() - param_suffix_length();
    let suffixPosition = param_is_kv_cache_ring() ? param_ring_params_1() : prefix;
    let queryPosition = suffixPosition + query;
    float q[256 / WIDTH];
    float accumulator[256 / WIDTH];
    [ForceUnroll] for (uint part = 0; part < 256 / WIDTH; part++) {
        let dim = lane + part * WIDTH;
        q[part] = dim < param_HEAD_DIM()
                      ? load_queries((head * param_suffix_length() + query) * param_HEAD_DIM() + dim) *
                            param_scale()
                      : 0.0;
        accumulator[part] = 0.0;
    }
    float maximum = -3.402823466e38;
    float denominator = 0.0;
    if (param_has_sinks()) {
        maximum = load_sinks(head);
        denominator = 1.0;
    }
    for (uint key = 0; key < param_sequence_length(); key++) {
        bool useKey = true;
        uint keyPosition = key;
        if (key >= prefix) {
            let suffixKey = key - prefix;
            keyPosition = suffixPosition + suffixKey;
            if (param_is_causal())
                useKey = suffixKey <= query;
        } else if (param_is_kv_cache_ring()) {
            keyPosition = (prefix + key - param_ring_params()) % max(prefix, 1u);
            useKey = keyPosition < param_ring_params_1();
        }
        if (param_is_sliding_window()) {
            if (param_is_causal())
                useKey = useKey && keyPosition <= queryPosition &&
                         queryPosition - keyPosition < param_sliding_window_size();
            else
                useKey = useKey && max(keyPosition, queryPosition) - min(keyPosition, queryPosition) <=
                                       param_sliding_window_size() / 2u;
        }
        if (!useKey)
            continue;
        float product = 0.0;
        [ForceUnroll] for (uint part = 0; part < 256 / WIDTH; part++) {
            let dim = lane + part * WIDTH;
            if (dim < param_HEAD_DIM())
                product +=
                    q[part] * load_keys(kvHead * param_k_head_stride() + key * param_k_seq_stride() + dim);
        }
        let score = WaveActiveSum(product);
        let nextMaximum = max(maximum, score);
        let factor = exp(maximum - nextMaximum);
        let probability = exp(score - nextMaximum);
        denominator = denominator * factor + probability;
        maximum = nextMaximum;
        [ForceUnroll] for (uint part = 0; part < 256 / WIDTH; part++) {
            let dim = lane + part * WIDTH;
            if (dim < param_HEAD_DIM())
                accumulator[part] =
                    accumulator[part] * factor + probability * load_values(kvHead * param_v_head_stride() +
                                                                           key * param_v_seq_stride() + dim);
        }
    }
    [ForceUnroll] for (uint part = 0; part < 256 / WIDTH; part++) {
        let dim = lane + part * WIDTH;
        if (dim < param_HEAD_DIM())
            store_out((query * param_num_heads() + head) * param_HEAD_DIM() + dim,
                      accumulator[part] / denominator);
    }
}
