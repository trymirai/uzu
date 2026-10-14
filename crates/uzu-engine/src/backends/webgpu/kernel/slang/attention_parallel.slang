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

// Split the key sequence across subgroups, then merge their online-softmax
// states. This preserves the single-pass kernel interface while filling more
// lanes for decode, where there are only a few query/head workgroups.
groupshared float maxima[256 / WIDTH];
groupshared float denominators[256 / WIDTH];
groupshared float factors[256 / WIDTH];
groupshared float partialOutputs[(256 / WIDTH) * 256];
[shader("compute")][numthreads(256, 1, 1)] void main(uint lane : SV_WaveLaneIndex, uint wave : SV_WaveIndex,
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
    if (param_has_sinks() && wave == 0u) {
        maximum = load_sinks(head);
        denominator = 1.0;
    }
    for (uint key = wave; key < param_sequence_length(); key += 256 / WIDTH) {
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
    if (lane == 0u) {
        maxima[wave] = maximum;
        denominators[wave] = denominator;
    }
    [ForceUnroll] for (uint part = 0; part < 256 / WIDTH; part++) {
        let dim = lane + part * WIDTH;
        if (dim < param_HEAD_DIM())
            partialOutputs[wave * 256u + dim] = accumulator[part];
    }
    GroupMemoryBarrierWithGroupSync();
    let partialMaximum = lane < 256 / WIDTH ? maxima[lane] : -3.402823466e38;
    let mergedMaximum = WaveActiveMax(partialMaximum);
    let factor = lane < 256 / WIDTH ? exp(partialMaximum - mergedMaximum) : 0.0;
    let partialDenominator = lane < 256 / WIDTH ? denominators[lane] : 0.0;
    let mergedDenominator = WaveActiveSum(partialDenominator * factor);
    if (wave == 0u && lane < 256 / WIDTH)
        factors[lane] = factor;
    GroupMemoryBarrierWithGroupSync();
    if (wave == 0u) {
        [ForceUnroll] for (uint part = 0; part < 256 / WIDTH; part++) {
            let dim = lane + part * WIDTH;
            if (dim < param_HEAD_DIM()) {
                float value = 0.0;
                [ForceUnroll] for (uint sourceWave = 0; sourceWave < 256 / WIDTH; sourceWave++) value +=
                    partialOutputs[sourceWave * 256u + dim] * factors[sourceWave];
                store_out((query * param_num_heads() + head) * param_HEAD_DIM() + dim,
                          value / mergedDenominator);
            }
        }
    }
}
