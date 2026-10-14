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

groupshared float reduction[WIDTH];
[shader("compute")][numthreads(WIDTH, 1, 1)] void main(uint3 local : SV_GroupThreadID,
                                                       uint3 group : SV_GroupID) {
    let lane = local.x;
    let head = group.x;
    let query = group.y;
    let kvHead = head / param_gqa_factor();
    let prefix = param_sequence_length() - param_suffix_length();
    var suffixPosition = prefix;
    if (param_is_kv_cache_ring()) {
        suffixPosition = param_ring_params_1();
    }
    let queryPosition = suffixPosition + query;
    var q = 0.0;
    if (lane < param_HEAD_DIM()) {
        q = load_queries((head * param_suffix_length() + query) * param_HEAD_DIM() + lane) * param_scale();
    }
    var accumulator = 0.0;
    var maxScore = -3.402823466e+38;
    var sum = 0.0;
    if (param_has_sinks()) {
        maxScore = load_sinks(head);
        sum = 1.0;
    }
    for (var key = 0u; key < param_sequence_length(); key++) {
        var useKey = true;
        var keyPosition = key;
        if (key >= prefix) {
            let suffixKey = key - prefix;
            keyPosition = suffixPosition + suffixKey;
            if (param_is_causal()) {
                useKey = suffixKey <= query;
            }
        } else if (param_is_kv_cache_ring()) {
            keyPosition = (prefix + key - param_ring_params()) % max(prefix, 1u);
            useKey = keyPosition < param_ring_params_1();
        }
        if (param_is_sliding_window()) {
            if (param_is_causal()) {
                useKey = useKey && keyPosition <= queryPosition &&
                         queryPosition - keyPosition < param_sliding_window_size();
            } else {
                let distance = max(keyPosition, queryPosition) - min(keyPosition, queryPosition);
                useKey = useKey && distance <= param_sliding_window_size() / 2u;
            }
        }
        if (!useKey) {
            continue;
        }
        var product = 0.0;
        if (lane < param_HEAD_DIM()) {
            product = q * load_keys(kvHead * param_k_head_stride() + key * param_k_seq_stride() + lane);
        }
        reduction[lane] = product;
        GroupMemoryBarrierWithGroupSync();
        for (var stride = param_lane_count() / 2u; stride > 0u; stride /= 2u) {
            if (lane < stride) {
                reduction[lane] += reduction[lane + stride];
            }
            GroupMemoryBarrierWithGroupSync();
        }
        let score = reduction[0];
        let nextMax = max(maxScore, score);
        let factor = exp(maxScore - nextMax);
        let probability = exp(score - nextMax);
        sum = sum * factor + probability;
        maxScore = nextMax;
        if (lane < param_HEAD_DIM()) {
            accumulator = accumulator * factor + probability * load_values(kvHead * param_v_head_stride() +
                                                                           key * param_v_seq_stride() + lane);
        }
        // All lanes must consume the reduced score before the next key reuses it.
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane < param_HEAD_DIM()) {
        store_out((query * param_num_heads() + head) * param_HEAD_DIM() + lane, accumulator / sum);
    }
}
