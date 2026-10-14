// Slang compute kernel. ABI declarations are resolved by the WebGPU binding adapter.
#include "abi.slang"
UZU_PARAM(bool, copy_to_shortcut)
UZU_PARAM(uint, element_count)
UZU_PARAM(float, epsilon)
UZU_PARAM(bool, full_layer)
UZU_PARAM(bool, has_biases)
UZU_PARAM(bool, has_scales)
UZU_PARAM(float, post_layer_scalar)
UZU_PARAM(bool, residual_add)
UZU_PARAM(float, scale_offset)
UZU_PARAM(bool, scale_output)
UZU_PARAM(bool, scale_residual_sum)
UZU_PARAM(bool, subtract_mean)
UZU_PARAM(bool, use_hadamard)
UZU_BUFFER(biases)
UZU_BUFFER(hadamard_factors)
UZU_BUFFER(input)
UZU_BUFFER(output)
UZU_BUFFER(scales)
UZU_BUFFER(shortcut)

// Subgroup variants run only on known widths of at least 32 lanes.
groupshared float sum_values[8];
groupshared float sum_squares[8];
[shader("compute")][numthreads(256, 1, 1)] void main(uint3 gid : SV_GroupID, uint waveLane : SV_WaveLaneIndex,
                                                     uint waveIndex : SV_WaveIndex,
                                                     uint waveWidth : SV_WaveLaneCount) {
    let lane = waveIndex * waveWidth + waveLane;

    let base = gid.x * param_element_count();
    var sum = 0.0;
    var sum_sq = 0.0;
    for (var i = lane; i < param_element_count(); i += 256u) {
        var v = load_input(base + i);
        if (param_copy_to_shortcut()) {
            if (param_residual_add()) {
                v = round_shortcut(v + load_shortcut(base + i));
                if (param_scale_residual_sum()) {
                    v = round_shortcut(v * param_post_layer_scalar());
                }
            }
            store_shortcut(base + i, v);
        }
        sum += v;
        sum_sq += v * v;
    }
    let waveSum = WaveActiveSum(sum);
    let waveSquares = WaveActiveSum(sum_sq);
    if (waveLane == 0u) {
        sum_values[waveIndex] = waveSum;
        sum_squares[waveIndex] = waveSquares;
    }
    GroupMemoryBarrierWithGroupSync();
    // Shortcut reads below revisit each invocation's own stores. Only the
    // shared reduction crosses invocations, so it needs the workgroup barrier.
    let numWaves = (256u + waveWidth - 1u) / waveWidth;
    let total = WaveActiveSum(waveLane < numWaves ? sum_values[waveLane] : 0.0);
    let totalSquares = WaveActiveSum(waveLane < numWaves ? sum_squares[waveLane] : 0.0);
    var mean = 0.0;
    if (param_subtract_mean()) {
        mean = total / float(param_element_count());
    }
    let rms = rsqrt(totalSquares / float(param_element_count()) - mean * mean + param_epsilon());
    for (var tile = 0u; tile < param_element_count(); tile += 256u) {
        let i = tile + lane;
        var result = 0.0;
        if (i < param_element_count()) {
            var v = load_input(base + i);
            if (param_residual_add()) {
                v = load_shortcut(base + i);
            }
            result = (v - mean) * rms;
            if (param_has_scales()) {
                let scale = load_scales(i) + param_scale_offset();
                if (param_full_layer()) {
                    result *= scale;
                } else {
                    result = round_output(result) * round_output(scale);
                }
            }
            result = round_output(result);
            if (param_has_biases()) {
                result = round_output(result + load_biases(i));
            }
            if (param_use_hadamard()) {
                result *= load_hadamard_factors(i);
            }
        }
        if (param_use_hadamard()) {
            [ForceUnroll] for (uint stride = 1; stride < 32; stride *= 2) {
                let paired = WaveReadLaneAt(result, waveLane ^ stride);
                result = (waveLane & stride) != 0 ? paired - result : result + paired;
            }
            result = round_output(result * 0.1767766952966369);
        }
        if (param_scale_output()) {
            result *= round_output(param_post_layer_scalar());
        }
        if (i < param_element_count()) store_output(base + i, result);
    }
}
