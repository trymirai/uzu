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

groupshared float sum_values[256];
groupshared float sum_squares[256];
[shader("compute")][numthreads(256, 1, 1)] void main(uint lane : SV_GroupIndex, uint3 gid : SV_GroupID) {
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
    sum_values[lane] = sum;
    sum_squares[lane] = sum_sq;
    GroupMemoryBarrierWithGroupSync();
    DeviceMemoryBarrierWithGroupSync();
    for (var stride = 128u; stride > 0u; stride /= 2u) {
        if (lane < stride) {
            sum_values[lane] += sum_values[lane + stride];
            sum_squares[lane] += sum_squares[lane + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    var mean = 0.0;
    if (param_subtract_mean()) {
        mean = sum_values[0] / float(param_element_count());
    }
    let rms = rsqrt(sum_squares[0] / float(param_element_count()) - mean * mean + param_epsilon());
    GroupMemoryBarrierWithGroupSync();
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
            sum_values[lane] = result;
            GroupMemoryBarrierWithGroupSync();
            for (var stride = 1u; stride < 32u; stride *= 2u) {
                let paired = sum_values[lane ^ stride];
                let own = sum_values[lane];
                GroupMemoryBarrierWithGroupSync();
                sum_values[lane] = choose(own + paired, paired - own, (lane & stride) != 0u);
                GroupMemoryBarrierWithGroupSync();
            }
            result = round_output(sum_values[lane] * 0.1767766952966369);
        }
        if (param_scale_output()) {
            result *= round_output(param_post_layer_scalar());
        }
        if (i < param_element_count()) {
            store_output(base + i, result);
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
