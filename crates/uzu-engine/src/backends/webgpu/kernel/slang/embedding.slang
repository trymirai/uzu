// Slang compute kernel. ABI declarations are resolved by the WebGPU binding adapter.
#include "abi.slang"
UZU_PARAM(uint, group_size)
UZU_PARAM(float, input_scale)
UZU_PARAM(uint, model_dim)
UZU_PARAM(uint, quantization_method)
UZU_PARAM(uint, quantization_mode)
UZU_PARAM(uint, table_kind)
UZU_PARAM(bool, use_hadamard)
UZU_PARAM(uint, vocab_size)
UZU_BUFFER(biases)
UZU_BUFFER(hadamard_factors)
UZU_BUFFER(output)
UZU_BUFFER(scales)
UZU_BUFFER(token_ids)
UZU_BUFFER(values)
UZU_BUFFER(zero_points)

groupshared float h[256];
func code4(index : uint) -> float {
    let raw = uint(load_values(index / 2u));
    return float((raw >> ((index % 2u) * 4u)) & 15u);
}
[shader("compute")][numthreads(256, 1, 1)] void main(uint lane : SV_GroupIndex, uint3 gid : SV_GroupID) {
    let col = gid.x * 256u + lane;
    let batch = gid.y;
    let token = word_token_ids(batch);
    var v = 0.0;
    if (col < param_model_dim() && token < param_vocab_size()) {
        if (param_table_kind() == 0u) {
            v = round_output(load_values(token * param_model_dim() + col) *
                             round_output(param_input_scale()));
        }
        if (param_table_kind() == 1u) {
            let groups = (param_model_dim() + param_group_size() - 1u) / param_group_size();
            let index = token * groups + col / param_group_size();
            let scale = load_scales(index);
            var code = 0.0;
            if (param_quantization_mode() == 0u) {
                code = code4(token * param_model_dim() + col);
            } else {
                let raw = uint(load_values(token * param_model_dim() + col));
                code = float(raw);
                if (param_quantization_mode() == 1u) {
                    code = float(asint(raw << 24u) >> 24);
                }
            }
            var bias = 0.0;
            if (param_quantization_method() == 0u) {
                bias = load_biases(index);
            }
            if (param_quantization_method() == 1u) {
                var zero = load_zero_points(index);
                if (param_quantization_mode() == 0u) {
                    let nibble = token * ((groups + 1u) / 2u) * 2u + col / param_group_size();
                    zero = float((uint(load_zero_points(nibble / 2u)) >> ((nibble % 2u) * 4u)) & 15u);
                }
                bias = -scale * zero;
            }
            if (param_quantization_method() == 2u) {
                bias = -scale * choose(128.0, 8.0, param_quantization_mode() == 0u);
            }
            v = round_output((scale * code + bias) * param_input_scale());
        }
    }
    if (param_use_hadamard()) {
        h[lane] = v;
        GroupMemoryBarrierWithGroupSync();
        for (var stride = 1u; stride < 32u; stride *= 2u) {
            let own = h[lane];
            let paired = h[lane ^ stride];
            GroupMemoryBarrierWithGroupSync();
            h[lane] = choose(own + paired, paired - own, (lane & stride) != 0u);
            GroupMemoryBarrierWithGroupSync();
        }
        v = h[lane] * 0.1767766952966369 * load_hadamard_factors(col);
    }
    if (col < param_model_dim()) {
        store_output(batch * param_model_dim() + col, v);
    }
}
