// Slang compute kernel. ABI declarations are resolved by the WebGPU binding adapter.
#include "abi.slang"
UZU_PARAM(uint, act_type)
UZU_PARAM(float, activation_alpha)
UZU_PARAM(bool, clip_gate)
UZU_PARAM(bool, clip_value)
UZU_PARAM(uint, columns)
UZU_PARAM(bool, custom_activation_alpha)
UZU_PARAM(float, gate_clip_max)
UZU_PARAM(float, gate_clip_min)
UZU_PARAM(uint, gated_dim)
UZU_PARAM(bool, interleaved)
UZU_PARAM(bool, use_hadamard)
UZU_PARAM(float, value_clip_max)
UZU_PARAM(float, value_clip_min)
UZU_PARAM(uint, value_offset)
UZU_PARAM(uint, value_row_stride)
UZU_BUFFER(act_operand)
UZU_BUFFER(fp_out)
UZU_BUFFER(hadamard_factors)
UZU_BUFFER(value_operand)

[shader("compute")][numthreads(256, 1, 1)] void main(uint3 gid : SV_GroupID, uint waveLane : SV_WaveLaneIndex,
                                                     uint waveIndex : SV_WaveIndex,
                                                     uint waveWidth : SV_WaveLaneCount) {
    let lane = waveIndex * waveWidth + waveLane;

    let col = gid.x * 256u + lane;
    let row = gid.y;
    var v = 0.0;
    if (col < param_columns()) {
        var gate = load_act_operand(row * param_gated_dim() + col);
        var value = load_value_operand(row * param_value_row_stride() + param_value_offset() + col);
        if (param_interleaved()) {
            gate = load_act_operand(row * 2u * param_gated_dim() + param_gated_dim() + col);
            value = load_act_operand(row * 2u * param_gated_dim() + col);
        }
        if (param_clip_gate()) {
            gate = round_fp_out(clamp(gate, param_gate_clip_min(), param_gate_clip_max()));
        }
        if (param_clip_value()) {
            value = round_fp_out(clamp(value, param_value_clip_min(), param_value_clip_max()));
        }
        var alpha = 1.0;
        if (param_custom_activation_alpha()) {
            alpha = param_activation_alpha();
        }
        v = round_fp_out(value * round_fp_out(activate(gate, param_act_type(), alpha)));
        if (param_use_hadamard()) {
            v *= load_hadamard_factors(col);
        }
    }
    if (param_use_hadamard()) {
        [ForceUnroll] for (uint stride = 1; stride < 32; stride *= 2) {
            let paired = WaveReadLaneAt(v, waveLane ^ stride);
            v = (waveLane & stride) != 0 ? paired - v : v + paired;
        }
        v *= 0.1767766952966369;
    }
    if (col < param_columns()) store_fp_out(row * param_columns() + col, v);
}
