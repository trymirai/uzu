// Slang compute kernel. ABI declarations are resolved by the WebGPU binding adapter.
#include "abi.slang"
UZU_PARAM(uint, columns)
UZU_PARAM(bool, has_bias)
UZU_PARAM(uint, ops)
UZU_BUFFER(bias)
UZU_BUFFER(fp_out)
UZU_BUFFER(input)
UZU_BUFFER(rht_factors)

[shader("compute")][numthreads(256, 1, 1)] void main(uint3 gid : SV_GroupID, uint waveLane : SV_WaveLaneIndex,
                                                     uint waveIndex : SV_WaveIndex,
                                                     uint waveWidth : SV_WaveLaneCount) {
    let lane = waveIndex * waveWidth + waveLane;

    let col = gid.x * 256u + lane;
    let row = gid.y;
    var v = 0.0;
    if (col < param_columns()) {
        v = load_input(row * param_columns() + col);
        if (param_ops() != 1u) {
            v *= load_rht_factors(col);
        }
    }

    [ForceUnroll] for (uint stride = 1; stride < 32; stride *= 2) {
        let paired = WaveReadLaneAt(v, waveLane ^ stride);
        v = (waveLane & stride) != 0 ? paired - v : v + paired;
    }
    v *= 0.1767766952966369;
    if (param_ops() == 1u && col < param_columns()) {
        v *= load_rht_factors(col);
    }
    if (param_has_bias() && col < param_columns()) {
        v = round_fp_out(v) + load_bias(col);
    }
    if (col < param_columns()) store_fp_out(row * param_columns() + col, v);
}
