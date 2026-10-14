// Slang compute kernel. ABI declarations are resolved by the WebGPU binding adapter.
#include "abi.slang"
UZU_PARAM(uint, columns)
UZU_PARAM(bool, has_bias)
UZU_PARAM(uint, ops)
UZU_BUFFER(bias)
UZU_BUFFER(fp_out)
UZU_BUFFER(input)
UZU_BUFFER(rht_factors)

groupshared float h[256];
[shader("compute")][numthreads(256, 1, 1)] void main(uint lane : SV_GroupIndex, uint3 gid : SV_GroupID) {
    let col = gid.x * 256u + lane;
    let row = gid.y;
    var v = 0.0;
    if (col < param_columns()) {
        v = load_input(row * param_columns() + col);
        if (param_ops() != 1u) {
            v *= load_rht_factors(col);
        }
    }

    h[lane] = v;
    GroupMemoryBarrierWithGroupSync();
    for (var stride = 1u; stride < 32u; stride *= 2u) {
        let own = h[lane];
        let paired = h[lane ^ stride];
        GroupMemoryBarrierWithGroupSync();
        h[lane] = choose(own + paired, paired - own, (lane & stride) != 0u);
        GroupMemoryBarrierWithGroupSync();
    }
    v = h[lane] * 0.1767766952966369;
    if (param_ops() == 1u && col < param_columns()) {
        v *= load_rht_factors(col);
    }
    if (param_has_bias() && col < param_columns()) {
        v = round_fp_out(v) + load_bias(col);
    }
    if (col < param_columns()) {
        store_fp_out(row * param_columns() + col, v);
    }
}
