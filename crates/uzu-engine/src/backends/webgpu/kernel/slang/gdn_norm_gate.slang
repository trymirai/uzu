// Slang compute kernel. ABI declarations are resolved by the WebGPU binding adapter.
#include "abi.slang"
UZU_PARAM(uint, conv_dim)
UZU_PARAM(uint, head_v_dim)
UZU_PARAM(float, norm_epsilon)
UZU_PARAM(uint, total_proj_dim)
UZU_PARAM(uint, value_dim)
UZU_BUFFER(in_out)
UZU_BUFFER(in_proj)
UZU_BUFFER(norm_weight)

groupshared float sums[256];
[shader("compute")][numthreads(256, 1, 1)] void main(uint lane : SV_GroupIndex, uint3 group : SV_GroupID) {
    let base = group.y * param_value_dim() + group.x * param_head_v_dim();
    var sum = 0.0;
    for (var i = lane; i < param_head_v_dim(); i += 256u) {
        let value = load_in_out(base + i);
        sum += value * value;
    }
    sums[lane] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (var stride = 128u; stride > 0u; stride /= 2u) {
        if (lane < stride) {
            sums[lane] += sums[lane + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    let inv = rsqrt(sums[0] / float(param_head_v_dim()) + param_norm_epsilon());
    for (var i = lane; i < param_head_v_dim(); i += 256u) {
        let z = load_in_proj(group.y * param_total_proj_dim() + param_conv_dim() +
                             group.x * param_head_v_dim() + i);
        store_in_out(base + i, load_in_out(base + i) * inv * load_norm_weight(i) * activate(z, 0u, 1.0));
    }
}
