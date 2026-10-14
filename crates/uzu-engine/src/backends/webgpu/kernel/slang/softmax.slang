// Slang compute kernel. ABI declarations are resolved by the WebGPU binding adapter.
#include "abi.slang"
UZU_PARAM(uint, batch_dim)
UZU_PARAM(bool, has_sinks)
UZU_PARAM(uint, row_length)
UZU_BUFFER(sinks)
UZU_BUFFER(values)

groupshared float reduction[256];
[shader("compute")][numthreads(256, 1, 1)] void main(uint lane : SV_GroupIndex, uint3 group : SV_GroupID) {
    let base = group.x * param_row_length();
    var maximum = -3.402823466e38;
    for (var i = lane; i < param_row_length(); i += 256u) {
        maximum = max(maximum, load_values(base + i));
    }
    if (param_has_sinks()) {
        maximum = max(maximum, load_sinks(group.x / param_batch_dim()));
    }
    reduction[lane] = maximum;
    GroupMemoryBarrierWithGroupSync();
    for (var stride = 128u; stride > 0u; stride /= 2u) {
        if (lane < stride) {
            reduction[lane] = max(reduction[lane], reduction[lane + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    maximum = reduction[0];
    var total = 0.0;
    for (var i = lane; i < param_row_length(); i += 256u) {
        total += exp(load_values(base + i) - maximum);
    }
    if (param_has_sinks() && lane == 0u) {
        total += exp(load_sinks(group.x / param_batch_dim()) - maximum);
    }
    GroupMemoryBarrierWithGroupSync();
    reduction[lane] = total;
    GroupMemoryBarrierWithGroupSync();
    for (var stride = 128u; stride > 0u; stride /= 2u) {
        if (lane < stride) {
            reduction[lane] += reduction[lane + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    total = reduction[0];
    for (var i = lane; i < param_row_length(); i += 256u) {
        store_values(base + i, round_values(exp(load_values(base + i) - maximum)) / total);
    }
}
