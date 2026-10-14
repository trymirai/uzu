// Reduce per-range ordered maxima into one greedy token for each batch item.
#include "abi.slang"
UZU_PARAM(uint, partial_count)
UZU_BUFFER(output)
UZU_BUFFER(partials)

groupshared uint keys[256];
groupshared uint indices[256];

bool better(uint key, uint index, uint other_key, uint other_index) {
    return index != 0xffffffffu &&
           (other_index == 0xffffffffu || key > other_key || (key == other_key && index < other_index));
}

[shader("compute")][numthreads(256, 1, 1)]
void main(uint lane : SV_GroupIndex, uint3 group : SV_GroupID) {
    uint batch = group.x;
    uint best_key = 0u;
    uint best_index = 0xffffffffu;
    for (uint part = lane; part < param_partial_count(); part += 256u) {
        uint offset = (batch * param_partial_count() + part) * 2u;
        uint key = word_partials(offset);
        uint index = word_partials(offset + 1u);
        if (better(key, index, best_key, best_index)) {
            best_key = key;
            best_index = index;
        }
    }
    keys[lane] = best_key;
    indices[lane] = best_index;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128u; stride > 0u; stride /= 2u) {
        if (lane < stride && better(keys[lane + stride], indices[lane + stride], keys[lane], indices[lane])) {
            keys[lane] = keys[lane + stride];
            indices[lane] = indices[lane + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0u) put_output(batch, indices[0]);
}
