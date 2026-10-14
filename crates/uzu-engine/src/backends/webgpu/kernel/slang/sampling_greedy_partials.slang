// uzu-widths: 4,16
// Plain greedy argmax, partitioned across independent vocabulary ranges.
#include "abi.slang"
UZU_PARAM(bool, has_bitmask)
UZU_PARAM(bool, write_final)
UZU_PARAM(uint, partial_count)
UZU_PARAM(uint, vocab_size)
UZU_BUFFER(bitmask)
UZU_BUFFER(logits)
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
    uint batch = group.y;
    uint start = group.x * (256u * WIDTH);
    uint best_key = 0u;
    uint best_index = 0xffffffffu;
    uint item_count = param_write_final() ? (param_vocab_size() - 1u) / 256u + 1u : WIDTH;
    for (uint item = 0u; item < item_count; item++) {
        uint index = start + item * 256u + lane;
        if (index >= param_vocab_size()) continue;
        uint bits = asuint(load_logits(batch * param_vocab_size() + index));
        if (param_has_bitmask()) {
            uint mask = word_bitmask(batch * ((param_vocab_size() + 31u) / 32u) + index / 32u);
            if ((mask & (1u << (index % 32u))) == 0u) bits = 0xff800000u;
        }
        uint magnitude = bits & 0x7fffffffu;
        uint key;
        if (magnitude > 0x7f800000u) {
            // CPU max_by folds in ascending index order. Its unordered compares
            // keep the earlier index: NaN at zero wins, later NaNs never do.
            if (index != 0u) continue;
            key = 0xffffffffu;
        } else {
            // Ordered IEEE keys preserve infinities and subnormals without
            // arithmetic. Signed zero is one tie, just as in f32::partial_cmp.
            if (magnitude == 0u) bits = 0u;
            key = (bits & 0x80000000u) != 0u ? ~bits : bits ^ 0x80000000u;
        }
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
    if (lane == 0u) {
        if (param_write_final()) {
            put_output(batch, indices[0]);
        } else {
            uint offset = (batch * param_partial_count() + group.x) * 2u;
            put_partials(offset, keys[0]);
            put_partials(offset + 1u, indices[0]);
        }
    }
}
