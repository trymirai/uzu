// uzu-widths: 1,2,4,8,16,32,64
// Bounded top-k selection. Normalization always includes the entire vocabulary.
// Tree reductions can round a nucleus boundary differently from a serial CPU
// sum; Philox integers, token indexing, ordering and cutoff rules are unchanged.
#include "abi.slang"
UZU_PARAM(bool, topk_merge)
UZU_PARAM(bool, write_final)
UZU_PARAM(bool, has_bitmask)
UZU_PARAM(bool, has_min_p)
UZU_PARAM(bool, has_temperature)
UZU_PARAM(bool, has_top_p)
UZU_PARAM(bool, is_stochastic)
UZU_PARAM(float, min_p)
UZU_PARAM(float, temperature)
UZU_PARAM(uint, top_k)
UZU_PARAM(float, top_p)
UZU_PARAM(uint, vocab_size)
UZU_PARAM(uint, topk_input_groups)
UZU_PARAM(uint, topk_output_groups)
UZU_BUFFER(bitmask)
UZU_BUFFER(logits)
UZU_BUFFER(output)
UZU_BUFFER(seeds)
UZU_BUFFER(topk_in)
UZU_BUFFER(topk_out)

groupshared float values[1024];
groupshared uint indices[1024];
groupshared float sums[256];
groupshared uint bad_values[256];

bool better(float value, uint index, float other_value, uint other_index) {
    return index != 0xffffffffu &&
           (other_index == 0xffffffffu || value > other_value || (value == other_value && index < other_index));
}

uint multiply_high(uint a, uint b) {
    uint a0 = a & 65535u;
    uint a1 = a >> 16u;
    uint b0 = b & 65535u;
    uint b1 = b >> 16u;
    uint w0 = a0 * b0;
    uint t = a1 * b0 + (w0 >> 16u);
    uint w1 = (t & 65535u) + a0 * b1;
    return a1 * b1 + (t >> 16u) + (w1 >> 16u);
}

float gumbel(uint index, uint batch) {
    uint block = index / 1024u;
    uint offset = ((param_vocab_size() + 4095u) / 4096u) * (index % 1024u) + block / 4u;
    uint4 counter = uint4(offset, 0u, 0u, 0u);
    uint2 key = uint2(word_seeds(batch * 2u), word_seeds(batch * 2u + 1u));
    for (uint round = 0u; round < 10u; round++) {
        counter = uint4(multiply_high(0xcd9e8d57u, counter.z) ^ counter.y ^ key.x, 0xcd9e8d57u * counter.z,
                        multiply_high(0xd2511f53u, counter.x) ^ counter.w ^ key.y, 0xd2511f53u * counter.x);
        key += uint2(0x9e3779b9u, 0xbb67ae85u);
    }
    float unit_sample = float(max(counter[block % 4u] >> 8u, 1u)) * (1.0 / 16777216.0);
    return -log(-log(unit_sample));
}

[shader("compute")][numthreads(256, 1, 1)]
void main(uint lane : SV_GroupIndex, uint3 group : SV_GroupID) {
    uint batch = group.y;
    uint record_words = WIDTH * 2u + 4u;
    uint local_bad = 0u;
    for (uint item = 0u; item < 4u; item++) {
        uint slot = lane + item * 256u;
        uint index = 0xffffffffu;
        float value = -3.402823466e38;
        if (param_topk_merge()) {
            uint input_group = group.x * (1024u / WIDTH) + slot / WIDTH;
            if (input_group < param_topk_input_groups()) {
                uint offset = (batch * param_topk_input_groups() + input_group) * record_words + (slot % WIDTH) * 2u;
                value = asfloat(word_topk_in(offset));
                index = word_topk_in(offset + 1u);
            }
        } else {
            uint token = group.x * 1024u + slot;
            if (token < param_vocab_size()) {
                bool allowed = true;
                if (param_has_bitmask()) {
                    uint mask = word_bitmask(batch * ((param_vocab_size() + 31u) / 32u) + token / 32u);
                    allowed = (mask & (1u << (token % 32u))) != 0u;
                }
                float raw = load_logits(batch * param_vocab_size() + token);
                if (allowed && (asuint(raw) & 0x7fffffffu) < 0x7f800000u) {
                    value = raw;
                    if (param_has_temperature()) value /= max(param_temperature(), 1.17549435e-38);
                    if ((asuint(value) & 0x7fffffffu) < 0x7f800000u) {
                        index = token;
                    } else {
                        local_bad = 1u;
                    }
                }
            }
        }
        values[slot] = value;
        indices[slot] = index;
    }
    GroupMemoryBarrierWithGroupSync();
    // Stable descending bitonic sort: equal transformed logits keep low IDs.
    for (uint size = 2u; size <= 1024u; size *= 2u) {
        for (uint distance = size / 2u; distance > 0u; distance /= 2u) {
            for (uint item = 0u; item < 4u; item++) {
                uint slot = lane + item * 256u;
                uint other = slot ^ distance;
                if (other > slot) {
                    bool exchange = (slot & size) == 0u
                        ? better(values[other], indices[other], values[slot], indices[slot])
                        : better(values[slot], indices[slot], values[other], indices[other]);
                    if (exchange) {
                        float value = values[slot];
                        uint index = indices[slot];
                        values[slot] = values[other];
                        indices[slot] = indices[other];
                        values[other] = value;
                        indices[other] = index;
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }
    float maximum = values[0];
    float local_sum = 0.0;
    if (param_topk_merge()) {
        for (uint item = lane; item < 1024u / WIDTH; item += 256u) {
            uint input_group = group.x * (1024u / WIDTH) + item;
            if (input_group < param_topk_input_groups()) {
                uint offset = (batch * param_topk_input_groups() + input_group) * record_words + WIDTH * 2u;
                local_bad |= word_topk_in(offset + 2u);
                if (param_has_top_p()) {
                    float input_sum = asfloat(word_topk_in(offset + 1u));
                    if (input_sum > 0.0) local_sum += input_sum * exp(asfloat(word_topk_in(offset)) - maximum);
                }
            }
        }
    } else if (param_has_top_p()) {
        for (uint item = 0u; item < 4u; item++) {
            uint slot = lane + item * 256u;
            if (indices[slot] != 0xffffffffu) local_sum += exp(values[slot] - maximum);
        }
    }
    sums[lane] = local_sum;
    bad_values[lane] = local_bad;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128u; stride > 0u; stride /= 2u) {
        if (lane < stride) {
            sums[lane] += sums[lane + stride];
            bad_values[lane] |= bad_values[lane + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    uint output_offset = (batch * param_topk_output_groups() + group.x) * record_words;
    if (!param_write_final() && lane < WIDTH) {
        put_topk_out(output_offset + lane * 2u, asuint(values[lane]));
        put_topk_out(output_offset + lane * 2u + 1u, indices[lane]);
    }
    if (lane == 0u) {
        put_topk_out(output_offset + WIDTH * 2u, asuint(maximum));
        put_topk_out(output_offset + WIDTH * 2u + 1u, asuint(sums[0]));
        put_topk_out(output_offset + WIDTH * 2u + 2u, bad_values[0]);
    }
    if (param_write_final()) {
        float normalizer = sums[0];
        GroupMemoryBarrierWithGroupSync();
        if (lane == 0u) {
            float mass = 0.0;
            for (uint rank = 0u; rank < param_top_k(); rank++) {
                sums[rank] = mass;
                if (param_has_top_p() && indices[rank] != 0xffffffffu)
                    mass += exp(values[rank] - maximum) / normalizer;
            }
        }
        GroupMemoryBarrierWithGroupSync();
        float score = -3.402823466e38;
        uint selected = 0xffffffffu;
        if (lane < param_top_k() && bad_values[0] == 0u) {
            uint index = indices[lane];
            float value = values[lane];
            if (index != 0xffffffffu && (!param_has_top_p() || sums[lane] < param_top_p()) &&
                (!param_has_min_p() || param_min_p() <= 0.0 || value >= maximum + log(param_min_p()))) {
                score = value;
                if (param_is_stochastic()) score += gumbel(index, batch);
                selected = index;
            }
        }
        values[lane] = score;
        indices[lane] = selected;
        GroupMemoryBarrierWithGroupSync();
        for (uint stride = 128u; stride > 0u; stride /= 2u) {
            if (lane < stride && better(values[lane + stride], indices[lane + stride], values[lane], indices[lane])) {
                values[lane] = values[lane + stride];
                indices[lane] = indices[lane + stride];
            }
            GroupMemoryBarrierWithGroupSync();
        }
        if (lane == 0u && bad_values[0] == 0u) put_output(batch, indices[0] == 0xffffffffu ? 0u : indices[0]);
    }
}
