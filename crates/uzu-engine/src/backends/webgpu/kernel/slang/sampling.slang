// Slang compute kernel. ABI declarations are resolved by the WebGPU binding adapter.
#include "abi.slang"
UZU_PARAM(bool, has_bitmask)
UZU_PARAM(bool, has_min_p)
UZU_PARAM(bool, has_temperature)
UZU_PARAM(bool, has_top_k)
UZU_PARAM(bool, has_top_p)
UZU_PARAM(bool, is_stochastic)
UZU_PARAM(float, min_p)
UZU_PARAM(float, temperature)
UZU_PARAM(uint, top_k)
UZU_PARAM(float, top_p)
UZU_PARAM(uint, vocab_size)
UZU_PARAM(bool, topk_fallback)
UZU_PARAM(uint, topk_record_words)
UZU_BUFFER(bitmask)
UZU_BUFFER(logits)
UZU_BUFFER(output)
UZU_BUFFER(seeds)
UZU_BUFFER(topk_status)

struct Candidate {
    float value;
    uint index;
};
groupshared float shared_values[256];
groupshared uint shared_indices[256];
groupshared float shared_candidate_logit;
groupshared float reduced_value;
groupshared uint reduced_index;

func better(value : float, index : uint, other_value : float, other_index : uint) -> bool {
    return value > other_value || (value == other_value && index < other_index);
}

func reduce_best(value : float, index : uint, lane : uint) -> Candidate {
    shared_values[lane] = value;
    shared_indices[lane] = index;
    GroupMemoryBarrierWithGroupSync();
    for (var stride = 128u; stride > 0u; stride /= 2u) {
        if (lane < stride && better(shared_values[lane + stride], shared_indices[lane + stride],
                                    shared_values[lane], shared_indices[lane])) {
            shared_values[lane] = shared_values[lane + stride];
            shared_indices[lane] = shared_indices[lane + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0u) {
        reduced_value = shared_values[0];
        reduced_index = shared_indices[0];
    }
    let best_value = workgroupUniformLoad(reduced_value);
    let best_index = workgroupUniformLoad(reduced_index);
    return Candidate(best_value, best_index);
}

func reduce_sum(value : float, lane : uint) -> float {
    shared_values[lane] = value;
    GroupMemoryBarrierWithGroupSync();
    for (var stride = 128u; stride > 0u; stride /= 2u) {
        if (lane < stride) {
            shared_values[lane] += shared_values[lane + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0u) {
        reduced_value = shared_values[0];
    }
    return workgroupUniformLoad(reduced_value);
}

func reduce_count(value : uint, lane : uint) -> uint {
    shared_indices[lane] = value;
    GroupMemoryBarrierWithGroupSync();
    for (var stride = 128u; stride > 0u; stride /= 2u) {
        if (lane < stride) {
            shared_indices[lane] += shared_indices[lane + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0u) {
        reduced_index = shared_indices[0];
    }
    return workgroupUniformLoad(reduced_index);
}

func multiply_high(a : uint, b : uint) -> uint {
    let a0 = a & 65535u;
    let a1 = a >> 16u;
    let b0 = b & 65535u;
    let b1 = b >> 16u;
    let w0 = a0 * b0;
    let t = a1 * b0 + (w0 >> 16u);
    let w1 = (t & 65535u) + a0 * b1;
    return a1 * b1 + (t >> 16u) + (w1 >> 16u);
}

func gumbel(index : uint, batch : uint) -> float {
    let block = index / 1024u;
    let offset = ((param_vocab_size() + 4095u) / 4096u) * (index % 1024u) + block / 4u;
    var counter = uint4(offset, 0u, 0u, 0u);
    var key = uint2(word_seeds(batch * 2u), word_seeds(batch * 2u + 1u));
    for (var round = 0u; round < 10u; round++) {
        counter = uint4(multiply_high(0xcd9e8d57u, counter.z) ^ counter.y ^ key.x, 0xcd9e8d57u * counter.z,
                        multiply_high(0xd2511f53u, counter.x) ^ counter.w ^ key.y, 0xd2511f53u * counter.x);
        key += uint2(0x9e3779b9u, 0xbb67ae85u);
    }
    let unit_sample = float(max(counter[block % 4u] >> 8u, 1u)) * (1.0 / 16777216.0);
    return -log(-log(unit_sample));
}

func sample_score(value : float, index : uint, batch : uint) -> float {
    if (param_is_stochastic()) {
        return value + gumbel(index, batch);
    }
    return value;
}

func allowed(index : uint, batch : uint) -> bool {
    if (param_has_bitmask()) {
        let mask = word_bitmask(batch * ((param_vocab_size() + 31u) / 32u) + index / 32u);
        if ((mask & (1u << (index % 32u))) == 0u) {
            return false;
        }
    }
    // NaNs and infinities cannot define a normalized sampling distribution.
    return (asuint(load_logits(batch * param_vocab_size() + index)) & 0x7fffffffu) < 0x7f800000u;
}

func raw_logit(index : uint, batch : uint) -> float {
    var value = load_logits(batch * param_vocab_size() + index);
    if (param_has_temperature()) {
        value /= max(param_temperature(), 1.17549435e-38);
    }
    return value;
}

[shader("compute")][numthreads(256, 1, 1)] void main(uint lane : SV_GroupIndex, uint3 group : SV_GroupID) {
    let batch = group.x;
    if (param_topk_fallback()) {
        if (lane == 0u) {
            reduced_index = word_topk_status((batch + 1u) * param_topk_record_words() - 2u);
        }
        if (workgroupUniformLoad(reduced_index) == 0u) return;
    }
    var local_max = -3.402823466e38;
    var local_norm = 0.0;
    var best = Candidate(-3.402823466e38, 0xffffffffu);
    for (var index = lane; index < param_vocab_size(); index += 256u) {
        if (!allowed(index, batch)) {
            continue;
        }
        let value = raw_logit(index, batch);
        if (param_has_top_p()) {
            let next_max = max(local_max, value);
            local_norm = local_norm * exp(local_max - next_max) + exp(value - next_max);
            local_max = next_max;
        } else {
            local_max = max(local_max, value);
        }
        let score = sample_score(value, index, batch);
        if (better(score, index, best.value, best.index)) {
            best = Candidate(score, index);
        }
    }
    var maximum = 0.0;
    var normalizer = 1.0;
    if (param_has_top_p() || param_has_min_p()) {
        maximum = reduce_best(local_max, lane, lane).value;
    }
    if (param_has_top_p()) {
        normalizer = reduce_sum(local_norm * exp(local_max - maximum), lane);
    }
    var candidate = reduce_best(best.value, best.index, lane);
    if (!(param_has_top_k() || param_has_top_p() || param_has_min_p())) {
        if (lane == 0u) {
            put_output(batch, choose(candidate.index, 0u, candidate.index == 0xffffffffu));
        }
        return;
    }
    while (true) {
        if (candidate.index == 0xffffffffu) {
            if (lane == 0u) {
                put_output(batch, 0u);
            }
            return;
        }
        if (lane == 0u) {
            shared_candidate_logit = raw_logit(candidate.index, batch);
        }
        let candidate_logit = workgroupUniformLoad(shared_candidate_logit);
        var count = 0u;
        var mass = 0.0;
        var next = Candidate(-3.402823466e38, 0xffffffffu);
        for (var index = lane; index < param_vocab_size(); index += 256u) {
            if (!allowed(index, batch)) {
                continue;
            }
            let value = raw_logit(index, batch);
            if (!better(value, index, candidate_logit, candidate.index)) {
                continue;
            }
            count += 1u;
            if (param_has_top_p()) {
                mass += exp(value - maximum) / normalizer;
            }
            let score = sample_score(value, index, batch);
            if (better(score, index, next.value, next.index)) {
                next = Candidate(score, index);
            }
        }
        let count_above = reduce_count(count, lane);
        let mass_above = reduce_sum(mass, lane);
        let next_candidate = reduce_best(next.value, next.index, lane);
        var accepted = true;
        if (param_has_top_k() && count_above >= param_top_k()) {
            accepted = false;
        }
        if (param_has_top_p() && mass_above >= param_top_p()) {
            accepted = false;
        }
        if (param_has_min_p() && param_min_p() > 0.0 && candidate_logit < maximum + log(param_min_p())) {
            accepted = false;
        }
        if (accepted) {
            if (lane == 0u) {
                put_output(batch, candidate.index);
            }
            return;
        }
        candidate = next_candidate;
    }
}
