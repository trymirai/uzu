// uzu-widths: 2, 4, 32
// Lane-sliced W4 GEMV: multiple outputs share each 16-value input slice. Quantization
// scales are folded after the partial dot product, as in the native GEMV path.
// Width 32 uses eight waves of four outputs and fuses the H32 output transform.
#include "abi.slang"
#if WIDTH == 32
#define OUTPUTS 4
#define THREADS 256
#else
#define OUTPUTS WIDTH
#define THREADS 64
#endif
UZU_PARAM(uint, w4_group_size)
UZU_PARAM(bool, input_bf16_words)
UZU_PARAM(uint, correction)
UZU_PARAM(bool, gather)
UZU_PARAM(bool, has_bias)
UZU_PARAM(bool, has_soft_cap)
UZU_PARAM(uint, k)
UZU_PARAM(uint, n)
UZU_PARAM(float, output_scale)
UZU_PARAM(uint, row_stride)
UZU_PARAM(uint, scale_group_stride)
UZU_PARAM(uint, scale_output_stride)
UZU_PARAM(bool, signed_codes)
UZU_PARAM(float, soft_cap)
UZU_PARAM(uint, zp_group_stride)
UZU_PARAM(uint, zp_output_stride)
UZU_BUFFER(a)
UZU_BUFFER(bias)
UZU_BUFFER(codes)
UZU_BUFFER(d)
UZU_BUFFER(rht_factors)
UZU_BUFFER(gather)
UZU_BUFFER(quant_biases)
UZU_BUFFER(scales)
UZU_BUFFER(zero_points)

float3 metadata(uint column, uint group) {
    uint index = column * param_scale_output_stride() + group * param_scale_group_stride();
    float scale = load_scales(index);
    float point = 8.0;
    float bias = 0.0;
    if (param_correction() == 1u) {
        point = 0.0;
        bias = load_quant_biases(index);
    } else if (param_correction() == 2u) {
        uint offset = column * param_zp_output_stride() + group * param_zp_group_stride();
        point = float((word_zero_points(offset / 8u) >> ((offset % 8u) * 4u)) & 15u);
    }
    return float3(scale, 8388608.0 + point, bias);
}

float transform(uint column, float value) {
    value *= param_output_scale();
    #if WIDTH != 32
    if (param_has_bias()) value += load_bias(column);
    #endif
    if (param_has_soft_cap()) value = param_soft_cap() * tanh(value / param_soft_cap());
    return value;
}

#if WIDTH == 32
groupshared float output_tile[32];
#endif

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void main(uint3 group: SV_GroupID, uint lane: SV_WaveLaneIndex, uint subgroup: SV_WaveIndex) {
    // Selected only when the adapter reports a fixed 32-lane subgroup.
    uint row = group.y;
    uint column = (group.x + group.z * 65535u) * (THREADS / 32u * OUTPUTS) + subgroup * OUTPUTS;
    uint columns[OUTPUTS];
    [ForceUnroll]
    for (uint outputIndex = 0; outputIndex < OUTPUTS; outputIndex++) {
        columns[outputIndex] = min(column + outputIndex, param_n() - 1u);
        if (param_gather()) columns[outputIndex] = word_gather(row * param_n() + columns[outputIndex]);
    }
    uint groupLanes = param_w4_group_size() / 16u;
    uint groupCount = param_k() / param_w4_group_size();
    uint slice = (lane % groupLanes) * 16u;
    vector<float, OUTPUTS> result = 0.0;
    for (uint quantGroup = lane / groupLanes; quantGroup < groupCount; quantGroup += 32u / groupLanes) {
        uint inner = quantGroup * param_w4_group_size() + slice;
        float3 quant[OUTPUTS];
        [ForceUnroll]
        for (uint outputIndex = 0; outputIndex < OUTPUTS; outputIndex++)
            quant[outputIndex] = metadata(columns[outputIndex], quantGroup);
        vector<float, OUTPUTS> partial = 0.0;
        float inputSum = 0.0;
        [ForceUnroll]
        for (uint chunk = 0; chunk < 2; chunk++) {
            float inputs[8];
            if (param_input_bf16_words()) {
                [ForceUnroll]
                for (uint pair = 0; pair < 4; pair++) {
                    uint bits = word_a((row * param_k() + inner) / 2u + chunk * 4u + pair);
                    inputs[pair * 2u] = asfloat(bits << 16u);
                    inputs[pair * 2u + 1u] = asfloat(bits & 0xffff0000u);
                }
            } else {
                [ForceUnroll]
                for (uint value = 0; value < 8; value++)
                    inputs[value] = load_a(row * param_k() + inner + chunk * 8u + value);
            }
            [ForceUnroll]
            for (uint outputIndex = 0; outputIndex < OUTPUTS; outputIndex++) {
                uint codes = word_codes((columns[outputIndex] * param_k() + inner) / 8u + chunk);
                if (param_signed_codes()) codes ^= 0x88888888u;
                [ForceUnroll]
                for (uint value = 0; value < 8; value++) {
                    float weight = asfloat(((codes >> (value * 4u)) & 15u) | 0x4b000000u) - quant[outputIndex].y;
                    partial[outputIndex] = fma(inputs[value], weight, partial[outputIndex]);
                }
            }
            if (param_correction() == 1u) {
                [ForceUnroll]
                for (uint value = 0; value < 8; value++) inputSum += inputs[value];
            }
        }
        [ForceUnroll]
        for (uint outputIndex = 0; outputIndex < OUTPUTS; outputIndex++) {
            result[outputIndex] = fma(quant[outputIndex].x, partial[outputIndex], result[outputIndex]);
            if (param_correction() == 1u)
                result[outputIndex] = fma(quant[outputIndex].z, inputSum, result[outputIndex]);
        }
    }
    vector<float, OUTPUTS> total = WaveActiveSum(result);
    #if WIDTH == 32
    if (lane == 0u) {
        [ForceUnroll]
        for (uint outputIndex = 0; outputIndex < OUTPUTS; outputIndex++)
            output_tile[subgroup * OUTPUTS + outputIndex] = round_d(transform(column + outputIndex, total[outputIndex]));
    }
    GroupMemoryBarrierWithGroupSync();
    float value = output_tile[lane];
    [ForceUnroll]
    for (uint stride = 1u; stride < 32u; stride *= 2u) {
        float paired = WaveReadLaneAt(value, lane ^ stride);
        value = (lane & stride) != 0u ? paired - value : value + paired;
    }
    uint outputColumn = (group.x + group.z * 65535u) * 32u + lane;
    if (subgroup == 0u && outputColumn < param_n()) {
        value *= 0.1767766952966369;
        value *= load_rht_factors(outputColumn);
        if (param_has_bias()) value = round_d(value) + load_bias(outputColumn);
        store_d(row * param_row_stride() + outputColumn, value);
    }
    #else
    if (lane == 0u) {
        [ForceUnroll]
        for (uint outputIndex = 0; outputIndex < OUTPUTS; outputIndex++) {
            if (column + outputIndex < param_n())
                store_d(row * param_row_stride() + column + outputIndex,
                        transform(column + outputIndex, total[outputIndex]));
        }
    }
    #endif
}
