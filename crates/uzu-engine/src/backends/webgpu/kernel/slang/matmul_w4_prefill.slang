// Each subgroup computes four rows by four columns. Packed weights and their
// metadata are reused across the rows; inputs are reused across the columns.
#include "abi.slang"
UZU_PARAM(uint, w4_group_size)
UZU_PARAM(bool, input_bf16_words)
UZU_PARAM(uint, correction)
UZU_PARAM(bool, has_bias)
UZU_PARAM(bool, has_soft_cap)
UZU_PARAM(uint, k)
UZU_PARAM(uint, m)
UZU_PARAM(uint, n)
UZU_PARAM(uint, prefill_column_pages)
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

[shader("compute")]
[numthreads(128, 1, 1)]
void main(uint3 group: SV_GroupID, uint lane: SV_WaveLaneIndex, uint subgroup: SV_WaveIndex) {
    // Routing requires fixed 32-lane subgroups, W4 groups32/64, and dense A.
    uint columnTile = group.x;
    uint rowTile = group.y;
    // Zero is a specialization for the common single-page grid, so normal
    // model shapes retain their original workgroup indexing and arithmetic.
    if (param_prefill_column_pages() != 0u) {
        columnTile += (group.z % param_prefill_column_pages()) * 65535u;
        rowTile += (group.z / param_prefill_column_pages()) * 65535u;
        // A final page may contain unused tiles. Test before converting tile
        // coordinates to element coordinates, which could otherwise overflow.
        if (columnTile > (param_n() - 1u) / 16u || rowTile > (param_m() - 1u) / 4u) return;
    }
    uint firstRow = rowTile * 4u;
    uint firstColumn = columnTile * 16u + subgroup * 4u;
    uint4 columns;
    [ForceUnroll]
    for (uint column = 0; column < 4; column++)
        columns[column] = min(firstColumn + column, param_n() - 1u);
    uint groupLanes = param_w4_group_size() / 16u;
    uint groupCount = param_k() / param_w4_group_size();
    uint slice = (lane % groupLanes) * 16u;
    float4 result[4];
    [ForceUnroll]
    for (uint row = 0; row < 4; row++) result[row] = 0.0;

    for (uint quantGroup = lane / groupLanes; quantGroup < groupCount; quantGroup += 32u / groupLanes) {
        uint inner = quantGroup * param_w4_group_size() + slice;
        float4 scales, points, biases;
        [ForceUnroll]
        for (uint column = 0; column < 4; column++) {
            float3 quant = metadata(columns[column], quantGroup);
            scales[column] = quant.x;
            points[column] = quant.y;
            biases[column] = quant.z;
        }
        float4 partial[4];
        [ForceUnroll]
        for (uint row = 0; row < 4; row++) partial[row] = 0.0;
        float4 inputSum = 0.0;
        [ForceUnroll]
        for (uint chunk = 0; chunk < 2; chunk++) {
            uint4 packed;
            [ForceUnroll]
            for (uint column = 0; column < 4; column++)
                packed[column] = word_codes((columns[column] * param_k() + inner) / 8u + chunk);
            if (param_signed_codes()) packed ^= 0x88888888u;
            float4 inputs[8];
            [ForceUnroll]
            for (uint row = 0; row < 4; row++) {
                uint inputRow = min(firstRow + row, param_m() - 1u);
                if (param_input_bf16_words()) {
                    [ForceUnroll]
                    for (uint pair = 0; pair < 4; pair++) {
                        uint bits = word_a((inputRow * param_k() + inner) / 2u + chunk * 4u + pair);
                        inputs[pair * 2u][row] = asfloat(bits << 16u);
                        inputs[pair * 2u + 1u][row] = asfloat(bits & 0xffff0000u);
                    }
                } else {
                    [ForceUnroll]
                    for (uint value = 0; value < 8; value++)
                        inputs[value][row] = load_a(inputRow * param_k() + inner + chunk * 8u + value);
                }
            }
            [ForceUnroll]
            for (uint value = 0; value < 8; value++) {
                float4 weights = asfloat(((packed >> (value * 4u)) & 15u) | 0x4b000000u) - points;
                [ForceUnroll]
                for (uint row = 0; row < 4; row++)
                    partial[row] = fma(inputs[value][row], weights, partial[row]);
                if (param_correction() == 1u) inputSum += inputs[value];
            }
        }
        [ForceUnroll]
        for (uint row = 0; row < 4; row++) {
            result[row] = fma(scales, partial[row], result[row]);
            if (param_correction() == 1u) result[row] = fma(biases, inputSum[row], result[row]);
        }
    }
    [ForceUnroll]
    for (uint row = 0; row < 4; row++) {
        float4 total = WaveActiveSum(result[row]);
        if (lane == 0u && firstRow + row < param_m()) {
            [ForceUnroll]
            for (uint column = 0; column < 4; column++) {
                if (firstColumn + column < param_n()) {
                    float value = total[column] * param_output_scale();
                    if (param_has_bias()) value += load_bias(firstColumn + column);
                    if (param_has_soft_cap()) value = param_soft_cap() * tanh(value / param_soft_cap());
                    store_d((firstRow + row) * param_row_stride() + firstColumn + column, value);
                }
            }
        }
    }
}
