#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"

using namespace metal;

// In-place fused element-wise multiply by factors + Walsh-Hadamard Transform.
// Hardcoded to METAL_SIMD_SIZE, matching Apple Silicon SIMD width.
// Each SIMD group processes one block entirely in registers
// using simd_shuffle_xor for the butterfly stages.
template <typename T>
VARIANTS(T, half, bfloat, float)
PUBLIC KERNEL(HadamardTransformMul)(
    device T* data,
    const device T* factors,
    constant uint& total_blocks,
    constant uint& channel_count,
    uint block_index GROUPS(total_blocks),
    uint lane_index THREADS(METAL_SIMD_SIZE)
) {
    uint batch_index = block_index / (channel_count / METAL_SIMD_SIZE);
    uint block_within_batch = block_index % (channel_count / METAL_SIMD_SIZE);
    uint element_index = batch_index * channel_count + block_within_batch * METAL_SIMD_SIZE + lane_index;
    uint factor_index = block_within_batch * METAL_SIMD_SIZE + lane_index;

    float value = float(data[element_index]) * float(factors[factor_index]);

    for (uint stride = 1; stride < METAL_SIMD_SIZE; stride <<= 1) {
        float partner_value = simd_shuffle_xor(value, static_cast<ushort>(stride));
        value = (lane_index & stride) ? (partner_value - value) : (partner_value + value);
    }

    // Orthogonal normalization: matches Python hadamard_transform / sqrt(block_size)
    constexpr float normalization_factor = 1.0f / 5.656854249f; // 1/sqrt(METAL_SIMD_SIZE)
    data[element_index] = T(value * normalization_factor);
}
