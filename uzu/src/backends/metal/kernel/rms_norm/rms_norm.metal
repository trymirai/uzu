#include <metal_stdlib>
#include "../definitions.metal"

using namespace metal;

#define BLOCK_SIZE 1024
#define SIMD_SIZE 32
#define GRAIN_SIZE 4

// Main template kernel - single implementation for all types
template<typename InputT, typename ScaleT, typename OutputT, typename AccumT>
void rms_norm_core(
    const device InputT* input_data,
    const device ScaleT* scales_data,
    device OutputT* output_data,
    uint element_count,
    constant float& epsilon,
    constant float& scale_offset,
    threadgroup AccumT* shared_sum,
    uint thread_in_row,
    bool full_layer
) {
    AccumT partial_sum = static_cast<AccumT>(0.0f);

    // Compute thread local partial sum
    for (uint base_i = thread_in_row * GRAIN_SIZE; base_i < element_count; base_i += BLOCK_SIZE * GRAIN_SIZE) {
        AccumT vals[GRAIN_SIZE];
        
        for (uint j = 0; j < GRAIN_SIZE; ++j) {
            uint i = base_i + j;
            vals[j] = (i < element_count) ? static_cast<AccumT>(input_data[i]) : 0.0f;
        }
        
        for (uint j = 0; j < GRAIN_SIZE; ++j) {
            partial_sum += vals[j] * vals[j];
        }
    }

    // Compute total sum across threadgroup
    AccumT total_sum = threadgroup_cooperative_reduce_sum<BLOCK_SIZE>(partial_sum, shared_sum, thread_in_row);

    // Compute RMS norm factor
    AccumT mean_square = static_cast<AccumT>(total_sum) / static_cast<AccumT>(element_count);
    AccumT rms_norm = rsqrt(mean_square + static_cast<AccumT>(epsilon));

    // Apply normalization and scaling using the same vectorized pattern
    for (uint base_i = thread_in_row * GRAIN_SIZE; base_i < element_count; base_i += BLOCK_SIZE * GRAIN_SIZE) {
        AccumT vals[GRAIN_SIZE];
        AccumT scaled_vals[GRAIN_SIZE];
        
        // Load GRAIN_SIZE input elements
        for (uint j = 0; j < GRAIN_SIZE; ++j) {
            uint i = base_i + j;
            vals[j] = (i < element_count) ? static_cast<AccumT>(input_data[i]) : 0.0f;
        }
        
        // Process GRAIN_SIZE elements: normalize and scale
        for (uint j = 0; j < GRAIN_SIZE; ++j) {
            uint i = base_i + j;
            if (i >= element_count) { continue; }

            AccumT normalized_high = vals[j] * rms_norm;

            if (full_layer) {
                // Full-layer: keep everything in accumulation precision
                AccumT scale_value_high = static_cast<AccumT>(scales_data[i]) + static_cast<AccumT>(scale_offset);
                scaled_vals[j] = normalized_high * scale_value_high;
            } else {
                // Only-normalization: cast down for the scale multiply
                OutputT normalized_low = static_cast<OutputT>(normalized_high);
                OutputT scale_value_low = static_cast<OutputT>(static_cast<AccumT>(scales_data[i]) + static_cast<AccumT>(scale_offset));
                OutputT product_low = normalized_low * scale_value_low;
                scaled_vals[j] = static_cast<AccumT>(product_low);
            }
        }
        
        // Store GRAIN_SIZE output elements
        for (uint j = 0; j < GRAIN_SIZE; ++j) {
            uint i = base_i + j;
            if (i < element_count) {
                output_data[i] = static_cast<OutputT>(scaled_vals[j]);
            }
        }
    }
}

// X-macro table defining all supported type combinations
// Format: (InputType, ScaleType, OutputType, AccumType, full_layer_mode, suffix)
#define FOREACH_RMS_COMBO(_) \
    /* float combinations */ \
    _(float, float, float, float, false, f32_f32_f32_f32_norm) \
    _(float, float, float, float, true,  f32_f32_f32_f32_full) \
    _(float, float, half,  float, false, f32_f32_f16_f32_norm) \
    _(float, float, half,  float, true,  f32_f32_f16_f32_full) \
    _(float, float, bfloat, float, false, f32_f32_bf16_f32_norm) \
    _(float, float, bfloat, float, true,  f32_f32_bf16_f32_full) \
    _(float, half,  float, float, false, f32_f16_f32_f32_norm) \
    _(float, half,  float, float, true,  f32_f16_f32_f32_full) \
    _(float, half,  half,  float, false, f32_f16_f16_f32_norm) \
    _(float, half,  half,  float, true,  f32_f16_f16_f32_full) \
    _(float, half,  bfloat, float, false, f32_f16_bf16_f32_norm) \
    _(float, half,  bfloat, float, true,  f32_f16_bf16_f32_full) \
    _(float, bfloat, float, float, false, f32_bf16_f32_f32_norm) \
    _(float, bfloat, float, float, true,  f32_bf16_f32_f32_full) \
    _(float, bfloat, half,  float, false, f32_bf16_f16_f32_norm) \
    _(float, bfloat, half,  float, true,  f32_bf16_f16_f32_full) \
    _(float, bfloat, bfloat, float, false, f32_bf16_bf16_f32_norm) \
    _(float, bfloat, bfloat, float, true,  f32_bf16_bf16_f32_full) \
    \
    /* half combinations */ \
    _(half, float, float, float, false, f16_f32_f32_f32_norm) \
    _(half, float, float, float, true,  f16_f32_f32_f32_full) \
    _(half, float, half,  float, false, f16_f32_f16_f32_norm) \
    _(half, float, half,  float, true,  f16_f32_f16_f32_full) \
    _(half, float, bfloat, float, false, f16_f32_bf16_f32_norm) \
    _(half, float, bfloat, float, true,  f16_f32_bf16_f32_full) \
    _(half, half,  float, float, false, f16_f16_f32_f32_norm) \
    _(half, half,  float, float, true,  f16_f16_f32_f32_full) \
    _(half, half,  half,  float, false, f16_f16_f16_f32_norm) \
    _(half, half,  half,  float, true,  f16_f16_f16_f32_full) \
    _(half, half,  bfloat, float, false, f16_f16_bf16_f32_norm) \
    _(half, half,  bfloat, float, true,  f16_f16_bf16_f32_full) \
    _(half, bfloat, float, float, false, f16_bf16_f32_f32_norm) \
    _(half, bfloat, float, float, true,  f16_bf16_f32_f32_full) \
    _(half, bfloat, half,  float, false, f16_bf16_f16_f32_norm) \
    _(half, bfloat, half,  float, true,  f16_bf16_f16_f32_full) \
    _(half, bfloat, bfloat, float, false, f16_bf16_bf16_f32_norm) \
    _(half, bfloat, bfloat, float, true,  f16_bf16_bf16_f32_full) \
    \
    /* bfloat combinations */ \
    _(bfloat, float, float, float, false, bf16_f32_f32_f32_norm) \
    _(bfloat, float, float, float, true,  bf16_f32_f32_f32_full) \
    _(bfloat, float, half,  float, false, bf16_f32_f16_f32_norm) \
    _(bfloat, float, half,  float, true,  bf16_f32_f16_f32_full) \
    _(bfloat, float, bfloat, float, false, bf16_f32_bf16_f32_norm) \
    _(bfloat, float, bfloat, float, true,  bf16_f32_bf16_f32_full) \
    _(bfloat, half,  float, float, false, bf16_f16_f32_f32_norm) \
    _(bfloat, half,  float, float, true,  bf16_f16_f32_f32_full) \
    _(bfloat, half,  half,  float, false, bf16_f16_f16_f32_norm) \
    _(bfloat, half,  half,  float, true,  bf16_f16_f16_f32_full) \
    _(bfloat, half,  bfloat, float, false, bf16_f16_bf16_f32_norm) \
    _(bfloat, half,  bfloat, float, true,  bf16_f16_bf16_f32_full) \
    _(bfloat, bfloat, float, float, false, bf16_bf16_f32_f32_norm) \
    _(bfloat, bfloat, float, float, true,  bf16_bf16_f32_f32_full) \
    _(bfloat, bfloat, half,  float, false, bf16_bf16_f16_f32_norm) \
    _(bfloat, bfloat, half,  float, true,  bf16_bf16_f16_f32_full) \
    _(bfloat, bfloat, bfloat, float, false, bf16_bf16_bf16_f32_norm) \
    _(bfloat, bfloat, bfloat, float, true,  bf16_bf16_bf16_f32_full) \
    \
    /* All combinations with half accumulation */ \
    _(float, float, float, half, false, f32_f32_f32_f16_norm) \
    _(float, float, float, half, true,  f32_f32_f32_f16_full) \
    _(float, float, half,  half, false, f32_f32_f16_f16_norm) \
    _(float, float, half,  half, true,  f32_f32_f16_f16_full) \
    _(float, float, bfloat, half, false, f32_f32_bf16_f16_norm) \
    _(float, float, bfloat, half, true,  f32_f32_bf16_f16_full) \
    _(float, half,  float, half, false, f32_f16_f32_f16_norm) \
    _(float, half,  float, half, true,  f32_f16_f32_f16_full) \
    _(float, half,  half,  half, false, f32_f16_f16_f16_norm) \
    _(float, half,  half,  half, true,  f32_f16_f16_f16_full) \
    _(float, half,  bfloat, half, false, f32_f16_bf16_f16_norm) \
    _(float, half,  bfloat, half, true,  f32_f16_bf16_f16_full) \
    _(float, bfloat, float, half, false, f32_bf16_f32_f16_norm) \
    _(float, bfloat, float, half, true,  f32_bf16_f32_f16_full) \
    _(float, bfloat, half,  half, false, f32_bf16_f16_f16_norm) \
    _(float, bfloat, half,  half, true,  f32_bf16_f16_f16_full) \
    _(float, bfloat, bfloat, half, false, f32_bf16_bf16_f16_norm) \
    _(float, bfloat, bfloat, half, true,  f32_bf16_bf16_f16_full) \
    \
    _(half, float, float, half, false, f16_f32_f32_f16_norm) \
    _(half, float, float, half, true,  f16_f32_f32_f16_full) \
    _(half, float, half,  half, false, f16_f32_f16_f16_norm) \
    _(half, float, half,  half, true,  f16_f32_f16_f16_full) \
    _(half, float, bfloat, half, false, f16_f32_bf16_f16_norm) \
    _(half, float, bfloat, half, true,  f16_f32_bf16_f16_full) \
    _(half, half,  float, half, false, f16_f16_f32_f16_norm) \
    _(half, half,  float, half, true,  f16_f16_f32_f16_full) \
    _(half, half,  half,  half, false, f16_f16_f16_f16_norm) \
    _(half, half,  half,  half, true,  f16_f16_f16_f16_full) \
    _(half, half,  bfloat, half, false, f16_f16_bf16_f16_norm) \
    _(half, half,  bfloat, half, true,  f16_f16_bf16_f16_full) \
    _(half, bfloat, float, half, false, f16_bf16_f32_f16_norm) \
    _(half, bfloat, float, half, true,  f16_bf16_f32_f16_full) \
    _(half, bfloat, half,  half, false, f16_bf16_f16_f16_norm) \
    _(half, bfloat, half,  half, true,  f16_bf16_f16_f16_full) \
    _(half, bfloat, bfloat, half, false, f16_bf16_bf16_f16_norm) \
    _(half, bfloat, bfloat, half, true,  f16_bf16_bf16_f16_full) \
    \
    _(bfloat, float, float, half, false, bf16_f32_f32_f16_norm) \
    _(bfloat, float, float, half, true,  bf16_f32_f32_f16_full) \
    _(bfloat, float, half,  half, false, bf16_f32_f16_f16_norm) \
    _(bfloat, float, half,  half, true,  bf16_f32_f16_f16_full) \
    _(bfloat, float, bfloat, half, false, bf16_f32_bf16_f16_norm) \
    _(bfloat, float, bfloat, half, true,  bf16_f32_bf16_f16_full) \
    _(bfloat, half,  float, half, false, bf16_f16_f32_f16_norm) \
    _(bfloat, half,  float, half, true,  bf16_f16_f32_f16_full) \
    _(bfloat, half,  half,  half, false, bf16_f16_f16_f16_norm) \
    _(bfloat, half,  half,  half, true,  bf16_f16_f16_f16_full) \
    _(bfloat, half,  bfloat, half, false, bf16_f16_bf16_f16_norm) \
    _(bfloat, half,  bfloat, half, true,  bf16_f16_bf16_f16_full) \
    _(bfloat, bfloat, float, half, false, bf16_bf16_f32_f16_norm) \
    _(bfloat, bfloat, float, half, true,  bf16_bf16_f32_f16_full) \
    _(bfloat, bfloat, half,  half, false, bf16_bf16_f16_f16_norm) \
    _(bfloat, bfloat, half,  half, true,  bf16_bf16_f16_f16_full) \
    _(bfloat, bfloat, bfloat, half, false, bf16_bf16_bf16_f16_norm) \
    _(bfloat, bfloat, bfloat, half, true,  bf16_bf16_bf16_f16_full)

// Generate RMS norm kernels
#define DEFINE_RMS_KERNEL(IN, SC, OUT, ACC, FULL_LAYER, SUF) \
kernel void rms_norm_##SUF( \
    const device IN*    input         [[buffer(0)]], \
    const device SC*    scales        [[buffer(1)]], \
    device       OUT*   output        [[buffer(2)]], \
    constant uint&      batch_size    [[buffer(3)]], \
    constant uint&      model_dim     [[buffer(4)]], \
    constant float&     epsilon       [[buffer(5)]], \
    constant float&     scale_offset  [[buffer(6)]], \
    uint batch_idx [[threadgroup_position_in_grid]], \
    uint thread_in_row [[thread_position_in_threadgroup]] \
) { \
    if (batch_idx >= batch_size) return; \
    \
    threadgroup ACC shared_sum[SIMD_SIZE]; \
    const uint input_offset = batch_idx * model_dim; \
    \
    rms_norm_core<IN, SC, OUT, ACC>( \
        input + input_offset, \
        scales, \
        output + input_offset, \
        model_dim, \
        epsilon, \
        scale_offset, \
        shared_sum, \
        thread_in_row, \
        FULL_LAYER \
    ); \
}

FOREACH_RMS_COMBO(DEFINE_RMS_KERNEL)
#undef DEFINE_RMS_KERNEL

// Generate QK norm kernels
#define DEFINE_QK_KERNEL(IN, SC, OUT, ACC, FULL_LAYER, SUF) \
kernel void qk_norm_##SUF( \
    const device IN*    qkv_input     [[buffer(0)]], \
    const device SC*    scales        [[buffer(1)]], \
    device       OUT*   qkv_output    [[buffer(2)]], \
    constant uint&      batch_size    [[buffer(3)]], \
    constant uint&      num_q_heads   [[buffer(4)]], \
    constant uint&      num_kv_heads  [[buffer(5)]], \
    constant uint&      head_dim      [[buffer(6)]], \
    constant float&     epsilon       [[buffer(7)]], \
    constant float&     scale_offset  [[buffer(8)]], \
    constant uint&      active_type_mask [[buffer(9)]], \
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]], \
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]] \
) { \
    uint batch_idx = threadgroup_position_in_grid.x; \
    uint sequential_head_idx = threadgroup_position_in_grid.y; \
    uint thread_in_threadgroup = thread_position_in_threadgroup.x; \
    \
    if (batch_idx >= batch_size) return; \
    \
    uint qk_type, actual_head_idx, logical_head_idx; \
    if (sequential_head_idx < num_q_heads) { \
        qk_type = 0; \
        actual_head_idx = sequential_head_idx; \
        logical_head_idx = actual_head_idx; \
    } else { \
        qk_type = 1; \
        actual_head_idx = sequential_head_idx - num_q_heads; \
        logical_head_idx = num_q_heads + actual_head_idx; \
    } \
    \
    if ((active_type_mask & (1 << qk_type)) == 0) return; \
    if (qk_type == 0 && actual_head_idx >= num_q_heads) return; \
    if (qk_type == 1 && actual_head_idx >= num_kv_heads) return; \
    \
    const uint total_heads_in_buffer = num_q_heads + 2 * num_kv_heads; \
    const uint slice_offset = batch_idx * total_heads_in_buffer * head_dim + logical_head_idx * head_dim; \
    \
    threadgroup ACC shared_sum[SIMD_SIZE]; \
    \
    rms_norm_core<IN, SC, OUT, ACC>( \
        qkv_input + slice_offset, \
        scales, \
        qkv_output + slice_offset, \
        head_dim, \
        epsilon, \
        scale_offset, \
        shared_sum, \
        thread_in_threadgroup, \
        FULL_LAYER \
    ); \
}

FOREACH_RMS_COMBO(DEFINE_QK_KERNEL)
#undef DEFINE_QK_KERNEL 