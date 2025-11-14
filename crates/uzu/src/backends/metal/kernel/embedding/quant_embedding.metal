#include <metal_stdlib>
using namespace metal;

// Quantized embedding lookup kernel for MLX-style 4-bit quantization
// Weights are packed as U8 (2 values per byte) with shape [vocab_size, model_dim/2]
// Scales are [vocab_size, num_groups] where each scale applies to group_size values

template<typename T>
[[kernel]] void quantized_embedding_lookup(
    const device uint64_t* token_ids [[buffer(0)]],          // [batch_size]
    const device uint8_t* weights [[buffer(1)]],             // [vocab_size, model_dim/2] packed
    const device T* scales [[buffer(2)]],                    // [vocab_size, num_groups]
    const device T* biases [[buffer(3)]],                    // [vocab_size, num_groups]
    device T* output [[buffer(4)]],                          // [batch_size, model_dim]
    constant uint32_t& batch_size [[buffer(5)]],
    constant uint32_t& vocab_size [[buffer(6)]],
    constant uint32_t& model_dim [[buffer(7)]],
    constant uint32_t& group_size [[buffer(8)]],
    uint thread_position_in_grid [[thread_position_in_grid]]
) {
    // Each thread handles one output value
    const uint batch_idx = thread_position_in_grid / model_dim;
    const uint dim_idx = thread_position_in_grid % model_dim;

    if (batch_idx >= batch_size) return;

    // Get the token ID for this batch element
    const uint64_t token_id = token_ids[batch_idx];
    if (token_id >= vocab_size) {
        output[thread_position_in_grid] = T(0);
        return;
    }

    // Calculate which group this dimension belongs to
    const uint group_idx = dim_idx / group_size;
    const uint num_groups = (model_dim + group_size - 1) / group_size;

    // Get the scale for this token and group
    const T scale = scales[token_id * num_groups + group_idx];
    const T bias = biases[token_id * num_groups + group_idx];

    // Get the packed weight value
    // Each byte contains two 4-bit values: low nibble and high nibble
    const uint byte_idx = token_id * (model_dim / 2) + (dim_idx / 2);
    const uint8_t packed = weights[byte_idx];

    // Extract the 4-bit value (0-15)
    uint8_t quantized;
    if (dim_idx % 2 == 0) {
        // Low nibble
        quantized = packed & 0x0F;
    } else {
        // High nibble
        quantized = (packed >> 4) & 0x0F;
    }

    // Dequantize: value = scale * quantized + bias
    output[thread_position_in_grid] = scale * T(quantized) + bias;
}

// Explicit instantiations for different data types
template [[host_name("quantized_embedding_lookup_f32")]]
[[kernel]] void quantized_embedding_lookup<float>(
    const device uint64_t* token_ids [[buffer(0)]],
    const device uint8_t* weights [[buffer(1)]],
    const device float* scales [[buffer(2)]],
    const device float* biases [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint32_t& batch_size [[buffer(5)]],
    constant uint32_t& vocab_size [[buffer(6)]],
    constant uint32_t& model_dim [[buffer(7)]],
    constant uint32_t& group_size [[buffer(8)]],
    uint thread_position_in_grid [[thread_position_in_grid]]
);

template [[host_name("quantized_embedding_lookup_f16")]]
[[kernel]] void quantized_embedding_lookup<half>(
    const device uint64_t* token_ids [[buffer(0)]],
    const device uint8_t* weights [[buffer(1)]],
    const device half* scales [[buffer(2)]],
    const device half* biases [[buffer(3)]],
    device half* output [[buffer(4)]],
    constant uint32_t& batch_size [[buffer(5)]],
    constant uint32_t& vocab_size [[buffer(6)]],
    constant uint32_t& model_dim [[buffer(7)]],
    constant uint32_t& group_size [[buffer(8)]],
    uint thread_position_in_grid [[thread_position_in_grid]]
);

template [[host_name("quantized_embedding_lookup_bf16")]]
[[kernel]] void quantized_embedding_lookup<bfloat>(
    const device uint64_t* token_ids [[buffer(0)]],
    const device uint8_t* weights [[buffer(1)]],
    const device bfloat* scales [[buffer(2)]],
    const device bfloat* biases [[buffer(3)]],
    device bfloat* output [[buffer(4)]],
    constant uint32_t& batch_size [[buffer(5)]],
    constant uint32_t& vocab_size [[buffer(6)]],
    constant uint32_t& model_dim [[buffer(7)]],
    constant uint32_t& group_size [[buffer(8)]],
    uint thread_position_in_grid [[thread_position_in_grid]]
);
