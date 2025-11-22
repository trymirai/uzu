#include <metal_stdlib>
using namespace metal;

template<typename T, uint PACKING_DIVISOR, bool SIGNED_STORAGE>
[[kernel]] void quantized_embedding_lookup(
    const device uint64_t* token_ids [[buffer(0)]],          // [batch_size]
    const device uint8_t* weights [[buffer(1)]],             // [vocab_size, model_dim/packing_divisor] packed
    const device T* scales [[buffer(2)]],                    // [vocab_size, num_groups]
    const device T* biases [[buffer(3)]],                    // [vocab_size, num_groups]
    device T* output [[buffer(4)]],                          // [batch_size, model_dim]
    constant uint32_t& batch_size [[buffer(5)]],
    constant uint32_t& vocab_size [[buffer(6)]],
    constant uint32_t& model_dim [[buffer(7)]],
    constant uint32_t& group_size [[buffer(8)]],
    uint thread_position_in_grid [[thread_position_in_grid]]
) {
    const uint batch_idx = thread_position_in_grid / model_dim;
    const uint dim_idx = thread_position_in_grid % model_dim;

    if (batch_idx >= batch_size) return;

    const uint64_t token_id = token_ids[batch_idx];
    if (token_id >= vocab_size) {
        output[thread_position_in_grid] = T(0);
        return;
    }

    const uint group_idx = dim_idx / group_size;
    const uint num_groups = (model_dim + group_size - 1) / group_size;

    const T scale = scales[token_id * num_groups + group_idx];
    const T bias = biases[token_id * num_groups + group_idx];

    const uint weights_stride = model_dim / PACKING_DIVISOR;
    const device uint8_t* weights_u8 = weights;
    const device int8_t* weights_i8 =
        reinterpret_cast<const device int8_t*>(weights);

    int quantized_value = 0;
    if (PACKING_DIVISOR == 2) {
        const uint byte_idx =
            token_id * weights_stride + (dim_idx / 2);
        const uint8_t packed = weights_u8[byte_idx];
        if ((dim_idx & 1) == 0) {
            quantized_value = int(packed & 0x0F);
        } else {
            quantized_value = int((packed >> 4) & 0x0F);
        }
    } else {
        const uint elem_idx = token_id * weights_stride + dim_idx;
        quantized_value =
            SIGNED_STORAGE ? int(weights_i8[elem_idx])
                           : int(weights_u8[elem_idx]);
    }

    output[thread_position_in_grid] = scale * T(quantized_value) + bias;
}

template [[host_name("quantized_embedding_lookup_f32_uint4")]]
[[kernel]] void quantized_embedding_lookup<float, 2, false>(
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

template [[host_name("quantized_embedding_lookup_f16_uint4")]]
[[kernel]] void quantized_embedding_lookup<half, 2, false>(
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

template [[host_name("quantized_embedding_lookup_bf16_uint4")]]
[[kernel]] void quantized_embedding_lookup<bfloat, 2, false>(
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

template [[host_name("quantized_embedding_lookup_f32_int8")]]
[[kernel]] void quantized_embedding_lookup<float, 1, true>(
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

template [[host_name("quantized_embedding_lookup_f16_int8")]]
[[kernel]] void quantized_embedding_lookup<half, 1, true>(
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

template [[host_name("quantized_embedding_lookup_bf16_int8")]]
[[kernel]] void quantized_embedding_lookup<bfloat, 1, true>(
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

template [[host_name("quantized_embedding_lookup_f32_uint8")]]
[[kernel]] void quantized_embedding_lookup<float, 1, false>(
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

template [[host_name("quantized_embedding_lookup_f16_uint8")]]
[[kernel]] void quantized_embedding_lookup<half, 1, false>(
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

template [[host_name("quantized_embedding_lookup_bf16_uint8")]]
[[kernel]] void quantized_embedding_lookup<bfloat, 1, false>(
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
