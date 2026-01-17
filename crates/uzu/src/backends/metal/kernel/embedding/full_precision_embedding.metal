#include <metal_stdlib>
using namespace metal;

template <typename T>
[[kernel, max_total_threads_per_threadgroup(256)]]
void full_precision_embedding_lookup(
    const device uint64_t* token_ids [[buffer(0)]], // [batch_size]
    const device T* weights [[buffer(1)]],          // [vocab_size, model_dim]
    device T* output [[buffer(2)]],                 // [batch_size, model_dim]
    constant uint32_t& batch_size [[buffer(3)]],
    constant uint32_t& vocab_size [[buffer(4)]],
    constant uint32_t& model_dim [[buffer(5)]],
    constant float& input_scale [[buffer(6)]],
    uint thread_position_in_grid [[thread_position_in_grid]]
) {
  const uint batch_idx = thread_position_in_grid / model_dim;
  const uint dim_idx = thread_position_in_grid % model_dim;

  if (batch_idx >= batch_size)
    return;

  const uint64_t token_id = token_ids[batch_idx];
  if (token_id >= vocab_size) {
    output[thread_position_in_grid] = T(0);
    return;
  }

  T value = weights[token_id * model_dim + dim_idx];
  output[thread_position_in_grid] = value * T(input_scale);
}

template [[host_name("full_precision_embedding_lookup_f32")]] [[kernel]] void
full_precision_embedding_lookup<float>(
    const device uint64_t* token_ids [[buffer(0)]],
    const device float* weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint32_t& batch_size [[buffer(3)]],
    constant uint32_t& vocab_size [[buffer(4)]],
    constant uint32_t& model_dim [[buffer(5)]],
    constant float& input_scale [[buffer(6)]],
    uint thread_position_in_grid [[thread_position_in_grid]]
);

template [[host_name("full_precision_embedding_lookup_f16")]] [[kernel]] void
full_precision_embedding_lookup<half>(
    const device uint64_t* token_ids [[buffer(0)]],
    const device half* weights [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint32_t& batch_size [[buffer(3)]],
    constant uint32_t& vocab_size [[buffer(4)]],
    constant uint32_t& model_dim [[buffer(5)]],
    constant float& input_scale [[buffer(6)]],
    uint thread_position_in_grid [[thread_position_in_grid]]
);

template [[host_name("full_precision_embedding_lookup_bf16")]] [[kernel]] void
full_precision_embedding_lookup<bfloat>(
    const device uint64_t* token_ids [[buffer(0)]],
    const device bfloat* weights [[buffer(1)]],
    device bfloat* output [[buffer(2)]],
    constant uint32_t& batch_size [[buffer(3)]],
    constant uint32_t& vocab_size [[buffer(4)]],
    constant uint32_t& model_dim [[buffer(5)]],
    constant float& input_scale [[buffer(6)]],
    uint thread_position_in_grid [[thread_position_in_grid]]
);
