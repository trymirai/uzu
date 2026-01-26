#include <metal_stdlib>
#include "../definitions.metal"

#define BLOCK_SIZE 256

SPECIALIZE(T, float, half, bfloat) KERNEL(FullPrecisionEmbeddingLookup) (
    const device uint64_t* token_ids, // [batch_size]
    const device T* weights,          // [vocab_size, model_dim]
    device T* output,                 // [batch_size, model_dim]
    constant uint32_t& batch_size,
    constant uint32_t& vocab_size,
    constant uint32_t& model_dim,
    constant float& input_scale,
    uint batch_idx GROUPS((batch_size * model_dim + BLOCK_SIZE - 1).div_ceil(BLOCK_SIZE)),
    uint dim_idx THREADS(BLOCK_SIZE)
) {
  if (batch_idx >= batch_size) {
    return;
  }

  const uint64_t token_id = token_ids[batch_idx];
  const uint thread_position_in_grid = batch_idx * batch_size + dim_idx;
  if (token_id >= vocab_size) {
    output[thread_position_in_grid] = T(0);
    return;
  }

  T value = weights[token_id * model_dim + dim_idx];
  output[thread_position_in_grid] = value * T(input_scale);
}