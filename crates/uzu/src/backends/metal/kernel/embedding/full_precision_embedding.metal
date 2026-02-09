#include <metal_stdlib>
#include "../definitions.metal"

#define BLOCK_SIZE 256

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(FullPrecisionEmbeddingLookup) (
    const device uint64_t* token_ids, // [batch_size]
    const device T* weights,          // [vocab_size, model_dim]
    device T* output,                 // [batch_size, model_dim]
    constant uint32_t& batch_size,
    constant uint32_t& vocab_size,
    constant uint32_t& model_dim,
    constant float& input_scale,
    uint dim_idx AXIS(model_dim, BLOCK_SIZE),
    uint batch_idx AXIS(batch_size, 1)
) {
  const uint64_t token_id = token_ids[batch_idx];
  const uint output_idx = batch_idx * model_dim + dim_idx;

  if (token_id >= vocab_size) {
    output[output_idx] = T(0);
    return;
  }

  T value = weights[token_id * model_dim + dim_idx];
  output[output_idx] = value * T(input_scale);
}
