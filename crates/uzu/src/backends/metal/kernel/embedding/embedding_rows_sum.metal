#include <metal_stdlib>
#include "../definitions.metal"

#define BLOCK_SIZE 256

template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(EmbeddingRowsSum)(
    const device uint64_t* row_indices, // [num_rows]
    const device T* weights,            // [total_rows, model_dim]
    device T* output,                   // [model_dim]
    constant uint32_t& num_rows,
    constant uint32_t& total_rows,
    constant uint32_t& model_dim,
    uint dim_idx AXIS(model_dim, BLOCK_SIZE)
) {
  float sum = 0.0f;
  for (uint row_idx = 0; row_idx < num_rows; row_idx++) {
    const uint64_t row = row_indices[row_idx];
    if (row < total_rows) {
      sum += float(weights[row * model_dim + dim_idx]);
    }
  }
  output[dim_idx] = T(sum);
}
