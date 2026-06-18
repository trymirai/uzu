#include <metal_stdlib>
#include "../activation/activations.h"
#include "../common/defines.h"
#include "../common/dsl.h"

#define ROWS_PER_THREADGROUP 4

using namespace metal;

// path_matrix: [B, T, T] uint8, inclusive ancestor matrix
// a:           [B, HV, T] transposed
// b:           [B, T, HV]
// prefix,beta: [B, T, HV] fp32
template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(BuildPrefixBeta)(
    const device uint8_t* path_matrix,
    const device T* a,
    const device T* b,
    const device float* a_log,
    const device float* dt_bias,
    device float* prefix,
    device float* beta,
    constant const uint& batch_size,
    constant const uint& tree_size,
    constant const uint& value_heads,
    const uint batch_idx GROUPS(batch_size),
    const uint row_idx GROUPS(tree_size),
    const uint head_group_idx GROUPS(value_heads.div_ceil(ROWS_PER_THREADGROUP)),
    const uint tid THREADS(ROWS_PER_THREADGROUP * METAL_SIMD_SIZE)
) {
  const uint simd_group = tid / METAL_SIMD_SIZE;
  const uint lane_id = tid - simd_group * METAL_SIMD_SIZE;
  const uint head_idx = head_group_idx * ROWS_PER_THREADGROUP + simd_group;
  if (head_idx >= value_heads) {
    return;
  }

  const uint batch_offset = batch_idx * tree_size * value_heads;
  const uint path_row_offset = batch_idx * tree_size * tree_size + row_idx * tree_size;
  const uint out_idx = batch_offset + row_idx * value_heads + head_idx;

  const float scale = fast::exp(float(a_log[head_idx]));
  const float dt = float(dt_bias[head_idx]);
  float partial = 0.0f;
  for (uint col_idx = lane_id; col_idx < tree_size; col_idx += METAL_SIMD_SIZE) {
    if (path_matrix[path_row_offset + col_idx] != 0) {
      const uint a_idx = batch_offset + head_idx * tree_size + col_idx;
      partial -= scale * activate_softplus(float(a[a_idx]) + dt);
    }
  }

  const float sum = simd_sum(partial);
  if (lane_id == 0) {
    const float b_val = float(b[out_idx]);
    beta[out_idx] = 1.0f / (1.0f + fast::exp(-b_val));
    prefix[out_idx] = sum;
  }
}
