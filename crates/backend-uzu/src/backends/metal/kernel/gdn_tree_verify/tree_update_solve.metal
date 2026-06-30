#include <metal_stdlib>
#include "../common/dsl.h"

using namespace metal;

constant float L2_NORM_EPSILON = 1e-6f;

// Solves chunked GDN update coefficients:
//   acc = beta * (v - exp(prefix) * k @ h0)
//   acc -= A[cur_block, prev_rows] @ U[prev_rows]
//   U[cur_block] = Ainv[cur_block] @ acc
//
// Shapes:
//   k      [B, T, num_k_heads, HEAD_K_DIM]
//   v      [B, T, num_v_heads, head_v_dim]
//   prefix [B, T, num_v_heads]
//   beta   [B, T, num_v_heads]
//   A      [B * num_v_heads, T, T]
//   Ainv   [B * num_v_heads, T, T]
//   h0     [pool, num_v_heads, head_v_dim, HEAD_K_DIM]
//   h0_idx [B], negative means zero initial state
//   U      [B * num_v_heads, T, head_v_dim]
//
// Grid: one threadgroup per (batch, value head, BV value-dim tile). Blocks are
// solved serially inside the group because each block depends on earlier U rows.
template <typename T, uint HEAD_K_DIM, uint BT, uint BV>
VARIANTS(T, float, bfloat)
VARIANTS(HEAD_K_DIM, 128)
VARIANTS(BT, 16)
VARIANTS(BV, 16)
PUBLIC KERNEL(GdnTreeUpdateSolve)(
    const device T* k,
    const device T* v,
    const device float* prefix,
    const device float* beta,
    const device float* a,
    const device float* a_inv,
    const device T* h0,
    const device int* h0_idx,
    device float* u,
    constant const uint& batch_size,
    constant const uint& tree_size,
    constant const uint& num_v_heads,
    constant const uint& num_k_heads,
    constant const uint& head_v_dim,
    threadgroup float acc_tile[BT * BV],
    const bool use_l2norm SPECIALIZE,
    const uint batch_idx GROUPS(batch_size),
    const uint hv GROUPS(num_v_heads),
    const uint value_tile_idx GROUPS(head_v_dim.div_ceil(BV)),
    const uint thread_idx THREADS(256)
) {
  const uint batch_value_head_idx = batch_idx * num_v_heads + hv;

  // Several value heads can share one key head.
  const uint groups_per_head = num_v_heads / num_k_heads;
  const uint hk = hv / groups_per_head;

  const uint value_dim_base = value_tile_idx * BV;
  const uint num_blocks = (tree_size + BT - 1u) / BT;
  const int h0_slot = h0_idx[batch_idx];

  const uint k_token_stride = num_k_heads * HEAD_K_DIM;
  const uint v_token_stride = num_v_heads * head_v_dim;
  const uint scalar_token_stride = num_v_heads;

  const device T* k_head = k + (batch_idx * tree_size * num_k_heads + hk) * HEAD_K_DIM;
  const device T* v_head_tile = v + (batch_idx * tree_size * num_v_heads + hv) * head_v_dim + value_dim_base;
  const device float* prefix_head = prefix + batch_idx * tree_size * num_v_heads + hv;
  const device float* beta_head = beta + batch_idx * tree_size * num_v_heads + hv;
  const device float* a_matrix = a + batch_value_head_idx * tree_size * tree_size;
  const device float* a_inv_matrix = a_inv + batch_value_head_idx * tree_size * tree_size;
  device float* u_head_tile = u + batch_value_head_idx * tree_size * head_v_dim + value_dim_base;
  const device T* h0_head_tile =
      h0_slot >= 0 ? h0 + (uint(h0_slot) * num_v_heads + hv) * head_v_dim * HEAD_K_DIM + value_dim_base * HEAD_K_DIM
                   : nullptr;

  const uint local_token_idx = thread_idx / BV;
  const uint local_value_dim_idx = thread_idx - local_token_idx * BV;
  const bool thread_has_tile_element = local_token_idx < BT && local_value_dim_idx < BV;

  for (uint block_idx = 0; block_idx < num_blocks; ++block_idx) {
    const uint token_idx = block_idx * BT + local_token_idx;
    const uint value_dim_idx = value_dim_base + local_value_dim_idx;
    const bool valid = thread_has_tile_element && token_idx < tree_size && value_dim_idx < head_v_dim;

    float acc = 0.0f;
    if (valid) {
      float k_scale = 1.0f;
      if (use_l2norm) {
        float k_norm_sq = 0.0f;
        for (uint key_dim_idx = 0; key_dim_idx < HEAD_K_DIM; ++key_dim_idx) {
          const float k_val = float(k_head[token_idx * k_token_stride + key_dim_idx]);
          k_norm_sq += k_val * k_val;
        }
        k_scale = rsqrt(k_norm_sq + L2_NORM_EPSILON);
      }

      float kh0 = 0.0f;
      if (h0_slot >= 0) {
        for (uint key_dim_idx = 0; key_dim_idx < HEAD_K_DIM; ++key_dim_idx) {
          const float k_val = float(k_head[token_idx * k_token_stride + key_dim_idx]) * k_scale;
          const float h0_val = float(h0_head_tile[local_value_dim_idx * HEAD_K_DIM + key_dim_idx]);
          kh0 += k_val * h0_val;
        }
      }

      const float v_val = float(v_head_tile[token_idx * v_token_stride + local_value_dim_idx]);
      const float prefix_val = prefix_head[token_idx * scalar_token_stride];
      const float beta_val = beta_head[token_idx * scalar_token_stride];
      acc = beta_val * (v_val - exp(prefix_val) * kh0);

      const uint current_token_idx = token_idx;
      const uint previous_token_count = block_idx * BT;
      for (uint source_token_idx = 0; source_token_idx < previous_token_count; ++source_token_idx) {
        const float a_val = a_matrix[current_token_idx * tree_size + source_token_idx];
        const float u_val = u_head_tile[source_token_idx * head_v_dim + local_value_dim_idx];
        acc -= a_val * u_val;
      }
    }

    if (thread_has_tile_element) {
      acc_tile[local_token_idx * BV + local_value_dim_idx] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (valid) {
      float solved = 0.0f;
      const uint current_token_idx = token_idx;
      for (uint source_local_token_idx = 0; source_local_token_idx < BT; ++source_local_token_idx) {
        const uint source_token_idx = block_idx * BT + source_local_token_idx;
        if (source_token_idx < tree_size) {
          const float a_inv_val = a_inv_matrix[current_token_idx * tree_size + source_token_idx];
          solved += a_inv_val * acc_tile[source_local_token_idx * BV + local_value_dim_idx];
        }
      }
      u_head_tile[token_idx * head_v_dim + local_value_dim_idx] = solved;
    }

    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  }
}
