#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../matmul/common/fragment.h"
#include "../matmul/common/mxu_fragment_ops.h"
#include "../matmul/common/simdgroup_fragment_ops.h"

using namespace metal;
using namespace uzu::matmul;

#define TREE_OUT_MATMUL_COLS 32u
#define TREE_OUT_MATMUL_SIMDGROUPS_PER_TG 4u

template <typename T, bool USE_H0>
METAL_FUNC void tree_out_scalar_position(
    const device T* q,
    const device float* prefix,
    const device float* qkd,
    const device T* u,
    const device T* h0,
    const device int* h0_indices,
    device T* o,
    const float scale,
    const uint tree_size,
    const uint qk_heads,
    const uint value_heads,
    const uint head_k_dim,
    const uint head_v_dim,
    const uint batch_idx,
    const uint value_head_idx,
    const uint row,
    const uint value_col
) {
  const uint value_heads_per_qk_head = value_heads / qk_heads;
  const uint qk_head_idx = value_head_idx / value_heads_per_qk_head;
  const uint q_row_base = (batch_idx * tree_size * qk_heads + row * qk_heads + qk_head_idx) * head_k_dim;
  const uint prefix_base = batch_idx * tree_size * value_heads + value_head_idx;
  const uint qkd_base = (batch_idx * value_heads + value_head_idx) * tree_size * tree_size;
  const uint u_base = (batch_idx * value_heads + value_head_idx) * tree_size * head_v_dim;
  const uint out_offset = ((batch_idx * tree_size + row) * value_heads + value_head_idx) * head_v_dim + value_col;

  float acc = 0.0f;

  const int h0_index = USE_H0 ? h0_indices[batch_idx] : -1;
  if (USE_H0 && h0_index >= 0) {
    const uint h0_base = ((uint(h0_index) * value_heads + value_head_idx) * head_v_dim + value_col) * head_k_dim;
    float dot = 0.0f;
    for (uint dim = 0; dim < head_k_dim; dim++) {
      dot += float(q[q_row_base + dim]) * float(h0[h0_base + dim]);
    }
    acc += exp(prefix[prefix_base + row * value_heads]) * scale * dot;
  }

  for (uint col = 0; col < tree_size; col++) {
    acc += qkd[qkd_base + row * tree_size + col] * float(u[u_base + col * head_v_dim + value_col]);
  }

  o[out_offset] = T(acc);
}

// Reference-shaped scalar Metal path, kept only so the CPU BuildTreeOut trait
// has a matching Metal implementation.
template <typename T, bool USE_H0>
VARIANTS(T, float, bfloat)
VARIANTS(USE_H0, false, true)
PUBLIC KERNEL(BuildTreeOut)(
    const device T* q,
    const device float* prefix,
    const device float* qkd,
    const device T* u,
    const device T* h0 OPTIONAL(USE_H0),
    const device int* h0_indices OPTIONAL(USE_H0),
    device T* o,
    constant const float& scale,
    constant const uint& batch_size,
    constant const uint& tree_size,
    constant const uint& qk_heads,
    constant const uint& value_heads,
    constant const uint& head_k_dim,
    constant const uint& head_v_dim,
    const uint position AXIS(batch_size * value_heads * tree_size * head_v_dim, 256)
) {
  const uint value_col = position % head_v_dim;
  const uint row = (position / head_v_dim) % tree_size;
  const uint value_head_idx = (position / (head_v_dim * tree_size)) % value_heads;
  const uint batch_idx = position / (head_v_dim * tree_size * value_heads);
  tree_out_scalar_position<T, USE_H0>(
      q,
      prefix,
      qkd,
      u,
      h0,
      h0_indices,
      o,
      scale,
      tree_size,
      qk_heads,
      value_heads,
      head_k_dim,
      head_v_dim,
      batch_idx,
      value_head_idx,
      row,
      value_col
  );
}

// One simdgroup owns a direct-load output fragment. USE_MXU selects MPP MXU
// fragments; false uses the portable simdgroup_matrix fragment path.
template <typename T, bool USE_H0, bool USE_MXU>
VARIANTS(T, bfloat)
VARIANTS(USE_H0, false, true)
VARIANTS(USE_MXU, false, true)
PUBLIC KERNEL(BuildTreeOutOutputTileMatmulDirect)(
    const device T* q,
    const device float* prefix,
    const device float* qkd,
    const device T* u,
    const device T* h0 OPTIONAL(USE_H0),
    const device int* h0_indices OPTIONAL(USE_H0),
    device T* o,
    constant const float& scale,
    constant const uint& batch_size,
    constant const uint& tree_size,
    constant const uint& qk_heads,
    constant const uint& value_heads,
    constant const uint& head_k_dim,
    constant const uint& head_v_dim,
    const ThreadContext thread_context,
    const uint v_tile_idx GROUPS(head_v_dim.div_ceil(TREE_OUT_MATMUL_COLS)),
    const uint row_tile_group_idx GROUPS(tree_size.div_ceil(
        if USE_MXU { TREE_OUT_MATMUL_SIMDGROUPS_PER_TG * 16 } else { TREE_OUT_MATMUL_SIMDGROUPS_PER_TG * 8 }
    )),
    const uint batch_value_head_idx GROUPS(batch_size * value_heads),
    const uint tid THREADS(TREE_OUT_MATMUL_SIMDGROUPS_PER_TG * METAL_SIMD_SIZE)
) {
  (void)tid;
  using Ops = metal::conditional_t<USE_MXU, MxuFragmentOps<>, SimdgroupFragmentOps>;
  using InputType = metal::conditional_t<USE_MXU, T, float>;
  constexpr ushort ROWS = Ops::FRAGMENT_ROWS;
  constexpr ushort COL_FRAGMENTS = TREE_OUT_MATMUL_COLS / Ops::FRAGMENT_COLS;
  using AccFragment = Fragment<float, 1, COL_FRAGMENTS, Ops>;
  using QFragment = OperandFragment<InputType, 1, 1, Ops>;
  using H0Fragment = OperandFragment<InputType, 1, COL_FRAGMENTS, Ops>;
  using QkdFragment = OperandFragment<float, 1, 1, Ops>;
  using UFragment = OperandFragment<InputType, 1, COL_FRAGMENTS, Ops>;

  const uint batch_idx = batch_value_head_idx / value_heads;
  const uint value_head_idx = batch_value_head_idx - batch_idx * value_heads;
  const uint value_heads_per_qk_head = value_heads / qk_heads;
  const uint qk_head_idx = value_head_idx / value_heads_per_qk_head;
  const uint row_base =
      (row_tile_group_idx * TREE_OUT_MATMUL_SIMDGROUPS_PER_TG + thread_context.simdgroup_index) * ROWS;
  const uint value_base = v_tile_idx * TREE_OUT_MATMUL_COLS;

  const short valid_rows = short(min(uint(ROWS), tree_size - min(row_base, tree_size)));
  const short valid_cols = short(min(TREE_OUT_MATMUL_COLS, head_v_dim - min(value_base, head_v_dim)));

  AccFragment acc;
  acc.clear();

  const int h0_index = USE_H0 ? h0_indices[batch_idx] : -1;
  if (USE_H0 && h0_index >= 0) {
    const uint q_base = (batch_idx * tree_size * qk_heads + row_base * qk_heads + qk_head_idx) * head_k_dim;
    const uint h0_base = ((uint(h0_index) * value_heads + value_head_idx) * head_v_dim + value_base) * head_k_dim;
    for (uint k0 = 0; k0 < head_k_dim; k0 += ROWS) {
      const short valid_k = short(min(uint(ROWS), head_k_dim - k0));
      QFragment q_frag;
      H0Fragment h0_frag;
      q_frag.load_from(
          thread_context.simd_lane_id,
          fragment_source(q + q_base + k0, int(qk_heads * head_k_dim)).bounded(valid_rows, valid_k)
      );
      h0_frag.load_from(
          thread_context.simd_lane_id,
          fragment_source(h0 + h0_base + k0, 1, int(head_k_dim)).bounded(valid_k, valid_cols)
      );
      fragment_mma(acc, q_frag, h0_frag);
    }

    acc.map_coords(thread_context.simd_lane_id, [&](short row, short, float value) {
      const uint global_row = row_base + uint(row);
      const float row_scale =
          global_row < tree_size
              ? exp(prefix[(batch_idx * tree_size + global_row) * value_heads + value_head_idx]) * scale
              : 0.0f;
      return value * row_scale;
    });
  }

  const uint qkd_base = (batch_idx * value_heads + value_head_idx) * tree_size * tree_size + row_base * tree_size;
  const uint u_base = (batch_idx * value_heads + value_head_idx) * tree_size * head_v_dim + value_base;
  for (uint j0 = 0; j0 < tree_size; j0 += ROWS) {
    const short valid_j = short(min(uint(ROWS), tree_size - j0));
    QkdFragment qkd_frag;
    UFragment u_frag;
    qkd_frag.load_from(
        thread_context.simd_lane_id,
        fragment_source(qkd + qkd_base + j0, int(tree_size)).bounded(valid_rows, valid_j)
    );
    u_frag.load_from(
        thread_context.simd_lane_id,
        fragment_source(u + u_base + j0 * head_v_dim, int(head_v_dim)).bounded(valid_j, valid_cols)
    );
    fragment_mma(acc, qkd_frag, u_frag);
  }

  device T* out_tile =
      o + ((batch_idx * tree_size + row_base) * value_heads + value_head_idx) * head_v_dim + value_base;
  acc.store_safe(thread_context.simd_lane_id, out_tile, int(value_heads * head_v_dim), short2(valid_cols, valid_rows));
}
