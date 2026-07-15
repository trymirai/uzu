#include <metal_stdlib>
#include "../../common/defines.h"
#include "../../common/dsl.h"
#include "../../common/thread_context.h"
#include "../../matmul/common/fragment.h"
#include "../../matmul/common/mxu_fragment_ops.h"
#include "../../matmul/common/simdgroup_fragment_ops.h"

using namespace metal;
using namespace uzu::matmul;

#define MATMUL_MXU_ROWS 16u
#define MATMUL_SIMDGROUP_ROWS 8u
#define MATMUL_COLS 32u
#define SIMDGROUPS_PER_TG 4u

// Build tree output for one (batch, value-head, row, value) tile.
//
// q:      [B, T, Hg, K], with query head hq = hv / (HV / Hg)
// prefix: [B, T, HV] path log-decay prefix from BuildTreePrefix
// qkd:    [B, HV, T, T] from BuildTreeGram
// u:      [B, HV, T, V]
// h0:     [H0, HV, V, K]
//
// o[row, value] = exp(prefix[row]) * scale * dot(q[row], h0[value])
//                 + sum_j qkd[row, j] * u[j, value]
template <typename QKT, typename OutputT, bool use_mxu, bool transposed_h0>
VARIANTS(QKT, float, bfloat)
VARIANTS(OutputT, float, bfloat)
VARIANTS(use_mxu, false, true)
VARIANTS(transposed_h0, false, true)
CONSTRAINT(!(use_mxu && transposed_h0))
PUBLIC KERNEL(BuildTreeOut)(
    const device QKT* q,
    const device float* prefix,
    const device float* qkd,
    const device float* u,
    const device float* h0 OPTIONAL(use_h0),
    const device int* h0_indices OPTIONAL(use_h0),
    device OutputT* o,
    constant const float& scale,
    constant const uint& batch_size,
    constant const uint& tree_size,
    constant const uint& qk_heads,
    constant const uint& value_heads,
    constant const uint& head_k_dim,
    constant const uint& head_v_dim,
    threadgroup uint u_tile_scratch OPTIONAL(transposed_h0)[MATMUL_SIMDGROUP_ROWS * MATMUL_COLS],
    const bool use_h0 SPECIALIZE,
    const ThreadContext thread_context,
    const uint v_tile_idx GROUPS(head_v_dim.div_ceil(MATMUL_COLS)),
    const uint row_tile_group_idx GROUPS(tree_size.div_ceil(
        if use_mxu {
          SIMDGROUPS_PER_TG * MATMUL_MXU_ROWS
        } else {
          SIMDGROUPS_PER_TG * MATMUL_SIMDGROUP_ROWS
        }
    )),
    const uint batch_value_head_idx GROUPS(batch_size * value_heads),
    const uint tid THREADS(SIMDGROUPS_PER_TG * METAL_SIMD_SIZE)
) {
  using Ops = metal::conditional_t<use_mxu, MxuFragmentOps<>, SimdgroupFragmentOps>;
  using InputType = metal::conditional_t<use_mxu, QKT, float>;
  using H0Read = metal::conditional_t<transposed_h0, ReadTranspose, ReadDirect>;
  constexpr ushort ROWS = Ops::FRAGMENT_ROWS;
  constexpr ushort COL_FRAGMENTS = MATMUL_COLS / Ops::FRAGMENT_COLS;
  using AccFragment = Fragment<float, 1, COL_FRAGMENTS, Ops>;
  using QFragment = OperandFragment<InputType, 1, 1, Ops>;
  using H0Fragment = OperandFragment<float, 1, COL_FRAGMENTS, Ops, H0Read>;
  using QkdFragment = OperandFragment<float, 1, 1, Ops>;
  using UFragment = OperandFragment<InputType, 1, COL_FRAGMENTS, Ops>;
  threadgroup float* u_tile = reinterpret_cast<threadgroup float*>(u_tile_scratch);

  const uint batch_idx = batch_value_head_idx / value_heads;
  const uint value_head_idx = batch_value_head_idx - batch_idx * value_heads;
  const uint qk_head_idx = value_head_idx / (value_heads / qk_heads);
  const uint row_base = (row_tile_group_idx * SIMDGROUPS_PER_TG + thread_context.simdgroup_index) * ROWS;
  const uint value_base = v_tile_idx * MATMUL_COLS;

  const short valid_rows = short(min(uint(ROWS), tree_size - min(row_base, tree_size)));
  const short valid_cols = short(min(MATMUL_COLS, head_v_dim - min(value_base, head_v_dim)));

  AccFragment acc;
  acc.clear();

  const int h0_index = use_h0 ? h0_indices[batch_idx] : -1;
  if (use_h0 && h0_index >= 0) {
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
      if constexpr (transposed_h0) {
        h0_frag.load_from(
            thread_context.simd_lane_id,
            fragment_source(h0 + h0_base + k0, int(head_k_dim)).bounded(valid_cols, valid_k)
        );
      } else {
        h0_frag.load_from(
            thread_context.simd_lane_id,
            fragment_source(h0 + h0_base + k0, 1, int(head_k_dim)).bounded(valid_k, valid_cols)
        );
      }
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
    if constexpr (transposed_h0) {
      for (uint index = tid; index < uint(ROWS) * MATMUL_COLS; index += SIMDGROUPS_PER_TG * METAL_SIMD_SIZE) {
        const uint local_j = index / MATMUL_COLS;
        const uint local_col = index - local_j * MATMUL_COLS;
        const bool in_bounds = local_j < uint(valid_j) && local_col < uint(valid_cols);
        u_tile[index] = in_bounds ? u[u_base + (j0 + local_j) * head_v_dim + local_col] : 0.0f;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      u_frag.load_from(
          thread_context.simd_lane_id,
          fragment_source(u_tile, int(MATMUL_COLS)).bounded(valid_j, valid_cols)
      );
      fragment_mma(acc, qkd_frag, u_frag);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {
      u_frag.load_from(
          thread_context.simd_lane_id,
          fragment_source(u + u_base + j0 * head_v_dim, int(head_v_dim)).bounded(valid_j, valid_cols)
      );
      fragment_mma(acc, qkd_frag, u_frag);
    }
  }

  device OutputT* out_tile =
      o + ((batch_idx * tree_size + row_base) * value_heads + value_head_idx) * head_v_dim + value_base;
  acc.store_safe(thread_context.simd_lane_id, out_tile, int(value_heads * head_v_dim), short2(valid_cols, valid_rows));
}
