#include <metal_stdlib>
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../matmul/common/loader.h"
#include "../matmul/common/simdgroup_fragment.h"
#include "attention.h"

using namespace metal;
using namespace uzu::matmul;
using namespace uzu::attention;

template <typename T>
struct TransformScale {
  T scale;
  METAL_FUNC TransformScale(T scale_) : scale(scale_) {}
  METAL_FUNC T apply(T x) const { return scale * x; }
};

template <typename T>
METAL_FUNC T row_reduce_max(T v) {
  // SimdgroupMultiplyAccumulate::get_lane_coordinates mapping groups lanes for
  // a row as: {lane, lane^1, lane^8, (lane^1)^8}. Reduce in two steps.
  v = metal::max(v, simd_shuffle_xor(v, 1));
  v = metal::max(v, simd_shuffle_xor(v, 8));
  return v;
}

template <typename T>
METAL_FUNC T row_reduce_sum(T v) {
  v += simd_shuffle_xor(v, 1);
  v += simd_shuffle_xor(v, 8);
  return v;
}

#define BLOCK_QUERY_ROWS 32
#define SIMDGROUPS_PER_ROW 4
#define SIMDGROUPS_PER_COLUMN 1

template <typename T, uint BK, uint BD>
VARIANTS(T, float, half, bfloat)
VARIANTS(BK, 16, 32)
VARIANTS(BD, 64, 128, 256)
PUBLIC KERNEL(AttentionGemm)(
    const device T* q,
    const device T* k,
    const device T* v,
    device T* o,
    const constant AttnParams& params,
    const constant AttnMaskParams& mask_params OPTIONAL(has_mask),
    const device T* mask OPTIONAL(has_mask),
    const device float* sinks OPTIONAL(has_sinks),
    const constant uint& num_heads,
    const constant uint& suffix_length,
    const bool align_q SPECIALIZE,
    const bool align_k SPECIALIZE,
    const bool do_causal SPECIALIZE,
    const bool has_mask SPECIALIZE,
    const bool has_sinks SPECIALIZE,
    threadgroup T q_smem[BLOCK_QUERY_ROWS * (BD + 16 / sizeof(T))],
    threadgroup T kv_smem[BK * (BD + 16 / sizeof(T))],
    const ThreadContext thread_context,
    const uint tgid_x GROUPS(suffix_length.div_ceil(BLOCK_QUERY_ROWS)),
    const uint tgid_y GROUPS(num_heads),
    const uint tgid_z GROUPS(1),
    const uint lid THREADS(128)
) {
  // -------------------------------------------------------------------------
  // Pointer setup (all strides are in elements)
  // tgid_x: query tile index (BLOCK_QUERY_ROWS rows)
  // tgid_y: query head index
  // tgid_z: batch index (currently 1 in uzu, but kept for completeness)
  const uint batch_idx = tgid_z;
  const uint head_idx = tgid_y;
  const uint q_tile_idx = tgid_x;

  q += batch_idx * params.q_strides[0] + head_idx * params.q_strides[1] +
       q_tile_idx * int64_t(BLOCK_QUERY_ROWS) * params.q_strides[2];

  const int kv_head_idx = int(tgid_y) / params.gqa_factor;
  k += batch_idx * params.k_strides[0] +
       int64_t(kv_head_idx) * params.k_strides[1];
  v += batch_idx * params.v_strides[0] +
       int64_t(kv_head_idx) * params.v_strides[1];

  o += batch_idx * params.o_strides[0] + head_idx * params.o_strides[1] +
       q_tile_idx * int64_t(BLOCK_QUERY_ROWS) * params.o_strides[2];

  if (has_mask) {
    mask += batch_idx * mask_params.m_strides[0] +
            head_idx * mask_params.m_strides[1];
  }

  // -------------------------------------------------------------------------
  // Threadgroup memory
  constexpr short query_padding = 16 / sizeof(T);
  constexpr short key_padding = 16 / sizeof(T);
  constexpr short value_padding = 16 / sizeof(T);

  constexpr short query_leading_dimension = BD + query_padding;
  constexpr short key_leading_dimension = BD + key_padding;
  constexpr short value_leading_dimension = BD + value_padding;

  threadgroup T* query_shared = q_smem;
  threadgroup T* key_shared = kv_smem;
  threadgroup T* value_shared = kv_smem;

  //
  // -------------------------------------------------------------------------
  // Block loaders
  using QueryLoader = ThreadgroupLoader<
      T,
      BLOCK_QUERY_ROWS,
      BD,
      query_leading_dimension,
      1,
      SIMDGROUPS_PER_ROW * SIMDGROUPS_PER_COLUMN * 32>;

  using KeyLoader = ThreadgroupLoader<
      T,
      BK,
      BD,
      key_leading_dimension,
      0,
      SIMDGROUPS_PER_ROW * SIMDGROUPS_PER_COLUMN * 32>;

  using ValueLoader = ThreadgroupLoader<
      T,
      BK,
      BD,
      value_leading_dimension,
      0,
      SIMDGROUPS_PER_ROW * SIMDGROUPS_PER_COLUMN * 32>;

  const int query_source_stride = int(params.q_strides[2]);
  const int key_source_stride = int(params.k_strides[2]);
  const int value_source_stride = int(params.v_strides[2]);

  thread QueryLoader query_loader(
      q,
      query_source_stride,
      query_shared,
      thread_context.threadgroup_index,
      thread_context.simdgroup_index
  );
  thread KeyLoader key_loader(
      k,
      key_source_stride,
      key_shared,
      thread_context.threadgroup_index,
      thread_context.simdgroup_index
  );
  thread ValueLoader value_loader(
      v,
      value_source_stride,
      value_shared,
      thread_context.threadgroup_index,
      thread_context.simdgroup_index
  );

  TransformScale<T> ts(static_cast<T>(params.scale * M_LOG2E_F));

  // -------------------------------------------------------------------------
  // MMA tiles
  constexpr short SIMDGROUP_BLOCK_SIZE = 8;
  using AccumType = float;
  using SimdgroupMultiplyAccumulateType = SimdgroupMultiplyAccumulate<
      AccumType,
      SIMDGROUP_BLOCK_SIZE,
      SIMDGROUP_BLOCK_SIZE>;

  constexpr int SIMDGROUPS_PER_THREADGROUP =
      SIMDGROUPS_PER_ROW * SIMDGROUPS_PER_COLUMN;
  static_assert(
      BLOCK_QUERY_ROWS >= (SIMDGROUPS_PER_THREADGROUP * SIMDGROUP_BLOCK_SIZE) &&
          BLOCK_QUERY_ROWS %
                  (SIMDGROUPS_PER_THREADGROUP * SIMDGROUP_BLOCK_SIZE) ==
              0,
      "Each simdgroup must host at least 1 simdgroup matrix along Q sequence."
  );

  // Q sequence multiply-accumulate blocks per simdgroup (QUERY_GRID_ROWS == 1
  // for the 32-row block layout)
  constexpr int QUERY_GRID_ROWS =
      BLOCK_QUERY_ROWS / (SIMDGROUPS_PER_THREADGROUP * SIMDGROUP_BLOCK_SIZE);
  constexpr int KEY_GRID_COLS = BK / SIMDGROUP_BLOCK_SIZE;
  constexpr int HEAD_DIM_GRID_COLS = BD / SIMDGROUP_BLOCK_SIZE;

  static_assert(QUERY_GRID_ROWS == 1, "Expected QUERY_GRID_ROWS == 1");

  SimdgroupFragment<
      AccumType,
      QUERY_GRID_ROWS,
      1,
      SimdgroupMultiplyAccumulateType>
      query_fragment;
  SimdgroupFragment<
      AccumType,
      1,
      KEY_GRID_COLS,
      SimdgroupMultiplyAccumulateType>
      key_fragment;
  SimdgroupFragment<
      AccumType,
      QUERY_GRID_ROWS,
      KEY_GRID_COLS,
      SimdgroupMultiplyAccumulateType>
      score_fragment;
  SimdgroupFragment<AccumType, 1, 1, SimdgroupMultiplyAccumulateType>
      value_fragment;
  SimdgroupFragment<
      AccumType,
      QUERY_GRID_ROWS,
      HEAD_DIM_GRID_COLS,
      SimdgroupMultiplyAccumulateType>
      output_fragment;

  output_fragment.clear();

  // -------------------------------------------------------------------------
  // Lane coordinates and pointer offsets
  const short2 lane_coordinates =
      SimdgroupMultiplyAccumulateType::get_lane_coordinates(
          thread_context.simdgroup_index
      );
  const short lane_row = lane_coordinates.y;
  const short lane_col = lane_coordinates.x;

  const short simdgroup_row_base = SIMDGROUP_BLOCK_SIZE * QUERY_GRID_ROWS *
                                   short(thread_context.threadgroup_index);

  const short query_shared_offset =
      (simdgroup_row_base + lane_row) * query_leading_dimension + lane_col;
  constexpr short query_tile_stride = SIMDGROUP_BLOCK_SIZE;

  const short key_shared_offset = lane_col * key_leading_dimension + lane_row;
  constexpr short key_tile_stride = SIMDGROUP_BLOCK_SIZE;

  const short value_shared_offset =
      lane_row * value_leading_dimension + lane_col;

  // -------------------------------------------------------------------------
  // Load Q block once (and apply scaling)
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (!align_q && int(tgid_x) == params.nq_aligned) {
    query_loader.load_safe(short2(BD, params.q_rem));
  } else {
    query_loader.load_unsafe();
  }
  query_loader.apply_inplace_op(ts);

  // -------------------------------------------------------------------------
  // Streaming softmax state for this row (shared across lanes in a row)
  const AccumType neg_inf = static_cast<AccumType>(-1e9f) * M_LOG2E_F;
  AccumType max_score = -INFINITY;
  AccumType sum_score = AccumType(0);

  if (has_sinks) {
    max_score = M_LOG2E_F * static_cast<AccumType>(sinks[tgid_y]);
    sum_score = AccumType(1);
  }

  // Determine K block loop limit (causal can early-stop)
  int kb_lim = params.nk;
  if (do_causal) {
    const int q_max = (int(tgid_x) + 1) * BLOCK_QUERY_ROWS + params.q_off;
    kb_lim = (q_max + BK - 1) / BK;
    kb_lim = min(params.nk, kb_lim);
  }

  const int q_rel = int(tgid_x) * BLOCK_QUERY_ROWS + int(simdgroup_row_base) +
                    int(lane_row);        // [0, q_len)
  const int q_abs = q_rel + params.q_off; // [0, k_len)

  // Loop over KV blocks
  for (int kb = 0; kb < kb_lim; kb++) {
    // Load K block
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!align_k && kb == params.nk_aligned) {
      key_loader.load_safe(short2(BD, params.k_rem));
    } else {
      key_loader.load_unsafe();
    }

    // Compute S = Q @ K^T for this block
    score_fragment.clear();

    threadgroup_barrier(mem_flags::mem_threadgroup);

    METAL_PRAGMA_UNROLL
    for (short dd = 0; dd < HEAD_DIM_GRID_COLS; dd++) {
      simdgroup_barrier(mem_flags::mem_none);

      query_fragment.template load<T, 1, 1, query_leading_dimension, 1>(
          &query_shared[query_shared_offset + dd * query_tile_stride]
      );
      key_fragment.template load<T, 1, 1, 1, key_leading_dimension>(
          &key_shared[key_shared_offset + dd * key_tile_stride]
      );

      simdgroup_barrier(mem_flags::mem_none);
      tile_multiply_accumulate(
          score_fragment,
          query_fragment,
          key_fragment,
          score_fragment
      );
    }

    // Mask out tail keys for the last (unaligned) K block
    if (!align_k && kb == params.nk_aligned) {
      const int k_rem = params.k_rem;
      METAL_PRAGMA_UNROLL
      for (short j = 0; j < KEY_GRID_COLS; j++) {
        thread auto& frag = score_fragment.multiply_accumulate_at(0, j);
        const int col0 = int(lane_col) + int(j) * SIMDGROUP_BLOCK_SIZE;
        if (col0 >= k_rem) {
          frag[0] = neg_inf;
        }
        if ((col0 + 1) >= k_rem) {
          frag[1] = neg_inf;
        }
      }
    }

    // Causal mask (only needed for the last few blocks near the diagonal)
    if (do_causal) {
      const int tail_blocks = (BLOCK_QUERY_ROWS + BK - 1) / BK + int(!align_k);
      const int tail_start = kb_lim - tail_blocks;
      if (kb >= tail_start) {
        METAL_PRAGMA_UNROLL
        for (short j = 0; j < KEY_GRID_COLS; j++) {
          thread auto& frag = score_fragment.multiply_accumulate_at(0, j);
          const int col_base =
              kb * BK + int(lane_col) + int(j) * SIMDGROUP_BLOCK_SIZE;
          if (q_abs < col_base) {
            frag[0] = neg_inf;
          }
          if (q_abs < (col_base + 1)) {
            frag[1] = neg_inf;
          }
        }
      }
    }

    // Add external mask (additive bias in natural-log domain; convert to log2)
    if (has_mask && q_rel < params.q_len) {
      const int64_t row_stride = mask_params.m_strides[2];
      const int64_t row_base = int64_t(q_rel) * row_stride;

      METAL_PRAGMA_UNROLL
      for (short j = 0; j < KEY_GRID_COLS; j++) {
        thread auto& frag = score_fragment.multiply_accumulate_at(0, j);
        const int col_base =
            kb * BK + int(lane_col) + int(j) * SIMDGROUP_BLOCK_SIZE;

        const int k0 = col_base;
        const int k1 = col_base + 1;

        if (k0 < params.k_len) {
          AccumType mv = static_cast<AccumType>(mask[row_base + int64_t(k0)]);
          mv = metal::max(mv, static_cast<AccumType>(-1e9f));
          frag[0] += M_LOG2E_F * mv;
        }
        if (k1 < params.k_len) {
          AccumType mv = static_cast<AccumType>(mask[row_base + int64_t(k1)]);
          mv = metal::max(mv, static_cast<AccumType>(-1e9f));
          frag[1] += M_LOG2E_F * mv;
        }
      }
    }

    // Load V block (overwriting K in shared memory)
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!align_k && kb == params.nk_aligned) {
      value_loader.load_safe(short2(BD, params.k_rem));
    } else {
      value_loader.load_unsafe();
    }

    // -----------------------------------------------------------------------
    // Streaming softmax update for this block

    // Row max for this block
    AccumType block_max_local = -INFINITY;
    METAL_PRAGMA_UNROLL
    for (short j = 0; j < KEY_GRID_COLS; j++) {
      const thread auto& frag = score_fragment.multiply_accumulate_at(0, j);
      block_max_local = metal::max(block_max_local, frag[0]);
      block_max_local = metal::max(block_max_local, frag[1]);
    }
    const AccumType block_max = row_reduce_max(block_max_local);

    const AccumType new_max = metal::max(max_score, block_max);
    const AccumType factor = fast::exp2(max_score - new_max);
    max_score = new_max;

    // Rescale running sum
    sum_score *= factor;

    // exp2(S - new_max) and row sum
    AccumType block_sum_local = AccumType(0);
    METAL_PRAGMA_UNROLL
    for (short j = 0; j < KEY_GRID_COLS; j++) {
      thread auto& frag = score_fragment.multiply_accumulate_at(0, j);
      frag[0] = fast::exp2(frag[0] - new_max);
      frag[1] = fast::exp2(frag[1] - new_max);
      block_sum_local += frag[0] + frag[1];
    }
    const AccumType block_sum = row_reduce_sum(block_sum_local);
    sum_score += block_sum;

    // Rescale output accumulator
    METAL_PRAGMA_UNROLL
    for (short id = 0; id < HEAD_DIM_GRID_COLS; id++) {
      thread auto& frag = output_fragment.multiply_accumulate_at(0, id);
      frag[0] *= factor;
      frag[1] *= factor;
    }

    // Accumulate output: output_fragment += score_fragment * Vblock
    threadgroup_barrier(mem_flags::mem_threadgroup);
    METAL_PRAGMA_UNROLL
    for (short id = 0; id < HEAD_DIM_GRID_COLS; id++) {
      METAL_PRAGMA_UNROLL
      for (short ik = 0; ik < KEY_GRID_COLS; ik++) {
        IF_CONSTEXPR(BD == 128) { simdgroup_barrier(mem_flags::mem_none); }

        const short kk = ik * SIMDGROUP_BLOCK_SIZE;
        const short dd = id * SIMDGROUP_BLOCK_SIZE;

        value_fragment.template load<T, 1, 1, value_leading_dimension, 1>(
            &value_shared
                [value_shared_offset + kk * value_leading_dimension + dd]
        );

        IF_CONSTEXPR(BD == 128) { simdgroup_barrier(mem_flags::mem_none); }

        SimdgroupMultiplyAccumulateType::multiply_accumulate(
            output_fragment.multiply_accumulate_at(0, id),
            score_fragment.multiply_accumulate_at(0, ik),
            value_fragment.multiply_accumulate_at(0, 0),
            output_fragment.multiply_accumulate_at(0, id)
        );
      }
    }

    // Prepare for next iteration
    key_loader.next();
    value_loader.next();
  }

  // -------------------------------------------------------------------------
  // Normalize output by sum_score (avoid div-by-zero for masked-out rows)
  const AccumType inv_sum = AccumType(1) / sum_score;
  METAL_PRAGMA_UNROLL
  for (short id = 0; id < HEAD_DIM_GRID_COLS; id++) {
    thread auto& frag = output_fragment.multiply_accumulate_at(0, id);
    frag[0] *= inv_sum;
    frag[1] *= inv_sum;
  }

  threadgroup_barrier(mem_flags::mem_none);

  // Store results (O is row-major with row-stride params.o_strides[2])
  o += int64_t(simdgroup_row_base + lane_row) * params.o_strides[2] +
       int64_t(lane_col);

  if (!align_q && int(tgid_x) == params.nq_aligned) {
    const short2 dst_tile_dims =
        short2(BD - lane_col, params.q_rem - (simdgroup_row_base + lane_row));

    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0) {
      return;
    }

    output_fragment.template store_safe<T, 1, 1>(
        o,
        int(params.o_strides[2]),
        dst_tile_dims
    );
  } else {
    output_fragment.template store<T, 1, 1>(o, int(params.o_strides[2]));
  }
}
