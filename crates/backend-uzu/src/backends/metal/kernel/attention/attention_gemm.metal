#include <metal_stdlib>
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../matmul/common/loader.h"
#include "../matmul/common/fragment.h"
#include "../matmul/common/simdgroup_multiply_accumulate.h"
#include "../generated/ring.h"
#include "../generated/trie.h"
#include "../generated/attention.h"
#include "mask.h"

using namespace metal;
using namespace uzu::matmul;
using namespace uzu::attention;
using namespace uzu::ring;
using namespace uzu::trie;

template <typename T>
struct TransformScale {
  T scale;
  METAL_FUNC TransformScale(T scale_) : scale(scale_) {}
  METAL_FUNC T apply(T x) const { return scale * x; }
};

template <typename T>
METAL_FUNC T row_reduce_max(T v) {
  // SimdgroupFragmentOps::get_position mapping groups lanes for
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
    const constant RingParams& ring_params OPTIONAL(is_kv_cache_ring),
    const device TrieNode* trie OPTIONAL(is_trie),
    const constant uint& sliding_window_size OPTIONAL(is_sliding_window),
    const device float* sinks OPTIONAL(has_sinks),
    const constant uint& num_heads,
    const constant uint& suffix_length,
    const bool align_q SPECIALIZE,
    const bool align_k SPECIALIZE,
    const bool is_kv_cache_ring SPECIALIZE,
    const bool is_causal SPECIALIZE,
    const bool is_trie SPECIALIZE,
    const bool is_sliding_window SPECIALIZE,
    const bool has_sinks SPECIALIZE,
    threadgroup T q_smem[BLOCK_QUERY_ROWS * (BD + 16 / sizeof(T))],
    threadgroup T kv_smem[BK * (BD + 16 / sizeof(T))],
    const ThreadContext thread_context,
    const uint q_tile_idx GROUPS(suffix_length.div_ceil(BLOCK_QUERY_ROWS)),
    const uint head_idx GROUPS(num_heads),
    const uint batch_idx GROUPS(1),
    const uint lid THREADS(128)
) {
  // -------------------------------------------------------------------------
  // Pointer setup (all strides are in elements)
  // q_tile_idx: query tile index (BLOCK_QUERY_ROWS rows)
  // head_idx: query head index
  // batch_idx: batch index (currently 1 in uzu, but kept for completeness)

  q += batch_idx * params.q_strides[0] + head_idx * params.q_strides[1] +
       q_tile_idx * int64_t(BLOCK_QUERY_ROWS) * params.q_strides[2];

  const int kv_head_idx = int(head_idx) / params.gqa_factor;
  k += batch_idx * params.k_strides[0] +
       int64_t(kv_head_idx) * params.k_strides[1];
  v += batch_idx * params.v_strides[0] +
       int64_t(kv_head_idx) * params.v_strides[1];

  o += batch_idx * params.o_strides[0] + head_idx * params.o_strides[1] +
       q_tile_idx * int64_t(BLOCK_QUERY_ROWS) * params.o_strides[2];

  if (is_trie) {
    trie += batch_idx * suffix_length;
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
  using SimdgroupFragmentOpsType = SimdgroupFragmentOps<AccumType>;

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
  constexpr ushort QUERY_GRID_ROWS =
      BLOCK_QUERY_ROWS / (SIMDGROUPS_PER_THREADGROUP * SIMDGROUP_BLOCK_SIZE);
  constexpr ushort KEY_GRID_COLS = BK / SIMDGROUP_BLOCK_SIZE;
  constexpr ushort HEAD_DIM_GRID_COLS = BD / SIMDGROUP_BLOCK_SIZE;

  static_assert(QUERY_GRID_ROWS == 1, "Expected QUERY_GRID_ROWS == 1");

  Fragment<AccumType, QUERY_GRID_ROWS, 1, SimdgroupFragmentOpsType>
      query_fragment(thread_context);
  Fragment<AccumType, 1, KEY_GRID_COLS, SimdgroupFragmentOpsType> key_fragment(
      thread_context
  );
  Fragment<AccumType, QUERY_GRID_ROWS, KEY_GRID_COLS, SimdgroupFragmentOpsType>
      score_fragment(thread_context);
  Fragment<AccumType, 1, 1, SimdgroupFragmentOpsType> value_fragment(
      thread_context
  );
  Fragment<
      AccumType,
      QUERY_GRID_ROWS,
      HEAD_DIM_GRID_COLS,
      SimdgroupFragmentOpsType>
      output_fragment(thread_context);

  output_fragment.clear();

  // -------------------------------------------------------------------------
  // Lane coordinates and pointer offsets
  const short2 position =
      SimdgroupFragmentOpsType::get_position(thread_context);
  const short lane_row = position.y;
  const short lane_col = position.x;

  const short simdgroup_row_base = SIMDGROUP_BLOCK_SIZE * QUERY_GRID_ROWS *
                                   short(thread_context.threadgroup_index);

  const short query_shared_offset =
      simdgroup_row_base * query_leading_dimension;
  constexpr short query_tile_stride = SIMDGROUP_BLOCK_SIZE;

  const short key_shared_offset = 0;
  constexpr short key_tile_stride = SIMDGROUP_BLOCK_SIZE;

  const short value_shared_offset = 0;

  // -------------------------------------------------------------------------
  // Load Q block once (and apply scaling)
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (!align_q && int(q_tile_idx) == params.nq_aligned) {
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
    max_score = M_LOG2E_F * static_cast<AccumType>(sinks[head_idx]);
    sum_score = AccumType(1);
  }

  // Determine K block loop limit (causal can early-stop)
  int kb_lim = params.nk;
  if (is_causal) {
    const int q_max = (int(q_tile_idx) + 1) * BLOCK_QUERY_ROWS + params.q_off;
    kb_lim = (q_max + BK - 1) / BK;
    kb_lim = min(params.nk, kb_lim);
  }

  const int q_rel = int(q_tile_idx) * BLOCK_QUERY_ROWS +
                    int(simdgroup_row_base) + int(lane_row); // [0, q_len)

  const int prefix_length = params.q_off;

  const int suffix_position =
      is_kv_cache_ring ? int(ring_params.ring_length) : prefix_length;

  const int query_position = (is_trie && q_rel < params.q_len)
                                 ? suffix_position + int(trie[q_rel].height)
                                 : suffix_position + q_rel;

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

      query_fragment.load(
          &query_shared[query_shared_offset + dd * query_tile_stride],
          query_leading_dimension,
          1
      );
      key_fragment.load(
          &key_shared[key_shared_offset + dd * key_tile_stride],
          1,
          key_leading_dimension
      );

      simdgroup_barrier(mem_flags::mem_none);
      SimdgroupFragmentOpsType::template tile_matmul<false, false>(
          score_fragment,
          query_fragment,
          key_fragment
      );
    }

    // Mask out tail keys for the last (unaligned) K block
    if (!align_k && kb == params.nk_aligned) {
      const int k_rem = params.k_rem;
      METAL_PRAGMA_UNROLL
      for (short j = 0; j < KEY_GRID_COLS; j++) {
        thread auto& frag = score_fragment.fragment_at(0, j);
        const int col0 = int(lane_col) + int(j) * SIMDGROUP_BLOCK_SIZE;
        if (col0 >= k_rem) {
          frag[0] = neg_inf;
        }
        if ((col0 + 1) >= k_rem) {
          frag[1] = neg_inf;
        }
      }
    }

    // Unified masking: causal, trie, ring, sliding window
    if (q_rel < params.q_len) {
      METAL_PRAGMA_UNROLL
      for (short j = 0; j < KEY_GRID_COLS; j++) {
        thread auto& frag = score_fragment.fragment_at(0, j);
        const int col_base =
            kb * BK + int(lane_col) + int(j) * SIMDGROUP_BLOCK_SIZE;

        const int k0 = col_base;
        const int k1 = col_base + 1;

        if (k0 < params.k_len && !should_use_key(
                                     ring_params,
                                     trie,
                                     sliding_window_size,
                                     q_rel,
                                     prefix_length,
                                     suffix_position,
                                     query_position,
                                     k0,
                                     is_kv_cache_ring,
                                     is_causal,
                                     is_trie,
                                     is_sliding_window
                                 ))
          frag[0] = neg_inf;

        if (k1 < params.k_len && !should_use_key(
                                     ring_params,
                                     trie,
                                     sliding_window_size,
                                     q_rel,
                                     prefix_length,
                                     suffix_position,
                                     query_position,
                                     k1,
                                     is_kv_cache_ring,
                                     is_causal,
                                     is_trie,
                                     is_sliding_window
                                 ))
          frag[1] = neg_inf;
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
      const thread auto& frag = score_fragment.fragment_at(0, j);
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
      thread auto& frag = score_fragment.fragment_at(0, j);
      frag[0] = fast::exp2(frag[0] - new_max);
      frag[1] = fast::exp2(frag[1] - new_max);
      block_sum_local += frag[0] + frag[1];
    }
    const AccumType block_sum = row_reduce_sum(block_sum_local);
    sum_score += block_sum;

    // Rescale output accumulator
    METAL_PRAGMA_UNROLL
    for (short id = 0; id < HEAD_DIM_GRID_COLS; id++) {
      thread auto& frag = output_fragment.fragment_at(0, id);
      frag[0] *= factor;
      frag[1] *= factor;
    }

    // Accumulate output: output_fragment += score_fragment * Vblock
    threadgroup_barrier(mem_flags::mem_threadgroup);
    METAL_PRAGMA_UNROLL
    for (short id = 0; id < HEAD_DIM_GRID_COLS; id++) {
      METAL_PRAGMA_UNROLL
      for (short ik = 0; ik < KEY_GRID_COLS; ik++) {
        if constexpr (BD == 128) {
          simdgroup_barrier(mem_flags::mem_none);
        }

        const short kk = ik * SIMDGROUP_BLOCK_SIZE;
        const short dd = id * SIMDGROUP_BLOCK_SIZE;

        value_fragment.load(
            &value_shared
                [value_shared_offset + kk * value_leading_dimension + dd],
            value_leading_dimension,
            1
        );

        if constexpr (BD == 128) {
          simdgroup_barrier(mem_flags::mem_none);
        }

        SimdgroupFragmentOpsType::multiply_accumulate(
            output_fragment.fragment_at(0, id),
            score_fragment.fragment_at(0, ik),
            value_fragment.fragment_at(0, 0),
            output_fragment.fragment_at(0, id)
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
    thread auto& frag = output_fragment.fragment_at(0, id);
    frag[0] *= inv_sum;
    frag[1] *= inv_sum;
  }

  threadgroup_barrier(mem_flags::mem_none);

  // Store results (O is row-major with row-stride params.o_strides[2])
  o += int64_t(simdgroup_row_base) * params.o_strides[2];

  if (!align_q && int(q_tile_idx) == params.nq_aligned) {
    const short2 dst_tile_dims = short2(BD, params.q_rem - simdgroup_row_base);

    if (dst_tile_dims.x <= 0 || dst_tile_dims.y <= 0) {
      return;
    }

    output_fragment.store_safe(
        o,
        int(params.o_strides[2]),
        1,
        dst_tile_dims.y,
        dst_tile_dims.x
    );
  } else {
    output_fragment.store(o, int(params.o_strides[2]), 1);
  }
}
