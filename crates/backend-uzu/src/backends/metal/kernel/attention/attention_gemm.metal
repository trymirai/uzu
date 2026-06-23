#include <metal_stdlib>
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../matmul/common/fragment.h"
#include "../matmul/common/loader.h"
#include "../matmul/common/mxu_fragment_ops.h"
#include "../matmul/common/simdgroup_fragment_ops.h"
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

#define FRAG_ROWS (USE_ACCELERATOR ? 16 : 8)
#define BLOCK_QUERY_ROWS (4 * FRAG_ROWS)
#define SIMDGROUPS_PER_ROW 4
#define SIMDGROUPS_PER_COLUMN 1

// Resident Q regresses simdgroup GPUs; keep it MXU-only.
#define Q_IS_RESIDENT (USE_ACCELERATOR)
#define QSMEM_SIZE (Q_IS_RESIDENT ? 1 : (BLOCK_QUERY_ROWS * (int(BD) + 16 / int(sizeof(T)))))

// Direct K/V reads regress small-cache simdgroup GPUs; keep them MXU-only.
#define KVSMEM_SIZE (USE_ACCELERATOR ? 1 : (int(BK) * (int(BD) + 16 / int(sizeof(T)))))

template <typename T, uint BK, uint BD, bool USE_ACCELERATOR>
VARIANTS(T, float, half, bfloat)
VARIANTS(BK, 16, 32)
VARIANTS(BD, 64, 128, 256)
VARIANTS(USE_ACCELERATOR, false, true)
// Match the Rust dispatch gate and skip never-dispatched MXU variants.
CONSTRAINT(!USE_ACCELERATOR || BK == 32)
CONSTRAINT(!USE_ACCELERATOR || T != "float")
CONSTRAINT(!USE_ACCELERATOR || BD != 256)
PUBLIC KERNEL(AttentionGemm)(
    const device T* q,
    const device T* k,
    const device T* v,
    device T* o,
    const constant AttnParams& params,
    const constant RingParams& ring_params OPTIONAL(is_kv_cache_ring),
    const device TrieNode* trie OPTIONAL(is_trie),
    const constant uint& sliding_window_size OPTIONAL(is_sliding_window),
    const device T* sinks OPTIONAL(has_sinks),
    const constant uint& num_heads,
    const constant uint& suffix_length,
    const bool align_q SPECIALIZE,
    const bool align_k SPECIALIZE,
    const bool is_kv_cache_ring SPECIALIZE,
    const bool is_causal SPECIALIZE,
    const bool is_trie SPECIALIZE,
    const bool is_sliding_window SPECIALIZE,
    const bool has_sinks SPECIALIZE,
    threadgroup T q_smem[QSMEM_SIZE],
    threadgroup T kv_smem[KVSMEM_SIZE],
    const ThreadContext thread_context,
    const uint q_tile_idx GROUPS(suffix_length.div_ceil(4 * if USE_ACCELERATOR { 16 } else { 8 })),
    const uint head_idx GROUPS(num_heads),
    const uint batch_idx GROUPS(1),
    const uint lid THREADS(128)
) {
  q += batch_idx * params.q_strides[0] + head_idx * params.q_strides[1] +
       q_tile_idx * int64_t(BLOCK_QUERY_ROWS) * params.q_strides[2];

  const int kv_head_idx = int(head_idx) / params.gqa_factor;
  k += batch_idx * params.k_strides[0] + int64_t(kv_head_idx) * params.k_strides[1];
  v += batch_idx * params.v_strides[0] + int64_t(kv_head_idx) * params.v_strides[1];

  o += batch_idx * params.o_strides[0] + head_idx * params.o_strides[1] +
       q_tile_idx * int64_t(BLOCK_QUERY_ROWS) * params.o_strides[2];

  if (is_trie) {
    trie += batch_idx * suffix_length;
  }

  const int query_source_stride = int(params.q_strides[2]);
  const int key_source_stride = int(params.k_strides[2]);
  const int value_source_stride = int(params.v_strides[2]);

  using AccumType = float;
  using Ops = metal::conditional_t<USE_ACCELERATOR, MxuFragmentOps, SimdgroupFragmentOps>;
  constexpr short SIMDGROUP_BLOCK_SIZE = Ops::FRAGMENT_ROWS;

  // MXU uses input dtype with f32 accumulate; simdgroup MMA needs matched types.
  using InputType = metal::conditional_t<USE_ACCELERATOR, T, AccumType>;

  constexpr int SIMDGROUPS_PER_THREADGROUP = SIMDGROUPS_PER_ROW * SIMDGROUPS_PER_COLUMN;
  static_assert(
      BLOCK_QUERY_ROWS >= (SIMDGROUPS_PER_THREADGROUP * SIMDGROUP_BLOCK_SIZE) &&
          BLOCK_QUERY_ROWS % (SIMDGROUPS_PER_THREADGROUP * SIMDGROUP_BLOCK_SIZE) == 0,
      "Each simdgroup must host at least 1 simdgroup matrix along Q sequence."
  );

  constexpr int QUERY_GRID_ROWS = BLOCK_QUERY_ROWS / (SIMDGROUPS_PER_THREADGROUP * SIMDGROUP_BLOCK_SIZE);
  constexpr int KEY_GRID_COLS = BK / SIMDGROUP_BLOCK_SIZE;
  constexpr int HEAD_DIM_GRID_COLS = BD / SIMDGROUP_BLOCK_SIZE;

  static_assert(QUERY_GRID_ROWS == 1, "Expected QUERY_GRID_ROWS == 1");
  static_assert(!USE_ACCELERATOR || KEY_GRID_COLS % 2 == 0, "MXU QK needs even N (KEY_GRID_COLS)");

  constexpr int Q_FRAG_COUNT = Q_IS_RESIDENT ? HEAD_DIM_GRID_COLS : 1;
  Fragment<InputType, QUERY_GRID_ROWS, 1, Ops> query_frags[Q_FRAG_COUNT];
  Fragment<InputType, 1, KEY_GRID_COLS, Ops> key_fragment;
  Fragment<AccumType, QUERY_GRID_ROWS, KEY_GRID_COLS, Ops> score_fragment;
  constexpr int VCHUNK = USE_ACCELERATOR ? 2 : HEAD_DIM_GRID_COLS;
  static_assert(HEAD_DIM_GRID_COLS % VCHUNK == 0, "head-dim must split into VCHUNK frags");
  static_assert(!USE_ACCELERATOR || VCHUNK % 2 == 0, "MXU PV needs even N (VCHUNK)");
  constexpr int N_OUT_CHUNKS = HEAD_DIM_GRID_COLS / VCHUNK;
  constexpr short CHUNK_COLS = VCHUNK * SIMDGROUP_BLOCK_SIZE;
  Fragment<AccumType, QUERY_GRID_ROWS, VCHUNK, Ops> output_chunks[N_OUT_CHUNKS];

  METAL_PRAGMA_UNROLL
  for (int c = 0; c < N_OUT_CHUNKS; ++c) {
    output_chunks[c].clear();
  }

  const short simdgroup_row_base = SIMDGROUP_BLOCK_SIZE * QUERY_GRID_ROWS * short(thread_context.simdgroup_index);
  constexpr short head_dim_tile_stride = SIMDGROUP_BLOCK_SIZE;

  const bool ragged_q = (!align_q && int(q_tile_idx) == params.nq_aligned);
  constexpr short query_leading_dimension = BD + 16 / sizeof(T);
  threadgroup T* query_shared = q_smem;
  const short query_shared_offset = simdgroup_row_base * query_leading_dimension;

  if constexpr (Q_IS_RESIDENT) {
    auto q_src = tile_source(q + int64_t(simdgroup_row_base) * query_source_stride, query_source_stride);
    if (ragged_q) {
      q_src = q_src.bounded(params.q_rem - simdgroup_row_base, SIMDGROUP_BLOCK_SIZE);
    }
    const InputType query_scale = static_cast<InputType>(params.scale * M_LOG2E_F);
    METAL_PRAGMA_UNROLL
    for (short dd = 0; dd < HEAD_DIM_GRID_COLS; dd++) {
      query_frags[dd].load_from(thread_context.simd_lane_id, q_src.advanced(dd * head_dim_tile_stride));
      query_frags[dd].for_each_element(thread_context.simd_lane_id, [&](short, short, thread InputType& v) {
        v *= query_scale;
      });
    }
  } else {
    using QueryLoader =
        ThreadgroupLoader<T, BLOCK_QUERY_ROWS, BD, query_leading_dimension, 1, SIMDGROUPS_PER_THREADGROUP * 32>;
    thread QueryLoader query_loader(q, query_source_stride, query_shared, thread_context);
    if (ragged_q) {
      query_loader.load_safe(short2(BD, params.q_rem));
    } else {
      query_loader.load_unsafe();
    }
    query_loader.apply_inplace_op(TransformScale<T>(static_cast<T>(params.scale * M_LOG2E_F)));
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Finite neg_inf keeps fully masked rows from producing NaN.
  const AccumType neg_inf = static_cast<AccumType>(-1e9f) * M_LOG2E_F;
  constexpr int ROWS_PER_LANE = QUERY_GRID_ROWS * Ops::THREAD_ELEMENT_ROWS;
  const AccumType init_max = has_sinks ? AccumType(M_LOG2E_F * static_cast<AccumType>(sinks[head_idx])) : -INFINITY;
  const AccumType init_sum = has_sinks ? AccumType(1) : AccumType(0);
  AccumType max_score[ROWS_PER_LANE];
  AccumType sum_score[ROWS_PER_LANE];
  METAL_PRAGMA_UNROLL
  for (int r = 0; r < ROWS_PER_LANE; ++r) {
    max_score[r] = init_max;
    sum_score[r] = init_sum;
  }

  uint32_t kb_lim = params.nk;
  if (is_causal) {
    const int q_max = (int(q_tile_idx) + 1) * BLOCK_QUERY_ROWS + params.q_off;
    kb_lim = min(params.nk, uint32_t((q_max + BK - 1) / BK));
  }

  const int q_base = int(q_tile_idx) * BLOCK_QUERY_ROWS + int(simdgroup_row_base);
  const int prefix_length = params.q_off;
  const int suffix_position = is_kv_cache_ring ? int(ring_params.ring_length) : prefix_length;

  constexpr short kv_leading_dimension = BD + 16 / sizeof(T);
  threadgroup T* kv_shared = kv_smem;

  for (int kb = 0; kb < kb_lim; kb++) {
    const bool tail_k = (!align_k && kb == params.nk_aligned);
    const device T* k_block = k + int64_t(kb) * int(BK) * key_source_stride;
    score_fragment.clear();

    auto k_dev = tile_source(k_block, key_source_stride);
    if (tail_k) {
      k_dev = k_dev.bounded(params.k_rem, SIMDGROUP_BLOCK_SIZE);
    }
    if constexpr (!USE_ACCELERATOR) {
      stage_tile<BK, BD, kv_leading_dimension, SIMDGROUPS_PER_THREADGROUP * 32>(
          k_block,
          key_source_stride,
          kv_shared,
          tail_k ? short(params.k_rem) : short(BK),
          thread_context
      );
    }
    auto k_shared = tile_source(kv_shared, 1, kv_leading_dimension);
    auto q_shared_src = tile_source(query_shared + query_shared_offset, query_leading_dimension);

    METAL_PRAGMA_UNROLL
    for (short dd = 0; dd < HEAD_DIM_GRID_COLS; dd++) {
      if constexpr (!Q_IS_RESIDENT) {
        query_frags[0].load_from(thread_context.simd_lane_id, q_shared_src.advanced(dd * head_dim_tile_stride));
      }
      const short q_idx = Q_IS_RESIDENT ? dd : 0;

      if constexpr (USE_ACCELERATOR) {
        Fragment<InputType, KEY_GRID_COLS, 1, Ops> key_block;
        key_block.load_from(thread_context.simd_lane_id, k_dev.advanced(dd * head_dim_tile_stride));
        simdgroup_barrier(mem_flags::mem_none);
        uzu::matmul::tile_matmul<false, true>(score_fragment, query_frags[q_idx], key_block);
      } else {
        key_fragment.load_from(thread_context.simd_lane_id, k_shared.advanced(dd * head_dim_tile_stride));
        simdgroup_barrier(mem_flags::mem_none);
        uzu::matmul::tile_matmul(score_fragment, query_frags[q_idx], key_fragment);
      }
    }

    {
      const bool apply_tail = (!align_k && kb == params.nk_aligned);
      const int k_rem = params.k_rem;
      if (!is_trie && !is_kv_cache_ring && !is_sliding_window) {
        const bool diag = is_causal && ((int(kb) + 1) * int(BK) - 1 > prefix_length + q_base);
        if (apply_tail || diag) {
          score_fragment.for_each_element(thread_context.simd_lane_id, [&](short row, short col, thread AccumType& v) {
            if (apply_tail && int(col) >= k_rem) {
              v = neg_inf;
              return;
            }
            if (diag && (kb * int(BK) + int(col)) > prefix_length + q_base + int(row)) {
              v = neg_inf;
            }
          });
        }
      } else {
        score_fragment.for_each_element(thread_context.simd_lane_id, [&](short row, short col, thread AccumType& v) {
          if (apply_tail && int(col) >= k_rem) {
            v = neg_inf;
            return;
          }
          const int q_rel = q_base + int(row);
          if (q_rel >= int(params.q_len)) {
            return;
          }
          const int key = kb * BK + int(col);
          if (key >= int(params.k_len)) {
            return;
          }
          const int query_position = is_trie ? suffix_position + int(trie[q_rel].height) : suffix_position + q_rel;
          if (!should_use_key(
                  ring_params,
                  trie,
                  sliding_window_size,
                  q_rel,
                  prefix_length,
                  suffix_position,
                  query_position,
                  key,
                  is_kv_cache_ring,
                  is_causal,
                  is_trie,
                  is_sliding_window
              )) {
            v = neg_inf;
          }
        });
      }
    }

    AccumType block_max[ROWS_PER_LANE];
    score_fragment.row_reduce(block_max, -INFINITY, [](AccumType a, AccumType b) { return metal::max(a, b); });

    AccumType new_max[ROWS_PER_LANE];
    AccumType factor[ROWS_PER_LANE];
    METAL_PRAGMA_UNROLL
    for (int r = 0; r < ROWS_PER_LANE; ++r) {
      new_max[r] = metal::max(max_score[r], block_max[r]);
      factor[r] = fast::exp2(max_score[r] - new_max[r]);
      max_score[r] = new_max[r];
      sum_score[r] *= factor[r];
    }

    score_fragment.row_bin_op(new_max, [](AccumType v, AccumType m) { return fast::exp2(v - m); });

    AccumType block_sum[ROWS_PER_LANE];
    score_fragment.row_reduce(block_sum, AccumType(0), [](AccumType a, AccumType b) { return a + b; });
    METAL_PRAGMA_UNROLL
    for (int r = 0; r < ROWS_PER_LANE; ++r) {
      sum_score[r] += block_sum[r];
    }

    METAL_PRAGMA_UNROLL
    for (int c = 0; c < N_OUT_CHUNKS; ++c) {
      output_chunks[c].row_bin_op(factor, [](AccumType v, AccumType f) { return v * f; });
    }

    const device T* v_block = v + int64_t(kb) * int(BK) * value_source_stride;
    auto v_dev = tile_source(v_block, value_source_stride);
    if (tail_k) {
      v_dev = v_dev.bounded(params.k_rem, CHUNK_COLS);
    }
    if constexpr (!USE_ACCELERATOR) {
      stage_tile<BK, BD, kv_leading_dimension, SIMDGROUPS_PER_THREADGROUP * 32>(
          v_block,
          value_source_stride,
          kv_shared,
          tail_k ? short(params.k_rem) : short(BK),
          thread_context
      );
    }
    auto v_shared = tile_source(kv_shared, kv_leading_dimension);
    METAL_PRAGMA_UNROLL
    for (int c = 0; c < N_OUT_CHUNKS; ++c) {
      Fragment<InputType, KEY_GRID_COLS, VCHUNK, Ops> value_chunk;
      if constexpr (USE_ACCELERATOR) {
        value_chunk.load_from(thread_context.simd_lane_id, v_dev.advanced(c * CHUNK_COLS));
      } else {
        value_chunk.load_from(thread_context.simd_lane_id, v_shared.advanced(c * CHUNK_COLS));
      }
      tile_matmul(output_chunks[c], score_fragment, value_chunk);
    }
  }

  AccumType inv_sum[ROWS_PER_LANE];
  METAL_PRAGMA_UNROLL
  for (int r = 0; r < ROWS_PER_LANE; ++r) {
    inv_sum[r] = AccumType(1) / sum_score[r];
  }
  METAL_PRAGMA_UNROLL
  for (int c = 0; c < N_OUT_CHUNKS; ++c) {
    output_chunks[c].row_bin_op(inv_sum, [](AccumType v, AccumType s) { return v * s; });
  }

  o += int64_t(simdgroup_row_base) * params.o_strides[2];

  if (ragged_q && params.q_rem <= int(simdgroup_row_base)) {
    return;
  }
  METAL_PRAGMA_UNROLL
  for (int c = 0; c < N_OUT_CHUNKS; ++c) {
    device T* o_chunk = o + c * CHUNK_COLS;
    if (ragged_q) {
      const short2 dst_tile_dims = short2(CHUNK_COLS, params.q_rem - simdgroup_row_base);
      output_chunks[c].store_safe(thread_context.simd_lane_id, o_chunk, int(params.o_strides[2]), dst_tile_dims);
    } else {
      output_chunks[c].store(thread_context.simd_lane_id, o_chunk, int(params.o_strides[2]));
    }
  }
}
