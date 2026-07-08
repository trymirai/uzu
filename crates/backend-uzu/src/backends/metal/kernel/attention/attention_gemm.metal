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

template <typename T, uint BK, uint BD, bool USE_MXU>
struct AttentionGemmLayout {
  using Ops = metal::conditional_t<USE_MXU, MxuFragmentOps<>, SimdgroupFragmentOps>;

  METAL_CONST int FRAGMENT_ROWS = Ops::FRAGMENT_ROWS;
  METAL_CONST int SIMDGROUPS_PER_THREADGROUP = 4;
  METAL_CONST int BLOCK_QUERY_ROWS = SIMDGROUPS_PER_THREADGROUP * FRAGMENT_ROWS;
  METAL_CONST int ROW_ALIGNMENT_BYTES = 16;
  METAL_CONST int ROW_PADDING_ELEMENTS = ROW_ALIGNMENT_BYTES / int(sizeof(T));
  METAL_CONST int Q_SMEM_SIZE = USE_MXU ? 1 : (BLOCK_QUERY_ROWS * (int(BD) + ROW_PADDING_ELEMENTS));
  METAL_CONST int KV_SMEM_SIZE = USE_MXU ? 1 : (int(BK) * (int(BD) + ROW_PADDING_ELEMENTS));
};

template <typename T, uint BK, uint BD, bool USE_MXU>
VARIANTS(T, float, half, bfloat)
VARIANTS(BK, 16, 32)
VARIANTS(BD, 64, 128, 256)
VARIANTS(USE_MXU, false, true)
CONSTRAINT(!USE_MXU || BK == 32)
CONSTRAINT(!USE_MXU || T != "float")
CONSTRAINT(!USE_MXU || BD != 256)
KERNEL(AttentionGemm)(
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
    threadgroup T q_smem[AttentionGemmLayout<T, BK, BD, USE_MXU>::Q_SMEM_SIZE],
    threadgroup T kv_smem[AttentionGemmLayout<T, BK, BD, USE_MXU>::KV_SMEM_SIZE],
    const ThreadContext thread_context,
    const uint q_tile_idx GROUPS(suffix_length.div_ceil(if USE_MXU { 64 } else { 32 })),
    const uint head_idx GROUPS(num_heads),
    const uint batch_idx GROUPS(1),
    const uint lid THREADS(128)
) {
  using AccumType = float;
  using Layout = AttentionGemmLayout<T, BK, BD, USE_MXU>;
  using Ops = typename Layout::Ops;
  constexpr short FRAGMENT_ROWS = Layout::FRAGMENT_ROWS;
  constexpr int BLOCK_QUERY_ROWS = Layout::BLOCK_QUERY_ROWS;
  constexpr int ROW_PADDING_ELEMENTS = Layout::ROW_PADDING_ELEMENTS;
  constexpr int SIMDGROUPS_PER_THREADGROUP = Layout::SIMDGROUPS_PER_THREADGROUP;
  constexpr bool QUERY_IS_RESIDENT = USE_MXU;

  using InputType = metal::conditional_t<USE_MXU, T, AccumType>;

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

  constexpr int QUERY_ROW_FRAGMENTS = 1;
  constexpr int KEY_COL_FRAGMENTS = BK / FRAGMENT_ROWS;
  constexpr int HEAD_DIM_FRAGMENTS = BD / FRAGMENT_ROWS;

  static_assert(!USE_MXU || KEY_COL_FRAGMENTS % 2 == 0, "MXU QK needs even N (KEY_COL_FRAGMENTS)");

  constexpr int QUERY_FRAGMENT_COUNT = QUERY_IS_RESIDENT ? HEAD_DIM_FRAGMENTS : 1;
  using QueryFragment = OperandFragment<InputType, QUERY_ROW_FRAGMENTS, 1, Ops>;
  using KeyFragment = OperandFragment<InputType, 1, KEY_COL_FRAGMENTS, Ops, ReadTranspose>;
  using ScoreFragment = Fragment<AccumType, QUERY_ROW_FRAGMENTS, KEY_COL_FRAGMENTS, Ops>;
  QueryFragment query_frags[QUERY_FRAGMENT_COUNT];
  ScoreFragment score_fragment;
  constexpr int VALUE_COL_FRAGMENTS = USE_MXU ? 2 : HEAD_DIM_FRAGMENTS;
  static_assert(HEAD_DIM_FRAGMENTS % VALUE_COL_FRAGMENTS == 0, "head-dim must split into VALUE_COL_FRAGMENTS frags");
  static_assert(!USE_MXU || VALUE_COL_FRAGMENTS % 2 == 0, "MXU PV needs even N (VALUE_COL_FRAGMENTS)");
  constexpr int OUTPUT_CHUNKS = HEAD_DIM_FRAGMENTS / VALUE_COL_FRAGMENTS;
  constexpr short OUTPUT_CHUNK_COLS = VALUE_COL_FRAGMENTS * FRAGMENT_ROWS;
  using ValueFragment = OperandFragment<InputType, KEY_COL_FRAGMENTS, VALUE_COL_FRAGMENTS, Ops>;
  using OutputFragment = Fragment<AccumType, QUERY_ROW_FRAGMENTS, VALUE_COL_FRAGMENTS, Ops>;
  OutputFragment output_chunks[OUTPUT_CHUNKS];

  METAL_PRAGMA_UNROLL
  for (int c = 0; c < OUTPUT_CHUNKS; ++c) {
    output_chunks[c].clear();
  }

  const short simdgroup_row_base = FRAGMENT_ROWS * QUERY_ROW_FRAGMENTS * short(thread_context.simdgroup_index);
  constexpr short head_dim_fragment_stride = FRAGMENT_ROWS;

  const bool ragged_q = (!align_q && int(q_tile_idx) == params.nq_aligned);
  constexpr short query_leading_dimension = BD + ROW_PADDING_ELEMENTS;
  threadgroup T* query_shared = q_smem;
  const short query_shared_offset = simdgroup_row_base * query_leading_dimension;

  if constexpr (QUERY_IS_RESIDENT) {
    auto q_src = fragment_source(q + int64_t(simdgroup_row_base) * query_source_stride, query_source_stride);
    if (ragged_q) {
      q_src = q_src.bounded(params.q_rem - simdgroup_row_base, FRAGMENT_ROWS);
    }
    const InputType query_scale = static_cast<InputType>(params.scale * M_LOG2E_F);
    METAL_PRAGMA_UNROLL
    for (short dd = 0; dd < HEAD_DIM_FRAGMENTS; dd++) {
      query_frags[dd].load_from(thread_context.simd_lane_id, q_src.advanced(dd * head_dim_fragment_stride));
      query_frags[dd].map_coords(thread_context.simd_lane_id, [&](short, short, InputType v) {
        return v * query_scale;
      });
    }
  } else {
    using QueryLoader = ThreadgroupLoader<
        T,
        BLOCK_QUERY_ROWS,
        BD,
        query_leading_dimension,
        1,
        SIMDGROUPS_PER_THREADGROUP * METAL_SIMD_SIZE>;
    thread QueryLoader query_loader(q, query_source_stride, query_shared, thread_context);
    if (ragged_q) {
      query_loader.load_safe(short2(BD, params.q_rem));
    } else {
      query_loader.load_unsafe();
    }
    query_loader.apply_inplace_op(TransformScale<T>(static_cast<T>(params.scale * M_LOG2E_F)));
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const AccumType masked_score = static_cast<AccumType>(-1e9f) * M_LOG2E_F;
  constexpr int ROWS_PER_LANE = QUERY_ROW_FRAGMENTS * Ops::THREAD_ELEMENT_ROWS;
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
  using KVBlockSource = BlockSource<
      T,
      BK,
      BD,
      BD + ROW_PADDING_ELEMENTS,
      SIMDGROUPS_PER_THREADGROUP * METAL_SIMD_SIZE,
      typename Ops::BlockStorage>;
  threadgroup T* kv_shared = kv_smem;

  for (int kb = 0; kb < kb_lim; kb++) {
    const bool tail_k = (!align_k && kb == params.nk_aligned);
    const short valid_k_rows = tail_k ? short(params.k_rem) : short(BK);
    const device T* k_block = k + int64_t(kb) * int(BK) * key_source_stride;
    score_fragment.clear();

    KVBlockSource key_source{
        k_block,
        key_source_stride,
        kv_shared,
        valid_k_rows,
        thread_context,
    };
    key_source.prepare();
    auto q_shared_src = fragment_source(query_shared + query_shared_offset, query_leading_dimension);

    METAL_PRAGMA_UNROLL
    for (short dd = 0; dd < HEAD_DIM_FRAGMENTS; dd++) {
      if constexpr (!QUERY_IS_RESIDENT) {
        query_frags[0].load_from(thread_context.simd_lane_id, q_shared_src.advanced(dd * head_dim_fragment_stride));
      }
      const short q_idx = QUERY_IS_RESIDENT ? dd : 0;

      KeyFragment key_fragment;
      key_source.load(key_fragment, dd * head_dim_fragment_stride);
      simdgroup_barrier(mem_flags::mem_none);
      fragment_mma(score_fragment, query_frags[q_idx], key_fragment);
    }

    const int k_rem = params.k_rem;
    if (!is_trie && !is_kv_cache_ring && !is_sliding_window) {
      const bool diag = is_causal && ((int(kb) + 1) * int(BK) - 1 > prefix_length + q_base);
      if (tail_k || diag) {
        score_fragment.map_coords(thread_context.simd_lane_id, [&](short row, short col, AccumType v) {
          if (tail_k && int(col) >= k_rem) {
            return masked_score;
          }
          if (diag && (kb * int(BK) + int(col)) > prefix_length + q_base + int(row)) {
            return masked_score;
          }
          return v;
        });
      }
    } else {
      score_fragment.map_coords(thread_context.simd_lane_id, [&](short row, short col, AccumType v) {
        if (tail_k && int(col) >= k_rem) {
          return masked_score;
        }
        const int q_rel = q_base + int(row);
        if (q_rel >= int(params.q_len)) {
          return v;
        }
        const int key = kb * BK + int(col);
        if (key >= int(params.k_len)) {
          return v;
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
          return masked_score;
        }
        return v;
      });
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

    score_fragment.map_rows(new_max, [](AccumType v, AccumType m) { return fast::exp2(v - m); });

    AccumType block_sum[ROWS_PER_LANE];
    score_fragment.row_reduce(block_sum, AccumType(0), [](AccumType a, AccumType b) { return a + b; });
    METAL_PRAGMA_UNROLL
    for (int r = 0; r < ROWS_PER_LANE; ++r) {
      sum_score[r] += block_sum[r];
    }

    METAL_PRAGMA_UNROLL
    for (int c = 0; c < OUTPUT_CHUNKS; ++c) {
      output_chunks[c].map_rows(factor, [](AccumType v, AccumType f) { return v * f; });
    }

    const device T* v_block = v + int64_t(kb) * int(BK) * value_source_stride;
    KVBlockSource value_source{
        v_block,
        value_source_stride,
        kv_shared,
        valid_k_rows,
        thread_context,
    };
    value_source.prepare();
    METAL_PRAGMA_UNROLL
    for (int c = 0; c < OUTPUT_CHUNKS; ++c) {
      ValueFragment value_chunk;
      value_source.load(value_chunk, c * OUTPUT_CHUNK_COLS);
      fragment_mma(output_chunks[c], score_fragment, value_chunk);
    }
  }

  AccumType inv_sum[ROWS_PER_LANE];
  METAL_PRAGMA_UNROLL
  for (int r = 0; r < ROWS_PER_LANE; ++r) {
    inv_sum[r] = AccumType(1) / sum_score[r];
  }
  METAL_PRAGMA_UNROLL
  for (int c = 0; c < OUTPUT_CHUNKS; ++c) {
    output_chunks[c].map_rows(inv_sum, [](AccumType v, AccumType s) { return v * s; });
  }

  o += int64_t(simdgroup_row_base) * params.o_strides[2];

  if (ragged_q && params.q_rem <= int(simdgroup_row_base)) {
    return;
  }
  METAL_PRAGMA_UNROLL
  for (int c = 0; c < OUTPUT_CHUNKS; ++c) {
    device T* o_chunk = o + c * OUTPUT_CHUNK_COLS;
    if (ragged_q) {
      const short2 dst_tile_dims = short2(OUTPUT_CHUNK_COLS, params.q_rem - simdgroup_row_base);
      output_chunks[c].store_safe(thread_context.simd_lane_id, o_chunk, int(params.o_strides[2]), dst_tile_dims);
    } else {
      output_chunks[c].store(thread_context.simd_lane_id, o_chunk, int(params.o_strides[2]));
    }
  }
}
