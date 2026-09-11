#include <metal_stdlib>
#include "../../common/dsl.h"
#include "../../common/integral_constant.h"
#include "../../common/thread_context.h"
#include "../../matmul/common/fragment.h"
#include "../../matmul/common/loader.h"
#include "../../matmul/common/mxu_fragment/ops.h"
#include "../../generated/attention.h"
#include "../../generated/trie.h"

using namespace metal;
using namespace uzu::matmul;
using namespace uzu::attention;
using namespace uzu::trie;

#define ATTENTION_GQA_SIMDGROUPS 8

template <typename T, uint BK, uint BD, uint D_SLICES>
struct AttentionGemmGroupedLayout {
  using Ops = MxuFragmentOps<>;

  UZU_CONST int SIMDGROUPS = ATTENTION_GQA_SIMDGROUPS;
  UZU_CONST int TILE_ROWS = Ops::FRAGMENT_ROWS;
  UZU_CONST int SLICE_COLS = int(BD) / int(D_SLICES);
  UZU_CONST int BANDS = SIMDGROUPS / int(D_SLICES);
  UZU_CONST int BLOCK_ROWS = BANDS * TILE_ROWS;
  UZU_CONST int SCORE_TILE_ELEMENTS = TILE_ROWS * int(BK);
  UZU_CONST int SCORE_EXCHANGE_SIZE = SIMDGROUPS * SCORE_TILE_ELEMENTS;
};

template <class FragmentType, class Fn>
METAL_FUNC void map_lane_rows(thread FragmentType& tile, const short2 position, Fn fn) {
  using Ops = typename FragmentType::FragmentOpsType;
  using Element = typename FragmentType::ElementType;
  static_assert(FragmentType::ROW_FRAGMENTS == 1, "requires one row fragment");
  static_assert(
      Ops::THREAD_ELEMENT_ROWS * Ops::THREAD_ELEMENT_COLS == Ops::ELEMENTS_PER_THREAD,
      "requires row-major thread elements"
  );

  thread Element* data = tile.elements();
  METAL_PRAGMA_UNROLL
  for (ushort fragment_col = 0; fragment_col < FragmentType::COL_FRAGMENTS; ++fragment_col) {
    const ushort fragment_base = fragment_col * Ops::ELEMENTS_PER_THREAD;
    const short col_base = position.x + short(fragment_col) * short(Ops::FRAGMENT_COLS);
    METAL_PRAGMA_UNROLL
    for (ushort row = 0; row < Ops::THREAD_ELEMENT_ROWS; ++row) {
      METAL_PRAGMA_UNROLL
      for (ushort col = 0; col < Ops::THREAD_ELEMENT_COLS; ++col) {
        const ushort element = fragment_base + row * Ops::THREAD_ELEMENT_COLS + col;
        data[element] = Element(fn(row, col_base + short(col), data[element]));
      }
    }
  }
}

// TODO: support more BD and should tune more to replace AttentionGEMM
template <typename T, uint BK, uint BD, uint D_SLICES>
VARIANTS(T, bfloat)
VARIANTS(BK, 32)
VARIANTS(BD, 128, 256)
VARIANTS(D_SLICES, 2, 4)
CONSTRAINT(D_SLICES == BD / 64)
KERNEL(AttentionGemmGrouped)(
    const device T* q,
    const device T* k,
    const device T* v,
    device T* o OPTIONAL(!split_kv),
    device float* partials OPTIONAL(split_kv),
    device float* partial_maxs OPTIONAL(split_kv),
    device float* partial_sums OPTIONAL(split_kv),
    const device TrieNode* trie OPTIONAL(is_trie),
    const constant AttnParams& params,
    const constant uint& m_tiles,
    const constant uint& kv_heads,
    const constant uint& batch_count,
    const constant uint& num_splits,
    const bool split_kv SPECIALIZE,
    const bool is_causal SPECIALIZE,
    const bool is_trie SPECIALIZE,
    threadgroup float score_exchange[AttentionGemmGroupedLayout<T, BK, BD, D_SLICES>::SCORE_EXCHANGE_SIZE],
    const ThreadContext thread_context,
    const uint tile_split_flat GROUPS(m_tiles * num_splits),
    const uint kv_head_idx GROUPS(kv_heads),
    const uint batch_idx GROUPS(batch_count),
    const uint lid THREADS(ATTENTION_GQA_SIMDGROUPS * METAL_SIMD_SIZE)
) {
  using Layout = AttentionGemmGroupedLayout<T, BK, BD, D_SLICES>;
  using Ops = typename Layout::Ops;
  using AccumType = float;

  constexpr int TILE_ROWS = Layout::TILE_ROWS;
  constexpr int SLICE_COLS = Layout::SLICE_COLS;
  constexpr int BLOCK_ROWS = Layout::BLOCK_ROWS;
  constexpr int SCORE_TILE_ELEMENTS = Layout::SCORE_TILE_ELEMENTS;
  constexpr int KEY_COL_FRAGMENTS = int(BK) / Ops::FRAGMENT_COLS;
  constexpr int HEAD_DIM_FRAGMENTS_PER_SLICE = SLICE_COLS / Ops::FRAGMENT_ROWS;
  constexpr int VALUE_COL_FRAGMENTS = SLICE_COLS / Ops::FRAGMENT_COLS;
  constexpr int ROWS_PER_LANE = Ops::THREAD_ELEMENT_ROWS;

  static_assert(Layout::SIMDGROUPS == 8, "requires eight simdgroups");
  static_assert(KEY_COL_FRAGMENTS % 2 == 0, "QK fragments must be even");
  static_assert(VALUE_COL_FRAGMENTS % 2 == 0, "PV fragments must be even");

  using QueryFragment = OperandFragment<T, 1, 1, Ops>;
  using KeyFragment = OperandFragment<T, 1, KEY_COL_FRAGMENTS, Ops, ReadTranspose>;
  using ValueFragment = OperandFragment<T, KEY_COL_FRAGMENTS, VALUE_COL_FRAGMENTS, Ops>;
  using ScoreFragment = Fragment<AccumType, 1, KEY_COL_FRAGMENTS, Ops>;
  using OutputFragment = Fragment<AccumType, 1, VALUE_COL_FRAGMENTS, Ops>;

  const ushort lane = ushort(thread_context.simd_lane_id);
  const ushort simdgroup_index = ushort(thread_context.simdgroup_index);
  const ushort band = simdgroup_index / ushort(D_SLICES);
  const ushort slice = simdgroup_index % ushort(D_SLICES);

  const uint m_tile_idx = split_kv ? (tile_split_flat % m_tiles) : tile_split_flat;
  const uint split_idx = split_kv ? (tile_split_flat / m_tiles) : 0u;

  const uint grouped_rows = params.gqa_factor * params.q_len;
  const uint row_base = m_tile_idx * uint(BLOCK_ROWS) + uint(band) * uint(TILE_ROWS);
  const uint head_in_group = row_base / params.q_len;
  const uint query_base = row_base - head_in_group * params.q_len;
  const uint q_head_idx = kv_head_idx * params.gqa_factor + head_in_group;

  const uint query_row_stride = params.q_strides[2];
  const uint key_row_stride = params.k_strides[2];
  const uint value_row_stride = params.v_strides[2];
  const uint output_row_stride = params.o_strides[2];
  const uint slice_offset = uint(slice) * uint(SLICE_COLS);

  q += size_t(batch_idx) * params.q_strides[0] + size_t(q_head_idx) * params.q_strides[1] +
       size_t(query_base) * query_row_stride + slice_offset;
  k += size_t(batch_idx) * params.k_strides[0] + size_t(kv_head_idx) * params.k_strides[1] + slice_offset;
  v += size_t(batch_idx) * params.v_strides[0] + size_t(kv_head_idx) * params.v_strides[1] + slice_offset;

  QueryFragment query_frags[HEAD_DIM_FRAGMENTS_PER_SLICE];
  {
    auto query_source = fragment_source(q, int(query_row_stride));
    const uint valid_rows = row_base < grouped_rows ? grouped_rows - row_base : 0u;
    if (valid_rows < uint(TILE_ROWS)) {
      query_source = query_source.bounded(short(valid_rows), short(Ops::FRAGMENT_COLS));
    }
    METAL_PRAGMA_UNROLL
    for (short head_dim_fragment_idx = 0; head_dim_fragment_idx < HEAD_DIM_FRAGMENTS_PER_SLICE;
         ++head_dim_fragment_idx) {
      query_frags[head_dim_fragment_idx].load_from(
          lane,
          query_source.advanced(head_dim_fragment_idx * Ops::FRAGMENT_ROWS)
      );
    }
  }

  OutputFragment output;
  output.clear();

  AccumType max_score[ROWS_PER_LANE];
  AccumType sum_score[ROWS_PER_LANE];
  METAL_PRAGMA_UNROLL
  for (int r = 0; r < ROWS_PER_LANE; ++r) {
    max_score[r] = -INFINITY;
    sum_score[r] = AccumType(0);
  }

  const uint prefix_length = params.q_off;
  const uint key_length = params.k_len;
  const short2 score_position = ScoreFragment::get_position(lane);
  uint last_visible_key[ROWS_PER_LANE];
  ulong row_trie_mask[ROWS_PER_LANE];
  METAL_PRAGMA_UNROLL
  for (ushort r = 0; r < ROWS_PER_LANE; ++r) {
    last_visible_key[r] = 0;
    row_trie_mask[r] = 0ul;
  }
  if (is_causal || is_trie) {
    METAL_PRAGMA_UNROLL
    for (ushort r = 0; r < ROWS_PER_LANE; ++r) {
      const uint row = row_base + uint(score_position.y) + uint(r) * uint(Ops::THREAD_ELEMENT_ROW_STRIDE);
      last_visible_key[r] = prefix_length + row % params.q_len;
    }
    if (is_trie) {
      const ulong all_suffix_keys = (params.q_len >= 64u) ? ~0ul : ((1ul << params.q_len) - 1ul);
      METAL_PRAGMA_UNROLL
      for (ushort r = 0; r < ROWS_PER_LANE; ++r) {
        row_trie_mask[r] = is_causal ? 0ul : all_suffix_keys;
      }
      if (is_causal) {
        for (uint node_index = 0; node_index < params.q_len; ++node_index) {
          const TrieNode node = trie[node_index];
          METAL_PRAGMA_UNROLL
          for (ushort r = 0; r < ROWS_PER_LANE; ++r) {
            const uint query_index = last_visible_key[r] - prefix_length;
            if (query_index >= node.trie_start && query_index <= node.trie_end) {
              row_trie_mask[r] |= (1ul << node_index);
            }
          }
        }
      }
    }
  }

  threadgroup float* exchange_slot = score_exchange + simdgroup_index * SCORE_TILE_ELEMENTS;
  const ushort band_exchange_base = band * ushort(D_SLICES);
  const uint key_block_stride = uint(BK) * key_row_stride;
  const uint value_block_stride = uint(BK) * value_row_stride;

  const uint block_begin = split_kv ? (split_idx * params.nk) / num_splits : 0u;
  uint block_end = split_kv ? ((split_idx + 1u) * params.nk) / num_splits : params.nk;

  if (is_causal) {
    const uint tile_first_row = m_tile_idx * uint(BLOCK_ROWS);
    const uint tile_last_row = min(tile_first_row + uint(BLOCK_ROWS) - 1u, grouped_rows - 1u);
    const uint max_query_index = (tile_first_row / params.q_len == tile_last_row / params.q_len)
                                     ? (tile_last_row % params.q_len)
                                     : (params.q_len - 1u);
    const uint causal_blocks = (uint(prefix_length) + max_query_index + uint(BK)) / uint(BK);
    block_end = min(block_end, causal_blocks);
  }

  const uint tail_keys = params.k_rem;
  const uint block_count = block_end > block_begin ? block_end - block_begin : 0u;
  const bool owns_tail_block = (tail_keys != 0u) && (block_count > 0u) && (block_end == params.nk);
  const uint full_blocks = owns_tail_block ? block_count - 1u : block_count;

  k += size_t(block_begin) * key_block_stride;
  v += size_t(block_begin) * value_block_stride;

  const AccumType score_scale = AccumType(params.scale) * AccumType(M_LOG2E_F);
  // Masked scores are multiplied by score_scale below, so pre-divide this value.
  const AccumType masked_score_pre_scale = (static_cast<AccumType>(-1e9f) * M_LOG2E_F) / score_scale;
  const uint first_key = block_begin * uint(BK);

  auto accumulate_kv_block = [&](const uint kb, auto tail_flag) {
    constexpr bool IS_TAIL = decltype(tail_flag)::value;

    ScoreFragment score;
    score.clear();
    {
      auto key_source = fragment_source(k + size_t(kb) * key_block_stride, int(key_row_stride));
      if constexpr (IS_TAIL) {
        key_source = key_source.bounded(short(tail_keys), short(Ops::FRAGMENT_COLS));
      }
      METAL_PRAGMA_UNROLL
      for (short head_dim_fragment_idx = 0; head_dim_fragment_idx < HEAD_DIM_FRAGMENTS_PER_SLICE;
           ++head_dim_fragment_idx) {
        KeyFragment key_fragment;
        key_fragment.load_from(lane, key_source.advanced(head_dim_fragment_idx * Ops::FRAGMENT_ROWS));
        simdgroup_barrier(mem_flags::mem_none);
        fragment_mma(score, query_frags[head_dim_fragment_idx], key_fragment);
      }
    }

    if constexpr (D_SLICES > 1) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      score.store(lane, exchange_slot, int(BK));
      threadgroup_barrier(mem_flags::mem_threadgroup);

      thread AccumType* score_elements = score.elements();
      METAL_PRAGMA_UNROLL
      for (ushort j = 1; j < ushort(D_SLICES); ++j) {
        const ushort other = (slice + j) % ushort(D_SLICES);
        ScoreFragment partial;
        partial.load_from(
            lane,
            fragment_source(score_exchange + (band_exchange_base + other) * SCORE_TILE_ELEMENTS, int(BK))
        );
        thread AccumType* partial_elements = partial.elements();
        METAL_PRAGMA_UNROLL
        for (ushort e = 0; e < ScoreFragment::ELEMENTS_PER_FRAGMENT; ++e) {
          score_elements[e] += partial_elements[e];
        }
      }
    }

    const uint key_base = first_key + kb * uint(BK);
    const bool masked_block = IS_TAIL || ((is_causal || is_trie) && (key_base + uint(BK) > prefix_length));
    if (masked_block) {
      map_lane_rows(score, score_position, [&](ushort row, short col, AccumType value) {
        const uint key = key_base + uint(col);
        bool visible = key < key_length;
        if (is_trie) {
          visible =
              visible && (key < prefix_length || ((row_trie_mask[row] >> ulong(key - prefix_length)) & 1ul) != 0ul);
        } else if (is_causal) {
          visible = visible && (key <= last_visible_key[row]);
        }
        return visible ? value : masked_score_pre_scale;
      });
    }

    AccumType block_max[ROWS_PER_LANE];
    score.row_reduce(block_max, -INFINITY, [](AccumType a, AccumType b) { return metal::max(a, b); });
    METAL_PRAGMA_UNROLL
    for (int r = 0; r < ROWS_PER_LANE; ++r) {
      block_max[r] *= score_scale;
    }

    AccumType new_max[ROWS_PER_LANE];
    AccumType rescale[ROWS_PER_LANE];
    METAL_PRAGMA_UNROLL
    for (int r = 0; r < ROWS_PER_LANE; ++r) {
      new_max[r] = metal::max(max_score[r], block_max[r]);
      rescale[r] = fast::exp2(max_score[r] - new_max[r]);
      max_score[r] = new_max[r];
      sum_score[r] *= rescale[r];
    }

    score.map_rows(new_max, [score_scale](AccumType value, AccumType m) {
      return fast::exp2(value * score_scale - m);
    });

    AccumType block_sum[ROWS_PER_LANE];
    score.row_reduce(block_sum, AccumType(0), [](AccumType a, AccumType b) { return a + b; });
    METAL_PRAGMA_UNROLL
    for (int r = 0; r < ROWS_PER_LANE; ++r) {
      sum_score[r] += block_sum[r];
    }

    output.map_rows(rescale, [](AccumType value, AccumType f) { return value * f; });

    {
      auto value_source = fragment_source(v + size_t(kb) * value_block_stride, int(value_row_stride));
      if constexpr (IS_TAIL) {
        value_source = value_source.bounded(short(tail_keys), short(SLICE_COLS));
      }
      ValueFragment value_fragment;
      value_fragment.load_from(lane, value_source);
      simdgroup_barrier(mem_flags::mem_none);
      fragment_mma(output, score, value_fragment);
    }
  };

  for (uint kb = 0; kb < full_blocks; ++kb) {
    accumulate_kv_block(kb, uzu::false_type{});
  }
  if (owns_tail_block) {
    accumulate_kv_block(full_blocks, uzu::true_type{});
  }

  if (split_kv) {
    const uint tile_index = ((batch_idx * kv_heads + kv_head_idx) * m_tiles + m_tile_idx) * num_splits + split_idx;
    device float* partial_tile = partials + size_t(tile_index) * BLOCK_ROWS * BD + size_t(band) * TILE_ROWS * BD;
    output.store(lane, partial_tile + slice_offset, int(BD));

    if (slice == 0) {
      const short2 position = OutputFragment::get_position(lane);
      if (position.x == 0) {
        const uint row_base_in_tile = uint(tile_index) * uint(BLOCK_ROWS) + uint(band) * uint(TILE_ROWS);
        METAL_PRAGMA_UNROLL
        for (int r = 0; r < ROWS_PER_LANE; ++r) {
          const uint row = row_base_in_tile + uint(position.y + r * Ops::THREAD_ELEMENT_ROW_STRIDE);
          partial_maxs[row] = max_score[r];
          partial_sums[row] = sum_score[r];
        }
      }
    }
  } else {
    AccumType inverse_sum[ROWS_PER_LANE];
    METAL_PRAGMA_UNROLL
    for (int r = 0; r < ROWS_PER_LANE; ++r) {
      inverse_sum[r] = AccumType(1) / sum_score[r];
    }
    output.map_rows(inverse_sum, [](AccumType value, AccumType s) { return value * s; });

    device T* output_base = o + size_t(batch_idx) * params.o_strides[0];
    if (params.q_len % uint(TILE_ROWS) == 0u) {
      if (row_base < grouped_rows) {
        device T* destination = output_base + size_t(q_head_idx) * params.o_strides[1] +
                                size_t(query_base) * output_row_stride + slice_offset;
        output.store(lane, destination, int(output_row_stride));
      }
    } else {
      const short2 position = OutputFragment::get_position(lane);
      thread AccumType* output_elements = output.elements();
      METAL_PRAGMA_UNROLL
      for (ushort r = 0; r < ROWS_PER_LANE; ++r) {
        const uint row = row_base + uint(position.y) + uint(r) * uint(Ops::THREAD_ELEMENT_ROW_STRIDE);
        if (row < grouped_rows) {
          const uint row_head = row / params.q_len;
          const uint row_position = row - row_head * params.q_len;
          device T* destination = output_base +
                                  (size_t(kv_head_idx) * params.gqa_factor + row_head) * params.o_strides[1] +
                                  size_t(row_position) * output_row_stride + slice_offset + position.x;
          METAL_PRAGMA_UNROLL
          for (ushort fragment_col = 0; fragment_col < VALUE_COL_FRAGMENTS; ++fragment_col) {
            METAL_PRAGMA_UNROLL
            for (ushort col = 0; col < Ops::THREAD_ELEMENT_COLS; ++col) {
              const ushort element = fragment_col * Ops::ELEMENTS_PER_THREAD + r * Ops::THREAD_ELEMENT_COLS + col;
              destination[fragment_col * Ops::FRAGMENT_COLS + col] = static_cast<T>(output_elements[element]);
            }
          }
        }
      }
    }
  }
}

template <typename T, uint BD>
VARIANTS(T, bfloat)
VARIANTS(BD, 128, 256)
KERNEL(AttentionGemmGroupedCombine)(
    const device float* partials,
    const device float* partial_maxs,
    const device float* partial_sums,
    device T* o,
    const constant AttnParams& params,
    const constant uint& grouped_rows,
    const constant uint& block_rows,
    const constant uint& m_tiles,
    const constant uint& kv_heads,
    const constant uint& batch_count,
    const constant uint& num_splits,
    const uint grouped_row_idx GROUPS(grouped_rows),
    const uint kv_head_idx GROUPS(kv_heads),
    const uint batch_idx GROUPS(batch_count),
    const uint column THREADS(BD)
) {
  const uint m_tile_idx = grouped_row_idx / block_rows;
  const uint row_in_tile = grouped_row_idx - m_tile_idx * block_rows;
  const uint tile_base = ((batch_idx * kv_heads + kv_head_idx) * m_tiles + m_tile_idx) * num_splits;

  const device float* row_maxs = partial_maxs + size_t(tile_base) * block_rows + row_in_tile;
  const device float* row_sums = partial_sums + size_t(tile_base) * block_rows + row_in_tile;
  const device float* row_partials = partials + size_t(tile_base) * block_rows * BD + size_t(row_in_tile) * BD + column;

  float global_max = -INFINITY;
  for (uint s = 0; s < num_splits; ++s) {
    if (row_sums[s * block_rows] > 0.0f) {
      global_max = metal::max(global_max, row_maxs[s * block_rows]);
    }
  }

  float numerator = 0.0f;
  float denominator = 0.0f;
  if (global_max > -INFINITY) {
    for (uint s = 0; s < num_splits; ++s) {
      const float split_sum = row_sums[s * block_rows];
      if (split_sum <= 0.0f) {
        continue;
      }
      const float rescale = fast::exp2(row_maxs[s * block_rows] - global_max);
      denominator += rescale * split_sum;
      numerator += rescale * row_partials[size_t(s) * block_rows * BD];
    }
  }

  const uint head_in_group = grouped_row_idx / params.q_len;
  const uint query_position = grouped_row_idx - head_in_group * params.q_len;
  const uint q_head_idx = kv_head_idx * params.gqa_factor + head_in_group;
  o += size_t(batch_idx) * params.o_strides[0] + size_t(q_head_idx) * params.o_strides[1] +
       size_t(query_position) * params.o_strides[2] + column;
  o[0] = static_cast<T>(denominator > 0.0f ? numerator / denominator : 0.0f);
}
