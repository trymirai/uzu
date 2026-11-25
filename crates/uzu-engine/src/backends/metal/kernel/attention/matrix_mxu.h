#pragma once

#include "../common/conversion.h"
#include "../common/thread_context.h"
#include "../generated/attention.h"
#include "../generated/trie.h"
#include "../matmul/common/fragment.h"
#include "kv_layout.h"
#include "mask.h"
#include "matrix_layout.h"
#include "rotated_output.h"

using namespace uzu::matmul;

template <class Ops, int DEPTH_FRAGMENTS>
METAL_FUNC void prepare_queries(
    const device int8_t* query_codes,
    threadgroup uint4* staged_query,
    uint query_row_base,
    uint query_row_step,
    uint num_query_rows,
    uint query_row_stride,
    uint lane,
    uint channel_slice
) {
  // Padding rows reread the last real row.
  const short2 query_position = Ops::get_position(lane);
  const uint query_row = query_row_base + uint(query_position.y) * query_row_step;
  const uint last_query_row = num_query_rows - 1u;
  const device int8_t* query_row0 =
      query_codes + size_t(min(query_row, last_query_row)) * query_row_stride + uint(query_position.x);
  const device int8_t* query_row1 =
      query_codes +
      size_t(min(query_row + uint(Ops::THREAD_ELEMENT_ROW_STRIDE) * query_row_step, last_query_row)) *
          query_row_stride +
      uint(query_position.x);
  constexpr int SLICE_PAIRS = DEPTH_FRAGMENTS / 2;
  METAL_PRAGMA_UNROLL
  for (int slice_pair = 0; slice_pair < SLICE_PAIRS; ++slice_pair) {
    const int query_fragment = (int(channel_slice) * SLICE_PAIRS + slice_pair) * 2;
    uint4 packed_queries;
    packed_queries.x = *reinterpret_cast<const device uint*>(query_row0 + query_fragment * int(Ops::FRAGMENT_ROWS));
    packed_queries.y = *reinterpret_cast<const device uint*>(query_row1 + query_fragment * int(Ops::FRAGMENT_ROWS));
    packed_queries.z =
        *reinterpret_cast<const device uint*>(query_row0 + (query_fragment + 1) * int(Ops::FRAGMENT_ROWS));
    packed_queries.w =
        *reinterpret_cast<const device uint*>(query_row1 + (query_fragment + 1) * int(Ops::FRAGMENT_ROWS));
    staged_query[(int(channel_slice) * SLICE_PAIRS + slice_pair) * METAL_SIMD_SIZE] = packed_queries;
  }
}

template <class OutputFragment>
struct AttentionMatrixSoftmax {
  using AccumType = typename OutputFragment::ElementType;
  UZU_CONST int ROWS_PER_LANE = OutputFragment::FragmentOpsType::THREAD_ELEMENT_ROWS;
  // Lazy rescale (FlashAttention-4): raise the running max only after an increase of 8 (2^8).
  UZU_CONST AccumType RESCALE_THRESHOLD = 8.0f;

  OutputFragment output;
  AccumType running_max[ROWS_PER_LANE] = {
      metal::numeric_limits<AccumType>::lowest(),
      metal::numeric_limits<AccumType>::lowest()
  };
  AccumType denominator[ROWS_PER_LANE] = {};
  AccumType value_bias[ROWS_PER_LANE] = {};

  METAL_FUNC AttentionMatrixSoftmax() { output.clear(); }

  METAL_FUNC void rescale(const threadgroup float4* slot_stats, uint blocks_in_step, uint stat_rows) thread {
    AccumType step_max[ROWS_PER_LANE] = {running_max[0], running_max[1]};
    for (uint published_block = 0; published_block < blocks_in_step; ++published_block) {
      const float4 block_stats = slot_stats[published_block * stat_rows];
      step_max[0] = metal::max(step_max[0], block_stats.x);
      step_max[1] = metal::max(step_max[1], block_stats.z);
    }
    if (step_max[0] > running_max[0] + RESCALE_THRESHOLD || step_max[1] > running_max[1] + RESCALE_THRESHOLD) {
      AccumType rescale[ROWS_PER_LANE];
      METAL_PRAGMA_UNROLL
      for (int row = 0; row < ROWS_PER_LANE; ++row) {
        const AccumType next_max =
            step_max[row] > running_max[row] + RESCALE_THRESHOLD ? step_max[row] : running_max[row];
        rescale[row] = fast::exp2(running_max[row] - next_max);
        running_max[row] = next_max;
        denominator[row] *= rescale[row];
        value_bias[row] *= rescale[row];
      }
      output.map_rows(rescale, [](AccumType value, AccumType factor) UZU_ALWAYS_INLINE { return value * factor; });
    }
  }
};

// Trie rows that see a key: [first, last], packed in one uint (ushort2 costs registers).
struct RowRange {
  uint packed;

  METAL_FUNC RowRange() thread = default;
  METAL_FUNC RowRange(uint first, uint last) thread : packed(first | last << 16) {}
  METAL_FUNC static RowRange all() { return RowRange(0u, metal::numeric_limits<ushort>::max()); }
  METAL_FUNC bool contains(uint row) const thread { return row >= (packed & 0xFFFFu) && row <= packed >> 16; }
};

// Ring blocks first (wrapping), then suffix blocks.
struct VisibleKeyBlocks {
  uint ring_blocks;
  uint first_ring_block;
  uint visible_ring_blocks;
  uint first_suffix_block;

  METAL_FUNC static VisibleKeyBlocks make(uint oldest_visible, uint prefix_length, uint ring_offset, bool unrotated) {
    constexpr uint BLOCK_KEYS = uzu::kv_layout::TILE_KEYS;
    const uint ring_blocks = prefix_length / BLOCK_KEYS;
    const uint visible_ring_keys = oldest_visible < prefix_length ? prefix_length - oldest_visible : 0u;
    const uint shifted = oldest_visible + ring_offset;
    const uint oldest_slot = shifted >= prefix_length ? shifted - prefix_length : shifted;
    const uint oldest_block = oldest_slot / BLOCK_KEYS;
    const bool oldest_in_partial_block = oldest_block >= ring_blocks;
    const uint first_ring_block = oldest_in_partial_block ? 0u : oldest_block;
    const uint visible_ring_blocks =
        unrotated || visible_ring_keys == 0u
            ? 0u
            : min(ring_blocks, div_ceil(oldest_slot % BLOCK_KEYS + visible_ring_keys, BLOCK_KEYS));
    return {
        .ring_blocks = ring_blocks,
        .first_ring_block = first_ring_block,
        .visible_ring_blocks = visible_ring_blocks,
        .first_suffix_block = unrotated ? 0u : max(oldest_visible, prefix_length) / BLOCK_KEYS
    };
  }

  METAL_FUNC uint count_until(uint end_block) const thread {
    return visible_ring_blocks + end_block - first_suffix_block;
  }

  METAL_FUNC uint block_at(uint i) const thread {
    const uint ring_block = first_ring_block + i;
    return i < visible_ring_blocks ? (ring_block >= ring_blocks ? ring_block - ring_blocks : ring_block)
                                   : first_suffix_block + i - visible_ring_blocks;
  }
};

template <typename ScaleType, uint HEAD_DIM, uint NUM_SIMDGROUPS, bool IS_SPARSE>
METAL_FUNC void attention_matrix_mxu(
    const device int8_t* query_codes,
    const device ScaleType* query_scales,
    const device int8_t* key_codes,
    const device ScaleType* key_scales,
    const device uchar* value_codes,
    const device ScaleType* value_scales,
    const device uzu::trie::TrieNode* trie,
    const constant uzu::ring::RingParams& ring_params,
    const constant uint& sliding_window_size,
    bool is_causal,
    bool is_kv_cache_ring,
    bool is_sliding_window,
    device bfloat* partial_values,
    device float* partial_maxima,
    device float* partial_denominators,
    device bfloat* output,
    const device bfloat* gate,
    uint gate_row_stride,
    const device bfloat* sinks,
    const constant uzu::attention::AttnParams& params,
    uint query_tiles,
    uint num_splits,
    uint split_base,
    uint trie_nodes,
    bool mask_free,
    bool is_trie,
    bool is_split,
    bool has_gate,
    bool has_sinks,
    const device int* selected_blocks,
    uint blocks_per_query,
    uint compression_ratio,
    uint num_kv_heads,
    threadgroup float* probability_exchange,
    threadgroup uint2* row_windows,
    threadgroup uint4* tile_scratch,
    ThreadContext thread_context,
    uint tile_split_index,
    uint kv_head_index
) {
  using Layout = AttentionMatrixLayout<
      uzu::kv_layout::TILE_KEYS,
      HEAD_DIM,
      HEAD_DIM / uzu::kv_layout::SLICE_DIMS,
      true,
      NUM_SIMDGROUPS>;
  using Scratch = AttentionMatrixScratch<HEAD_DIM, true, NUM_SIMDGROUPS>;
  using Ops = typename Layout::Ops;
  using AccumType = float;

  constexpr uint DEPTH_SLICES = HEAD_DIM / uzu::kv_layout::SLICE_DIMS;

  constexpr int KEY_FRAGMENTS = int(uzu::kv_layout::TILE_KEYS) / Ops::FRAGMENT_COLS;
  constexpr int DEPTH_FRAGMENTS = Layout::SLICE_COLS / Ops::FRAGMENT_ROWS;
  constexpr int VALUE_FRAGMENTS = Layout::SLICE_COLS / Ops::FRAGMENT_COLS;
  constexpr int FRAGMENTS_PER_PANEL = 2;
  constexpr int VALUE_PANELS = VALUE_FRAGMENTS / FRAGMENTS_PER_PANEL;
  constexpr int ROWS_PER_LANE = Ops::THREAD_ELEMENT_ROWS;
  constexpr int KEYS_PER_LANE = KEY_FRAGMENTS * Ops::THREAD_ELEMENT_COLS;
  constexpr uint STAT_ROWS = uint(Layout::TILE_ROWS / ROWS_PER_LANE);

  static_assert(KEY_FRAGMENTS == 2, "probabilities pack two key fragments");
  static_assert(ROWS_PER_LANE == 2, "stats pack two rows per lane");

  // P is stored as u8: p_code = rint(255 * exp2(scores - block_max)), so p = p_code / 255.
  constexpr AccumType PROBABILITY_CODE_MAX = static_cast<AccumType>(metal::numeric_limits<uchar>::max());

  using ScoreFragment = Fragment<AccumType, 1, KEY_FRAGMENTS, Ops>;
  using ProbabilityFragment = Fragment<uchar, 1, KEY_FRAGMENTS, Ops>;

  const ushort lane = ushort(thread_context.simd_lane_id);
  const ushort query_band = ushort(thread_context.simdgroup_index) / ushort(DEPTH_SLICES);
  const ushort channel_slice = ushort(thread_context.simdgroup_index) % ushort(DEPTH_SLICES);

  const uint query_tile_index = tile_split_index % query_tiles;
  const uint split_index = split_base + tile_split_index / query_tiles;

  const uint query_row_step = IS_SPARSE ? params.q_len : 1u;
  const uint num_query_rows =
      IS_SPARSE ? (params.gqa_factor - 1u) * params.q_len + query_tile_index + 1u : params.gqa_factor * params.q_len;
  const uint query_row_base =
      IS_SPARSE ? query_tile_index
                : query_tile_index * uint(Layout::BLOCK_ROWS) + uint(query_band) * uint(Layout::TILE_ROWS);

  const uint channel_offset = uint(channel_slice) * uint(Layout::SLICE_COLS);

  const size_t query_group_offset = size_t(kv_head_index) * params.q_strides[1];
  query_codes += query_group_offset;

  AttentionMatrixSoftmax<Fragment<AccumType, 1, VALUE_FRAGMENTS, Ops>> softmax;

  const uint prefix_length = params.q_off;
  const uint key_length = params.k_len;
  const short2 score_position = ScoreFragment::get_position(lane);
  const uint lane_rows[ROWS_PER_LANE] = {
      uint(score_position.y) * query_row_step,
      (uint(score_position.y) + uint(Ops::THREAD_ELEMENT_ROW_STRIDE)) * query_row_step
  };
  const uint lane_key = uzu::kv_layout::column_key(uint(score_position.x));
  auto lane_column_key = [&](uint lane_col) UZU_ALWAYS_INLINE {
    return lane_key + uzu::kv_layout::column_key(
                          lane_col / Ops::THREAD_ELEMENT_COLS * Ops::FRAGMENT_COLS + lane_col % Ops::THREAD_ELEMENT_COLS
                      );
  };
  const uint query_band_offset = uint(query_band) * uint(Layout::TILE_ROWS);
  const uint ring_offset = is_kv_cache_ring ? ring_params.ring_offset : 0u;
  const uint window = is_sliding_window ? sliding_window_size : INVALID_POSITION;
  // Window 0 wraps: sees nothing.
  const uint behind = is_causal ? window - 1u : window / 2u;
  const uint ahead = is_causal ? 0u : window / 2u;
  const uint span = behind + ahead + 1u;
  if (!IS_SPARSE && !mask_free && score_position.x == 0) {
    METAL_PRAGMA_UNROLL
    for (ushort row = 0; row < ROWS_PER_LANE; ++row) {
      const uint query_index = (query_row_base + lane_rows[row]) % params.q_len;
      const uint position = is_trie && is_sliding_window ? trie[query_index].height : query_index;
      row_windows[query_band_offset + lane_rows[row]] = uint2(prefix_length - behind + position, query_index);
    }
  }

  const SparseKeys sparse_keys(
      selected_blocks,
      blocks_per_query,
      compression_ratio,
      trie,
      trie_nodes,
      prefix_length,
      query_tile_index,
      IS_SPARSE && is_trie
  );
  const uint tile_first_row = query_tile_index * uint(Layout::BLOCK_ROWS);
  const uint tile_last_row = min(tile_first_row + uint(Layout::BLOCK_ROWS) - 1u, num_query_rows - 1u);
  const bool one_head = tile_first_row / params.q_len == tile_last_row / params.q_len;
  const uint first_query = one_head && !is_trie ? tile_first_row % params.q_len : 0u;
  const uint last_query = one_head ? tile_last_row % params.q_len : params.q_len - 1u;
  const uint oldest_visible = is_sliding_window ? max(prefix_length + first_query, behind) - behind : 0u;
  auto key_position = [&](uint key) UZU_ALWAYS_INLINE {
    if (key >= key_length) {
      return INVALID_POSITION;
    }
    if (key >= prefix_length) {
      return key;
    }
    return key >= ring_offset ? key - ring_offset : key + prefix_length - ring_offset;
  };
  constexpr uint BLOCK_KEYS = uzu::kv_layout::TILE_KEYS;
  const bool unrotated = IS_SPARSE || !(is_kv_cache_ring || is_sliding_window);
  const VisibleKeyBlocks visible = VisibleKeyBlocks::make(oldest_visible, prefix_length, ring_offset, unrotated);
  // This split's share of the visible blocks, stopped at the last query's causal limit.
  const uint total = IS_SPARSE ? div_ceil(sparse_keys.key_count(), BLOCK_KEYS) : visible.count_until(params.nk);
  const uint first_block = split_index * total / num_splits;
  uint end_block = (split_index + 1u) * total / num_splits;
  if (is_causal) {
    end_block = min(end_block, visible.count_until(div_ceil(prefix_length + last_query + 1u, BLOCK_KEYS)));
  }
  const uint block_count = end_block > first_block ? end_block - first_block : 0u;

  const AccumType log2_attention_scale = AccumType(params.scale) * AccumType(M_LOG2E_F);

  const uzu::kv_layout::KVShape shape{HEAD_DIM, num_kv_heads, uzu::kv_layout::KVLayout::MxuTiled};

  AccumType query_score_scales[ROWS_PER_LANE];
  METAL_PRAGMA_UNROLL
  for (int row = 0; row < ROWS_PER_LANE; ++row) {
    const uint query_row = min(query_row_base + lane_rows[row], num_query_rows - 1u);
    query_score_scales[row] = AccumType(query_scales[query_group_offset / HEAD_DIM + query_row]) * log2_attention_scale;
  }

  threadgroup uchar* code_slot = reinterpret_cast<threadgroup uchar*>(probability_exchange);
  threadgroup float* stat_slot = probability_exchange + Scratch::STATS_OFFSET;
  const ushort band_first_simdgroup = query_band * ushort(int(DEPTH_SLICES));

  threadgroup uchar* const lane_codes = code_slot + uint(band_first_simdgroup) * uint(Layout::SCORE_TILE_ELEMENTS) +
                                        uint(lane) * uzu::kv_layout::LANE_BYTES;
  threadgroup float4* const lane_stats = reinterpret_cast<threadgroup float4*>(stat_slot) +
                                         uint(band_first_simdgroup) * STAT_ROWS + uint(score_position.y);
  threadgroup float2* const lane_sums =
      reinterpret_cast<threadgroup float2*>(probability_exchange + Scratch::SUMS_OFFSET) +
      uint(band_first_simdgroup) * STAT_ROWS + uint(score_position.y);
  uint slot_offset = 0u;

  threadgroup uint4* const staged_query =
      tile_scratch + uint(query_band) * uint(Scratch::QUERY_VECTORS_PER_BAND) + uint(lane);
  threadgroup uint* const slot_lists = reinterpret_cast<threadgroup uint*>(row_windows);
  constexpr int LANE_KEYS = int(uzu::kv_layout::LANE_KEYS);
  constexpr int KEY_PAIRS = LANE_KEYS / int(uzu::kv_layout::PAIR_KEYS);
  const uint lane_first_key = uzu::kv_layout::lane_first_key(lane);
  const uint lane_channel = uzu::kv_layout::lane_channel(lane);
  const bool gather_codes = IS_SPARSE && is_trie;
  const uint first_key_offset = (uint)shape.k_code_index(lane_first_key, kv_head_index, lane_channel);
  uint first_value_offsets[KEY_PAIRS];
  METAL_PRAGMA_UNROLL
  for (int pair = 0; pair < KEY_PAIRS; ++pair) {
    first_value_offsets[pair] = (uint)shape.v_code_index(
        lane_first_key + uint(pair) * uzu::kv_layout::PAIR_KEYS,
        kv_head_index,
        channel_offset + lane_channel
    );
  }
  const uint head_scale_base = shape.scale_index(0u, kv_head_index);
  auto compute_probabilities = [&](uint block_index,
                                   uint channel_slice,
                                   threadgroup float4* slot_stats,
                                   threadgroup float2* slot_sums,
                                   threadgroup uchar* slot_codes,
                                   threadgroup uint* step_slots) UZU_ALWAYS_INLINE {
    const uint key_block = visible.block_at(first_block + block_index + uint(channel_slice));
    const uint key_base = key_block * uint(uzu::kv_layout::TILE_KEYS);

    uint lane_slot = uint(uzu::kv_layout::NO_KEY_SLOT);
    uint key_offset;
    if constexpr (IS_SPARSE) {
      lane_slot = sparse_keys.slot(key_base + uint(lane));
      step_slots[uint(channel_slice) * uint(uzu::kv_layout::TILE_KEYS) + uint(lane)] = lane_slot;
      key_offset = (uint)shape.k_code_index(
          slot_address(simd_shuffle(lane_slot, ushort(lane_first_key))),
          kv_head_index,
          lane_channel
      );
    } else {
      key_offset = first_key_offset + key_block * shape.block_stride();
    }

    ScoreFragment scores;

    // Q x K^T (int8).
    Fragment<int, 1, KEY_FRAGMENTS, Ops> score;
    score.clear();
    uint4 key_offsets(key_offset);
    if constexpr (IS_SPARSE) {
      const uint lane_address = slot_address(lane_slot);
      METAL_PRAGMA_UNROLL
      for (uint key = 0; key < uint(LANE_KEYS); ++key) {
        key_offsets[key] = (uint)shape.k_code_index(
            simd_shuffle(lane_address, ushort(lane_first_key + key)),
            kv_head_index,
            lane_channel
        );
      }
    }
    for (int depth_slice = 0; depth_slice < int(DEPTH_SLICES); ++depth_slice) {
      METAL_PRAGMA_UNROLL
      for (int slice_fragment = 0; slice_fragment < DEPTH_FRAGMENTS; slice_fragment += 2) {
        const int depth_fragment = depth_slice * DEPTH_FRAGMENTS + slice_fragment;
        using CodeVector = typename Ops::template ThreadVector<int8_t>;
        OperandFragment<int8_t, 1, 2, Ops> query_fragments;
        const uint4 packed_queries = staged_query[(depth_fragment / 2) * METAL_SIMD_SIZE];
        query_fragments.fragment_data[0] = as_type<CodeVector>(packed_queries.xy);
        query_fragments.fragment_data[1] = as_type<CodeVector>(packed_queries.zw);
        OperandFragment<int8_t, 2, 2, Ops, ReadTranspose> key_fragments;
        METAL_PRAGMA_UNROLL
        for (int pair_index = 0; pair_index < 2; ++pair_index) {
          const int fragment_offset = (depth_fragment + pair_index) * int(uzu::kv_layout::BLOCK_FRAGMENT_BYTES);
          const uint4 packed_keys = uzu::kv_layout::load_code_words(
              reinterpret_cast<const device uchar*>(key_codes) + fragment_offset,
              key_offsets,
              gather_codes
          );
          key_fragments.fragment_data[pair_index] = as_type<CodeVector>(uint2(packed_keys.x, packed_keys.y));
          key_fragments.fragment_data[2 + pair_index] = as_type<CodeVector>(uint2(packed_keys.z, packed_keys.w));
        }
        simdgroup_barrier(mem_flags::mem_none);
        fragment_mma(score, query_fragments, key_fragments);
      }
    }

    // Dequantize scores to log2 units.
    AccumType key_scale_values[KEYS_PER_LANE];
    AccumType log2_value_scales[KEYS_PER_LANE];
    AccumType inverse_value_scales[KEYS_PER_LANE];
    uint key_positions[KEYS_PER_LANE];
    {
      METAL_PRAGMA_UNROLL
      for (int fragment_col = 0; fragment_col < KEY_FRAGMENTS; ++fragment_col) {
        METAL_PRAGMA_UNROLL
        for (int column = 0; column < Ops::THREAD_ELEMENT_COLS; ++column) {
          const int lane_col = fragment_col * Ops::THREAD_ELEMENT_COLS + column;
          const int scale_col = fragment_col * Ops::FRAGMENT_COLS + column;
          uint scale_index;
          bool valid;
          if constexpr (IS_SPARSE) {
            const uint column_slot = simd_shuffle(lane_slot, ushort(lane_column_key(lane_col)));
            scale_index = shape.scale_index(min(column_slot, params.k_len - 1u), kv_head_index);
            valid = column_slot < params.k_len;
          } else {
            scale_index =
                head_scale_base + key_block * shape.scale_block_stride() + uint(score_position.x) + uint(scale_col);
            key_positions[lane_col] = key_position(key_base + lane_column_key(lane_col));
            valid = mask_free || key_positions[lane_col] != INVALID_POSITION;
          }
          // Zero so masked scores stay -inf.
          key_scale_values[lane_col] =
              metal::select(AccumType(0), AccumType(key_scales[scale_index]), !IS_SPARSE || valid);
          const float value_scale = float(value_scales[scale_index]);
          log2_value_scales[lane_col] = valid ? metal::precise::log2(value_scale) : (IS_SPARSE ? -INFINITY : 0.0f);
          inverse_value_scales[lane_col] = valid ? metal::precise::divide(1.0f, value_scale) : 0.0f;
        }
      }
    }
    scores.for_each_lane(
        [&](ushort row, ushort column, thread AccumType& value, thread int& integer_score) UZU_ALWAYS_INLINE {
          value = metal::fma(
              AccumType(integer_score) * key_scale_values[column],
              query_score_scales[row],
              log2_value_scales[column]
          );
        },
        score
    );

    if (!IS_SPARSE && !mask_free) {
      RowRange seeing_rows[KEYS_PER_LANE];
      METAL_PRAGMA_UNROLL
      for (int lane_col = 0; lane_col < KEYS_PER_LANE; ++lane_col) {
        const uint key = key_base + lane_column_key(lane_col);
        seeing_rows[lane_col] = RowRange::all();
        if (is_trie && key >= prefix_length && key < key_length) {
          const auto node = trie[key - prefix_length];
          seeing_rows[lane_col] = RowRange(node.trie_start, node.trie_end);
          if (is_sliding_window) {
            key_positions[lane_col] = prefix_length + node.height;
          }
        }
      }
      scores.for_each_lane([&](ushort row, ushort column, thread AccumType& value) UZU_ALWAYS_INLINE {
        const uint2 query = row_windows[query_band_offset + lane_rows[row]];
        const uint oldest_visible = query.x;
        const uint query_index = query.y;
        bool visible = key_positions[column] - oldest_visible < span;
        if (is_trie && is_causal) {
          visible = visible && seeing_rows[column].contains(query_index);
        }
        value = visible ? value : -metal::numeric_limits<AccumType>::infinity();
      });
    }

    // Quantize probabilities and reduce row statistics.
    AccumType block_max[ROWS_PER_LANE];
    AccumType probability_code_sum[ROWS_PER_LANE];
    ProbabilityFragment probability_codes;
    scores.row_reduce(
        block_max,
        metal::numeric_limits<AccumType>::lowest(),
        [](AccumType a, AccumType b) UZU_ALWAYS_INLINE { return metal::max(a, b); }
    );
    scores.for_each_lane(
        [&](ushort row, ushort column, thread AccumType& value, thread uchar& code) UZU_ALWAYS_INLINE {
          value = metal::rint(fast::exp2(value - (block_max[row] - metal::log2(PROBABILITY_CODE_MAX))));
          code = uzu::round_to_byte(value);
        },
        probability_codes
    );
    scores.row_reduce(probability_code_sum, AccumType(0), [](AccumType a, AccumType b) UZU_ALWAYS_INLINE {
      return a + b;
    });
    AccumType denominator_sum[ROWS_PER_LANE];
    scores.for_each_lane([&](ushort row, ushort column, thread AccumType& value)
                             UZU_ALWAYS_INLINE { value *= inverse_value_scales[column]; });
    scores.row_reduce(denominator_sum, AccumType(0), [](AccumType a, AccumType b) UZU_ALWAYS_INLINE { return a + b; });

    // Publish probabilities and row statistics.
    *reinterpret_cast<threadgroup uint4*>(slot_codes + uint(channel_slice) * uint(Layout::SCORE_TILE_ELEMENTS)) =
        uint4(as_type<uint2>(probability_codes.fragment_data[0]), as_type<uint2>(probability_codes.fragment_data[1]));
    if (score_position.x == 0) {
      slot_stats[uint(channel_slice) * STAT_ROWS] =
          float4(block_max[0], probability_code_sum[0], block_max[1], probability_code_sum[1]);
      slot_sums[uint(channel_slice) * STAT_ROWS] = float2(denominator_sum[0], denominator_sum[1]);
    }
  };

  auto accumulate_output = [&](uint block_index,
                               uint blocks_in_step,
                               threadgroup float4* slot_stats,
                               threadgroup float2* slot_sums,
                               threadgroup uchar* slot_codes,
                               threadgroup uint* step_slots) UZU_ALWAYS_INLINE {
    // P x V over the published blocks.
    for (uint published_block = 0; published_block < blocks_in_step; ++published_block) {
      const float4 block_stats = slot_stats[published_block * STAT_ROWS];
      const AccumType block_max[ROWS_PER_LANE] = {block_stats.x, block_stats.z};
      const AccumType probability_code_sum[ROWS_PER_LANE] = {block_stats.y, block_stats.w};
      const float2 block_sums = slot_sums[published_block * STAT_ROWS];
      const AccumType denominator_sum[ROWS_PER_LANE] = {block_sums.x, block_sums.y};
      AccumType block_weights[ROWS_PER_LANE];
      METAL_PRAGMA_UNROLL
      for (int row = 0; row < ROWS_PER_LANE; ++row) {
        block_weights[row] = fast::exp2(block_max[row] - softmax.running_max[row]) * (1.0f / PROBABILITY_CODE_MAX);
        softmax.denominator[row] += block_weights[row] * denominator_sum[row];
        softmax.value_bias[row] +=
            block_weights[row] * AccumType(uzu::kv_layout::V_CODE_BIAS) * probability_code_sum[row];
        softmax.value_bias[row] += block_weights[row] * uzu::FLOAT_OFFSET;
      }

      ProbabilityFragment probability_codes;
      const uint4 packed_codes =
          *reinterpret_cast<const threadgroup uint4*>(slot_codes + published_block * Layout::SCORE_TILE_ELEMENTS);
      probability_codes.fragment_data[0] = as_type<typename ProbabilityFragment::ThreadVectorType>(packed_codes.xy);
      probability_codes.fragment_data[1] = as_type<typename ProbabilityFragment::ThreadVectorType>(packed_codes.zw);

      uint value_offsets[LANE_KEYS] = {};
      if constexpr (IS_SPARSE) {
        METAL_PRAGMA_UNROLL
        for (uint key = 0; key < uint(LANE_KEYS); ++key) {
          value_offsets[key] = (uint)shape.v_code_index(
              slot_address(step_slots[published_block * uint(uzu::kv_layout::TILE_KEYS) + lane_first_key + key]),
              kv_head_index,
              channel_offset + lane_channel
          );
        }
      } else {
        METAL_PRAGMA_UNROLL
        for (int pair = 0; pair < KEY_PAIRS; ++pair) {
          value_offsets[uzu::kv_layout::PAIR_KEYS * pair] =
              first_value_offsets[pair] +
              visible.block_at(first_block + block_index + published_block) * shape.block_stride();
        }
      }
      thread AccumType* output_elements = softmax.output.elements();
      // Accumulate V panels directly into their output columns.
      using ValueVector = typename Ops::template ThreadVector<uchar>;
      METAL_PRAGMA_UNROLL
      for (int panel = 0; panel < VALUE_PANELS; ++panel) {
        OperandFragment<uchar, KEY_FRAGMENTS, FRAGMENTS_PER_PANEL, Ops> value_fragments;
        METAL_PRAGMA_UNROLL
        for (int key_fragment = 0; key_fragment < KEY_FRAGMENTS; ++key_fragment) {
          const device uchar* panel_values = value_codes + panel * uzu::kv_layout::BLOCK_FRAGMENT_BYTES;
          const uint first_offset = value_offsets[uzu::kv_layout::PAIR_KEYS * key_fragment];
          const uint second_offset = value_offsets[uzu::kv_layout::PAIR_KEYS * key_fragment + 1];
          const uint4 offsets(
              first_offset,
              second_offset,
              first_offset + uzu::kv_layout::TILE_LANE_BYTES,
              second_offset + uzu::kv_layout::TILE_LANE_BYTES
          );
          const uint4 packed_values = uzu::kv_layout::load_code_words(panel_values, offsets, gather_codes);
          value_fragments.fragment_data[key_fragment * FRAGMENTS_PER_PANEL] = as_type<ValueVector>(packed_values.xy);
          value_fragments.fragment_data[key_fragment * FRAGMENTS_PER_PANEL + 1] =
              as_type<ValueVector>(packed_values.zw);
        }
        simdgroup_barrier(mem_flags::mem_none);
        Fragment<int, 1, FRAGMENTS_PER_PANEL, Ops> product;
        product.clear();
        fragment_mma(product, probability_codes, value_fragments);
        thread int* product_elements = product.elements();
        METAL_PRAGMA_UNROLL
        for (ushort row = 0; row < Ops::THREAD_ELEMENT_ROWS; ++row) {
          const AccumType block_weight = block_weights[row];
          METAL_PRAGMA_UNROLL
          for (ushort value_fragment = 0; value_fragment < FRAGMENTS_PER_PANEL; ++value_fragment) {
            METAL_PRAGMA_UNROLL
            for (ushort column = 0; column < Ops::THREAD_ELEMENT_COLS; ++column) {
              const ushort panel_element_index =
                  value_fragment * Ops::ELEMENTS_PER_THREAD + row * Ops::THREAD_ELEMENT_COLS + column;
              const ushort output_element_index =
                  (panel * FRAGMENTS_PER_PANEL + value_fragment) * Ops::ELEMENTS_PER_THREAD +
                  row * Ops::THREAD_ELEMENT_COLS + column;
              const AccumType value = uzu::float_with_offset(product_elements[panel_element_index]);
              output_elements[output_element_index] =
                  metal::fma(block_weight, value, output_elements[output_element_index]);
            }
          }
        }
      }
    }
  };

  prepare_queries<Ops, DEPTH_FRAGMENTS>(
      query_codes,
      staged_query,
      query_row_base,
      query_row_step,
      num_query_rows,
      params.q_strides[2],
      lane,
      channel_slice
  );
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Per step, each depth-slice simdgroup scores one key block and publishes P; then all accumulate P x V.
  uint block_index = 0;
  while (block_index < block_count) {
    const uint blocks_in_step = min(uint(int(DEPTH_SLICES)), block_count - block_index);
    threadgroup float4* slot_stats = lane_stats + slot_offset / 4u;
    threadgroup float2* slot_sums = lane_sums + slot_offset / 2u;
    threadgroup uchar* slot_codes = lane_codes + 4u * slot_offset;
    threadgroup uint* step_slots =
        slot_lists + (slot_offset != 0u ? DEPTH_SLICES * uint(uzu::kv_layout::TILE_KEYS) : 0u);

    if (uint(channel_slice) < blocks_in_step) {
      compute_probabilities(block_index, channel_slice, slot_stats, slot_sums, slot_codes, step_slots);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    softmax.rescale(slot_stats, blocks_in_step, STAT_ROWS);

    accumulate_output(block_index, blocks_in_step, slot_stats, slot_sums, slot_codes, step_slots);

    // Alternate slots to avoid a second barrier.
    slot_offset = uint(Scratch::SLOT_FLOATS) - slot_offset;
    block_index += blocks_in_step;
  }

  softmax.output.map_rows(softmax.value_bias, [](float value, float bias) UZU_ALWAYS_INLINE { return value - bias; });

  const uint row_base = IS_SPARSE ? query_tile_index : query_tile_index * uint(Layout::BLOCK_ROWS) + query_band_offset;
  const uint output_row =
      ((split_index - split_base) * num_kv_heads + kv_head_index) * query_tiles * uint(Layout::BLOCK_ROWS) + row_base;
  if (IS_SPARSE || is_split) {
    softmax.output
        .store(lane, partial_values + size_t(output_row) * HEAD_DIM + channel_offset, int(query_row_step * HEAD_DIM));
    if (channel_slice == 0 && score_position.x == 0) {
      METAL_PRAGMA_UNROLL
      for (int row = 0; row < ROWS_PER_LANE; ++row) {
        partial_maxima[output_row + lane_rows[row]] = softmax.running_max[row];
        partial_denominators[output_row + lane_rows[row]] = softmax.denominator[row];
      }
    }
  } else {
    // Slices meet in threadgroup memory.
    threadgroup float2* tile_stats = reinterpret_cast<threadgroup float2*>(row_windows);
    threadgroup bfloat* tile_values = reinterpret_cast<threadgroup bfloat*>(tile_scratch);
    static_assert(
        Layout::BLOCK_ROWS * HEAD_DIM * sizeof(bfloat) <= Scratch::TILE_SCRATCH_VECTORS * sizeof(uint4),
        "the output tile must fit the query staging"
    );
    threadgroup_barrier(mem_flags::mem_threadgroup);
    softmax.output.store(lane, tile_values + query_band_offset * HEAD_DIM + channel_offset, HEAD_DIM);
    if (channel_slice == 0 && score_position.x == 0) {
      METAL_PRAGMA_UNROLL
      for (int row = 0; row < ROWS_PER_LANE; ++row) {
        tile_stats[query_band_offset + uint(score_position.y) + uint(row * Ops::THREAD_ELEMENT_ROW_STRIDE)] =
            float2(softmax.running_max[row], softmax.denominator[row]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr ushort VALUES_PER_LANE = HEAD_DIM / METAL_SIMD_SIZE;
    for (uint row = thread_context.simdgroup_index; row < uint(Layout::BLOCK_ROWS); row += NUM_SIMDGROUPS) {
      const uint grouped_row = row_base - query_band_offset + row;
      if (grouped_row >= num_query_rows) {
        continue;
      }
      const float2 stats = tile_stats[row];
      float rotated[VALUES_PER_LANE];
      METAL_PRAGMA_UNROLL
      for (ushort value_index = 0; value_index < VALUES_PER_LANE; ++value_index) {
        rotated[value_index] = float(tile_values[row * HEAD_DIM + value_index * METAL_SIMD_SIZE + lane]);
      }
      store_output_row<bfloat, HEAD_DIM>(
          lane,
          rotated,
          stats.x,
          stats.y,
          sinks,
          has_sinks,
          output,
          gate,
          params,
          kv_head_index,
          grouped_row,
          gate_row_stride,
          has_gate
      );
    }
  }
}
