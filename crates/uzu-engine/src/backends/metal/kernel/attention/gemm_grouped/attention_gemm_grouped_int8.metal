#include <metal_stdlib>
#include "../../common/dsl.h"
#include "../../common/conversion.h"
#include "../../common/thread_context.h"
#include "../../matmul/common/fragment.h"
#include "../../generated/attention.h"
#include "../../generated/trie.h"
#include "../attention_gate.h"
#include "../kv_layout.h"
#include "../../hadamard_transform/hadamard_transform.h"
#include "util.h"

using namespace metal;
using namespace uzu::matmul;

template <uint HEAD_DIM>
struct Int8Scratch {
  using Layout = AttentionGemmGroupedLayout<uzu::kv_layout::TILE_KEYS, HEAD_DIM, HEAD_DIM / uzu::kv_layout::SLICE_DIMS>;

  // Exchange offsets are in floats.
  UZU_CONST int STATS_OFFSET = Layout::SIMDGROUPS * Layout::SCORE_TILE_ELEMENTS / int(sizeof(float));
  UZU_CONST int SUMS_OFFSET = STATS_OFFSET + Layout::SIMDGROUPS * Layout::TILE_ROWS * 2;
  UZU_CONST int SLOT_FLOATS = SUMS_OFFSET + Layout::SIMDGROUPS * Layout::TILE_ROWS;
  UZU_CONST int TRIE_MASKS = Layout::BLOCK_ROWS;

  UZU_CONST int QUERY_VECTORS_PER_BAND = int(HEAD_DIM) / (2 * uzu::kv_layout::TILE_DIMS) * METAL_SIMD_SIZE;
  UZU_CONST int QUERY_VECTORS = Layout::BANDS * QUERY_VECTORS_PER_BAND;
};

template <typename T, bool MASK_FREE, uint HEAD_DIM>
VARIANTS(T, bfloat)
VARIANTS(MASK_FREE, false, true)
VARIANTS(HEAD_DIM, 256)
KERNEL(AttentionGemmGroupedInt8)(
    const device int8_t* q_codes,
    const device T* q_scales,
    const device int8_t* k_codes,
    const device T* k_scales,
    const device uchar* v_codes,
    const device T* v_scales,
    const device uzu::trie::TrieNode* trie OPTIONAL(is_trie),
    device bfloat* partials,
    device float* partial_maxs,
    device float* partial_sums,
    const constant uzu::attention::AttnParams& params,
    const constant uint& query_tiles,
    const constant uint& num_splits,
    const constant uint& split_base,
    const constant uint& split_count,
    const constant uint& trie_nodes,
    const bool is_trie SPECIALIZE,
    const uint32_t num_kv_heads SPECIALIZE,
    threadgroup float p_exchange[2 * Int8Scratch<HEAD_DIM>::SLOT_FLOATS],
    threadgroup ulong trie_masks[Int8Scratch<HEAD_DIM>::TRIE_MASKS],
    threadgroup uint4 query_stage[Int8Scratch<HEAD_DIM>::QUERY_VECTORS],
    const ThreadContext thread_context,
    const uint tile_split_flat GROUPS(query_tiles * split_count),
    const uint kv_head_idx GROUPS(num_kv_heads),
    // Keep Z indexing: folding it to zero increases register spills on M5.
    const uint batch_idx GROUPS(1),
    const uint lid THREADS(ATTENTION_GQA_SIMDGROUPS * METAL_SIMD_SIZE)
) {
  using Layout = AttentionGemmGroupedLayout<uzu::kv_layout::TILE_KEYS, HEAD_DIM, HEAD_DIM / uzu::kv_layout::SLICE_DIMS>;
  using Scratch = Int8Scratch<HEAD_DIM>;
  using Ops = typename Layout::Ops;
  using AccumType = float;

  constexpr uint D_SLICES = HEAD_DIM / uzu::kv_layout::SLICE_DIMS;

  constexpr int KEY_TILES = int(uzu::kv_layout::TILE_KEYS) / Ops::FRAGMENT_COLS;
  constexpr int DEPTH_TILES = Layout::SLICE_COLS / Ops::FRAGMENT_ROWS;
  constexpr int VALUE_TILES = Layout::SLICE_COLS / Ops::FRAGMENT_COLS;
  constexpr int PANEL_TILES = 2;
  constexpr int PANELS = VALUE_TILES / PANEL_TILES;
  constexpr int ROWS_PER_LANE = Ops::THREAD_ELEMENT_ROWS;
  constexpr int LANE_COLS = KEY_TILES * Ops::THREAD_ELEMENT_COLS;
  constexpr uint STAT_ROWS = uint(Layout::TILE_ROWS / ROWS_PER_LANE);

  static_assert(KEY_TILES == 2, "probabilities pack two key fragments");
  static_assert(ROWS_PER_LANE == 2, "stats pack two rows per lane");

  // Online softmax in log2 units. The running max starts finite so the first rescale is exp2(-huge) = 0, not NaN;
  // masked scores underflow to 0.
  constexpr AccumType INITIAL_MAX = -1e30f;
  constexpr AccumType MASKED_SCORE = -1e9f;
  // Rescale only when a block max exceeds the running max by more than this (block weights stay <= 2^8).
  constexpr AccumType RESCALE_THRESHOLD = 8.0f;

  // P is stored as u8: p_code = rint(255 * exp2(score - block_max)), so p = p_code / 255.
  constexpr AccumType CODE_MAX = 255.0f;

  using ScoreFragment = Fragment<AccumType, 1, KEY_TILES, Ops>;
  using ProbabilityFragment = Fragment<uchar, 1, KEY_TILES, Ops>;

  const ushort lane = ushort(thread_context.simd_lane_id);
  const ushort band = ushort(thread_context.simdgroup_index) / ushort(D_SLICES);
  const ushort slice = ushort(thread_context.simdgroup_index) % ushort(D_SLICES);

  const uint query_tile_idx = tile_split_flat % query_tiles;
  const uint split_idx = split_base + tile_split_flat / query_tiles;

  const uint grouped_rows = params.gqa_factor * params.q_len;
  const uint row_base = query_tile_idx * uint(Layout::BLOCK_ROWS) + uint(band) * uint(Layout::TILE_ROWS);

  const uint query_row_stride = params.q_strides[2];
  const uint key_row_stride = params.k_strides[2];
  const uint value_row_stride = params.v_strides[2];
  const uint slice_offset = uint(slice) * uint(Layout::SLICE_COLS);

  const size_t query_group_base = size_t(batch_idx) * params.q_strides[0] + size_t(kv_head_idx) * params.q_strides[1];
  q_codes += query_group_base + size_t(row_base) * query_row_stride;
  k_codes += size_t(batch_idx) * params.k_strides[0] +
             uint(uzu::kv_layout::TILE_KEYS) * size_t(kv_head_idx) * params.k_strides[1];
  k_codes += int(lane) * uzu::kv_layout::LANE_BYTES;
  const device uchar* v_tile_codes =
      v_codes +
      (size_t(batch_idx) * params.v_strides[0] +
       size_t(uzu::kv_layout::TILE_KEYS) * (size_t(kv_head_idx) * params.v_strides[1] + slice_offset)) +
      size_t(lane) * uzu::kv_layout::LANE_BYTES;

  // Padding rows reread the last real row.
  int query_offset_row0 = 0;
  int query_offset_row1 = 0;
  {
    const short2 query_position = Ops::get_position(lane);
    const uint last_query_row = grouped_rows - 1u;
    query_offset_row0 =
        (int(min(row_base + uint(query_position.y), last_query_row)) - int(row_base)) * int(query_row_stride) +
        int(query_position.x);
    query_offset_row1 =
        (int(min(row_base + uint(query_position.y) + uint(Ops::THREAD_ELEMENT_ROW_STRIDE), last_query_row)) -
         int(row_base)) *
            int(query_row_stride) +
        int(query_position.x);
  }

  Fragment<AccumType, 1, VALUE_TILES, Ops> output;
  output.clear();

  AccumType max_score[ROWS_PER_LANE] = {INITIAL_MAX, INITIAL_MAX};
  AccumType sum_score[ROWS_PER_LANE] = {0.0f, 0.0f};
  AccumType output_bias[ROWS_PER_LANE] = {0.0f, 0.0f};

  const uint prefix_length = params.q_off;
  const uint key_length = params.k_len;
  const short2 score_position = ScoreFragment::get_position(lane);
  const uint band_rows[ROWS_PER_LANE] = {
      uint(score_position.y),
      uint(score_position.y) + uint(Ops::THREAD_ELEMENT_ROW_STRIDE)
  };
  const uint block_band_base = uint(band) * uint(Layout::TILE_ROWS);
  uint last_visible_key[MASK_FREE ? 1 : ROWS_PER_LANE];
  if (!MASK_FREE) {
    METAL_PRAGMA_UNROLL
    for (ushort r = 0; r < ROWS_PER_LANE; ++r) {
      last_visible_key[r] = prefix_length + (row_base + band_rows[r]) % params.q_len;
    }
  }
  if (!MASK_FREE && is_trie) {
    ulong row_trie_mask[ROWS_PER_LANE] = {0ul, 0ul};
    for (uint node_index = 0; node_index < trie_nodes; ++node_index) {
      const uzu::trie::TrieNode node = trie[node_index];
      METAL_PRAGMA_UNROLL
      for (ushort r = 0; r < ROWS_PER_LANE; ++r) {
        const uint query_index = last_visible_key[r] - prefix_length;
        if (query_index >= node.trie_start && query_index <= node.trie_end) {
          row_trie_mask[r] |= (1ul << node_index);
        }
      }
    }
    if (score_position.x == 0) {
      METAL_PRAGMA_UNROLL
      for (ushort r = 0; r < ROWS_PER_LANE; ++r) {
        trie_masks[block_band_base + band_rows[r]] = row_trie_mask[r];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const uint key_block_stride = uint(uzu::kv_layout::TILE_KEYS) * key_row_stride;
  const uint value_block_stride = uint(uzu::kv_layout::TILE_KEYS) * value_row_stride;

  const uint block_begin = (split_idx * params.nk) / num_splits;
  const uint tile_first_row = query_tile_idx * uint(Layout::BLOCK_ROWS);
  const uint tile_last_row = min(tile_first_row + uint(Layout::BLOCK_ROWS) - 1u, grouped_rows - 1u);
  const uint max_query_index = (tile_first_row / params.q_len == tile_last_row / params.q_len)
                                   ? (tile_last_row % params.q_len)
                                   : (params.q_len - 1u);
  const uint block_end_candidate = ((split_idx + 1u) * params.nk) / num_splits;
  const uint block_end =
      min(block_end_candidate,
          (prefix_length + max_query_index + uint(uzu::kv_layout::TILE_KEYS)) / uint(uzu::kv_layout::TILE_KEYS));
  const uint tail_keys = params.k_rem;
  const uint block_count = block_end > block_begin ? block_end - block_begin : 0u;
  const bool owns_tail_block = (tail_keys != 0u) && (block_count > 0u) && (block_end == params.nk);

  k_codes += size_t(block_begin) * key_block_stride;
  v_tile_codes += size_t(block_begin) * value_block_stride;
  const uint first_key = block_begin * uint(uzu::kv_layout::TILE_KEYS);

  const AccumType score_scale = AccumType(params.scale) * AccumType(M_LOG2E_F);

  const uint scale_block_stride = uint(uzu::kv_layout::TILE_KEYS) * num_kv_heads;
  const size_t split_scale_offset =
      uzu::kv_layout::Int8KVShape{HEAD_DIM, num_kv_heads}.scale_index(first_key, kv_head_idx);
  const device T* key_scales_base = k_scales + split_scale_offset;
  const device T* value_scales_base = v_scales + split_scale_offset;

  AccumType query_row_scale[ROWS_PER_LANE];
  METAL_PRAGMA_UNROLL
  for (int r = 0; r < ROWS_PER_LANE; ++r) {
    const uint query_row = min(row_base + band_rows[r], grouped_rows - 1u);
    query_row_scale[r] = AccumType(q_scales[query_group_base / HEAD_DIM + query_row]) * score_scale;
  }

  threadgroup uchar* code_slot = reinterpret_cast<threadgroup uchar*>(p_exchange);
  threadgroup float* stat_slot = p_exchange + Scratch::STATS_OFFSET;
  const ushort band_first_simdgroup = band * ushort(int(D_SLICES));

  threadgroup uchar* const lane_codes = code_slot + uint(band_first_simdgroup) * uint(Layout::SCORE_TILE_ELEMENTS) +
                                        uint(lane) * uzu::kv_layout::LANE_BYTES;
  threadgroup float4* const lane_stats = reinterpret_cast<threadgroup float4*>(stat_slot) +
                                         uint(band_first_simdgroup) * STAT_ROWS + uint(score_position.y);
  threadgroup float2* const lane_sums = reinterpret_cast<threadgroup float2*>(p_exchange + Scratch::SUMS_OFFSET) +
                                        uint(band_first_simdgroup) * STAT_ROWS + uint(score_position.y);
  uint slot_offset = 0u;
  uint scale_offset = uint(slice) * scale_block_stride + uint(score_position.x);

  // Each slice simdgroup stages its depth slice of the band's Q.
  threadgroup uint4* const staged_query = query_stage + uint(band) * uint(Scratch::QUERY_VECTORS_PER_BAND) + uint(lane);
  {
    constexpr int SLICE_PAIRS = DEPTH_TILES / 2;
    METAL_PRAGMA_UNROLL
    for (int slice_pair = 0; slice_pair < SLICE_PAIRS; ++slice_pair) {
      const int query_fragment = (int(slice) * SLICE_PAIRS + slice_pair) * 2;
      uint4 query_pair;
      query_pair.x = *reinterpret_cast<const device uint*>(
          q_codes + (query_offset_row0 + query_fragment * int(Ops::FRAGMENT_ROWS))
      );
      query_pair.y = *reinterpret_cast<const device uint*>(
          q_codes + (query_offset_row1 + query_fragment * int(Ops::FRAGMENT_ROWS))
      );
      query_pair.z = *reinterpret_cast<const device uint*>(
          q_codes + (query_offset_row0 + (query_fragment + 1) * int(Ops::FRAGMENT_ROWS))
      );
      query_pair.w = *reinterpret_cast<const device uint*>(
          q_codes + (query_offset_row1 + (query_fragment + 1) * int(Ops::FRAGMENT_ROWS))
      );
      staged_query[(int(slice) * SLICE_PAIRS + slice_pair) * METAL_SIMD_SIZE] = query_pair;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Per step, each slice simdgroup scores one key block and publishes P; then all accumulate P x V.
  uint kb = 0;
  while (kb < block_count) {
    const uint step_blocks = min(uint(int(D_SLICES)), block_count - kb);
    const bool publishes_block = uint(slice) < step_blocks;
    threadgroup float4* slot_stats = lane_stats + slot_offset / 4u;
    threadgroup float2* slot_sums = lane_sums + slot_offset / 2u;
    threadgroup uchar* slot_codes = lane_codes + 4u * slot_offset;

    if (publishes_block) {
      const uint own_block = kb + uint(slice);
      const bool is_tail_block = owns_tail_block && (own_block + 1u == block_count);
      const uint key_base = first_key + own_block * uint(uzu::kv_layout::TILE_KEYS);
      const device int8_t* key_tile_base = k_codes + own_block * key_block_stride;

      ScoreFragment score;

      // Q x K^T (int8).
      Fragment<int, 1, KEY_TILES, Ops> int_score;
      int_score.clear();
      for (int depth_slice = 0; depth_slice < int(D_SLICES); ++depth_slice) {
        METAL_PRAGMA_UNROLL
        for (int slice_fragment = 0; slice_fragment < DEPTH_TILES; slice_fragment += 2) {
          const int depth_fragment = depth_slice * DEPTH_TILES + slice_fragment;
          using CodeVector = typename Ops::template ThreadVector<int8_t>;
          OperandFragment<int8_t, 1, 2, Ops> query_fragments;
          const uint4 query_pair = staged_query[(depth_fragment / 2) * METAL_SIMD_SIZE];
          query_fragments.fragment_data[0] = as_type<CodeVector>(query_pair.xy);
          query_fragments.fragment_data[1] = as_type<CodeVector>(query_pair.zw);
          OperandFragment<int8_t, 2, 2, Ops, ReadTranspose> key_fragments;
          METAL_PRAGMA_UNROLL
          for (int pair_index = 0; pair_index < 2; ++pair_index) {
            const uint4 key_chunk = *reinterpret_cast<const device uint4*>(
                key_tile_base + (depth_fragment + pair_index) * int(uzu::kv_layout::FRAGMENT_BYTES)
            );
            key_fragments.fragment_data[pair_index] = as_type<CodeVector>(uint2(key_chunk.x, key_chunk.y));
            key_fragments.fragment_data[2 + pair_index] = as_type<CodeVector>(uint2(key_chunk.z, key_chunk.w));
          }
          simdgroup_barrier(mem_flags::mem_none);
          fragment_mma(int_score, query_fragments, key_fragments);
        }
      }

      // Dequantize scores to log2 units.
      AccumType key_scale[LANE_COLS];
      AccumType log2_value_scale[LANE_COLS];
      AccumType inverse_value_scale[LANE_COLS];
      {
        const device T* key_scale_lane = key_scales_base + scale_offset;
        const device T* value_scale_lane = value_scales_base + scale_offset;
        METAL_PRAGMA_UNROLL
        for (int fragment_col = 0; fragment_col < KEY_TILES; ++fragment_col) {
          METAL_PRAGMA_UNROLL
          for (int col = 0; col < Ops::THREAD_ELEMENT_COLS; ++col) {
            const int lane_col = fragment_col * Ops::THREAD_ELEMENT_COLS + col;
            const int scale_col = fragment_col * Ops::FRAGMENT_COLS + col;
            const bool valid = MASK_FREE || key_base + uint(score_position.x) + uint(scale_col) < key_length;
            key_scale[lane_col] = AccumType(key_scale_lane[scale_col]);
            const float v_scale = float(value_scale_lane[scale_col]);
            log2_value_scale[lane_col] = valid ? metal::precise::log2(v_scale) : 0.0f;
            inverse_value_scale[lane_col] = valid ? metal::precise::divide(1.0f, v_scale) : 0.0f;
          }
        }
      }
      score.for_each_lane(
          [&](ushort row, ushort col, thread AccumType& value, thread int& int_value) UZU_ALWAYS_INLINE {
            value = metal::fma(AccumType(int_value) * key_scale[col], query_row_scale[row], log2_value_scale[col]);
          },
          int_score
      );

      if (!MASK_FREE && (is_tail_block || (key_base + uint(uzu::kv_layout::TILE_KEYS) > prefix_length))) {
        ulong trie_row_masks[ROWS_PER_LANE] = {0ul, 0ul};
        if (is_trie) {
          METAL_PRAGMA_UNROLL
          for (ushort r = 0; r < ROWS_PER_LANE; ++r) {
            trie_row_masks[r] = trie_masks[block_band_base + band_rows[r]];
          }
        }
        map_lane_rows(score, score_position, [&](ushort lane_row, short col, AccumType value) UZU_ALWAYS_INLINE {
          const uint key = key_base + uint(col);
          bool visible = key < key_length;
          if (is_trie) {
            visible = visible &&
                      (key < prefix_length || ((trie_row_masks[lane_row] >> ulong(key - prefix_length)) & 1ul) != 0ul);
          } else {
            visible = visible && key <= last_visible_key[MASK_FREE ? 0 : lane_row];
          }
          return visible ? value : MASKED_SCORE;
        });
      }

      // Quantize probabilities and reduce row statistics.
      AccumType block_max[ROWS_PER_LANE];
      AccumType code_sum[ROWS_PER_LANE];
      ProbabilityFragment p_codes;
      score.row_reduce(block_max, -INFINITY, [](AccumType a, AccumType b) UZU_ALWAYS_INLINE {
        return metal::max(a, b);
      });
      score.for_each_lane(
          [&](ushort row, ushort col, thread AccumType& value, thread uchar& code) UZU_ALWAYS_INLINE {
            value = metal::rint(fast::exp2(value - (block_max[row] - metal::log2(CODE_MAX))));
            code = uzu::round_to_byte(value);
          },
          p_codes
      );
      score.row_reduce(code_sum, AccumType(0), [](AccumType a, AccumType b) UZU_ALWAYS_INLINE { return a + b; });
      AccumType p_sum[ROWS_PER_LANE];
      score.for_each_lane([&](ushort row, ushort col, thread AccumType& value)
                              UZU_ALWAYS_INLINE { value *= inverse_value_scale[col]; });
      score.row_reduce(p_sum, AccumType(0), [](AccumType a, AccumType b) UZU_ALWAYS_INLINE { return a + b; });

      // Publish probabilities and row statistics.
      *reinterpret_cast<threadgroup uint4*>(slot_codes + uint(slice) * uint(Layout::SCORE_TILE_ELEMENTS)) =
          uint4(as_type<uint2>(p_codes.fragment_data[0]), as_type<uint2>(p_codes.fragment_data[1]));
      if (score_position.x == 0) {
        slot_stats[uint(slice) * STAT_ROWS] = float4(block_max[0], code_sum[0], block_max[1], code_sum[1]);
        slot_sums[uint(slice) * STAT_ROWS] = float2(p_sum[0], p_sum[1]);
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Raise the running max.
    {
      AccumType new_max[ROWS_PER_LANE] = {max_score[0], max_score[1]};
      for (uint published_block = 0; published_block < step_blocks; ++published_block) {
        const float4 packed_stats = slot_stats[published_block * STAT_ROWS];
        new_max[0] = metal::max(new_max[0], packed_stats.x);
        new_max[1] = metal::max(new_max[1], packed_stats.z);
      }
      if (new_max[0] > max_score[0] + RESCALE_THRESHOLD || new_max[1] > max_score[1] + RESCALE_THRESHOLD) {
        AccumType rescale[ROWS_PER_LANE];
        METAL_PRAGMA_UNROLL
        for (int r = 0; r < ROWS_PER_LANE; ++r) {
          const AccumType raised_max = new_max[r] > max_score[r] + RESCALE_THRESHOLD ? new_max[r] : max_score[r];
          rescale[r] = fast::exp2(max_score[r] - raised_max);
          max_score[r] = raised_max;
          sum_score[r] *= rescale[r];
          output_bias[r] *= rescale[r];
        }
        output.map_rows(rescale, [](AccumType value, AccumType factor) UZU_ALWAYS_INLINE { return value * factor; });
      }
    }
    // P x V over the published blocks.
    for (uint published_block = 0; published_block < step_blocks; ++published_block) {
      const float4 packed_stats = slot_stats[published_block * STAT_ROWS];
      const AccumType block_max[ROWS_PER_LANE] = {packed_stats.x, packed_stats.z};
      const AccumType code_sum[ROWS_PER_LANE] = {packed_stats.y, packed_stats.w};
      const float2 packed_sums = slot_sums[published_block * STAT_ROWS];
      const AccumType p_sum[ROWS_PER_LANE] = {packed_sums.x, packed_sums.y};
      AccumType alpha[ROWS_PER_LANE];
      METAL_PRAGMA_UNROLL
      for (int r = 0; r < ROWS_PER_LANE; ++r) {
        alpha[r] = fast::exp2(block_max[r] - max_score[r]) * (1.0f / CODE_MAX);
        sum_score[r] += alpha[r] * p_sum[r];
        output_bias[r] += alpha[r] * AccumType(uzu::kv_layout::V_CODE_BIAS) * code_sum[r];
        output_bias[r] += alpha[r] * uzu::FLOAT_OFFSET;
      }

      ProbabilityFragment p_codes;
      const uint4 packed_codes =
          *reinterpret_cast<const threadgroup uint4*>(slot_codes + published_block * Layout::SCORE_TILE_ELEMENTS);
      p_codes.fragment_data[0] = as_type<typename ProbabilityFragment::ThreadVectorType>(packed_codes.xy);
      p_codes.fragment_data[1] = as_type<typename ProbabilityFragment::ThreadVectorType>(packed_codes.zw);

      const uint value_block_index = kb + published_block;
      thread AccumType* accumulated = output.elements();
      // Accumulate V panels directly into their output columns.
      using ValueVector = typename Ops::template ThreadVector<uchar>;
      const device uchar* value_block = v_tile_codes + value_block_index * value_block_stride;
      METAL_PRAGMA_UNROLL
      for (int panel = 0; panel < PANELS; ++panel) {
        OperandFragment<uchar, KEY_TILES, PANEL_TILES, Ops> value_panel;
        METAL_PRAGMA_UNROLL
        for (int key_fragment = 0; key_fragment < KEY_TILES; ++key_fragment) {
          const uint4 tile = *reinterpret_cast<const device uint4*>(
              value_block + (key_fragment * PANELS + panel) * uzu::kv_layout::FRAGMENT_BYTES
          );
          value_panel.fragment_data[key_fragment * PANEL_TILES] = as_type<ValueVector>(tile.xy);
          value_panel.fragment_data[key_fragment * PANEL_TILES + 1] = as_type<ValueVector>(tile.zw);
        }
        simdgroup_barrier(mem_flags::mem_none);
        Fragment<int, 1, PANEL_TILES, Ops> product;
        product.clear();
        fragment_mma(product, p_codes, value_panel);
        thread int* int_product = product.elements();
        METAL_PRAGMA_UNROLL
        for (ushort row = 0; row < Ops::THREAD_ELEMENT_ROWS; ++row) {
          const AccumType block_weight = alpha[row];
          METAL_PRAGMA_UNROLL
          for (ushort panel_col = 0; panel_col < PANEL_TILES; ++panel_col) {
            METAL_PRAGMA_UNROLL
            for (ushort col = 0; col < Ops::THREAD_ELEMENT_COLS; ++col) {
              const ushort panel_element = panel_col * Ops::ELEMENTS_PER_THREAD + row * Ops::THREAD_ELEMENT_COLS + col;
              const ushort element =
                  (panel * PANEL_TILES + panel_col) * Ops::ELEMENTS_PER_THREAD + row * Ops::THREAD_ELEMENT_COLS + col;
              const AccumType value_dot = uzu::float_with_offset(int_product[panel_element]);
              accumulated[element] = metal::fma(block_weight, value_dot, accumulated[element]);
            }
          }
        }
      }
    }

    // Alternate slots to avoid a second barrier.
    slot_offset = uint(Scratch::SLOT_FLOATS) - slot_offset;
    kb += step_blocks;
    scale_offset += step_blocks * scale_block_stride;
  }

  output.map_rows(output_bias, [](AccumType value, AccumType bias) UZU_ALWAYS_INLINE { return value - bias; });

  const uint tile_index =
      ((split_idx - split_base + batch_idx) * num_kv_heads + kv_head_idx) * query_tiles + query_tile_idx;
  const size_t partial_offset =
      size_t(tile_index) * Layout::BLOCK_ROWS * HEAD_DIM + size_t(band) * Layout::TILE_ROWS * HEAD_DIM + slice_offset;
  AccumType inverse_split_sum[ROWS_PER_LANE];
  METAL_PRAGMA_UNROLL
  for (int r = 0; r < ROWS_PER_LANE; ++r) {
    inverse_split_sum[r] = sum_score[r] > AccumType(0) ? AccumType(1) / sum_score[r] : AccumType(1);
  }
  output.map_rows(inverse_split_sum, [](AccumType value, AccumType s) UZU_ALWAYS_INLINE { return value * s; });
  output.store(lane, partials + partial_offset, int(HEAD_DIM));

  if (slice == 0 && score_position.x == 0) {
    METAL_PRAGMA_UNROLL
    for (int r = 0; r < ROWS_PER_LANE; ++r) {
      const uint row = tile_index * uint(Layout::BLOCK_ROWS) + block_band_base + band_rows[r];
      partial_maxs[row] = max_score[r];
      partial_sums[row] = sum_score[r];
    }
  }
}

template <typename T, uint BD>
VARIANTS(T, bfloat)
VARIANTS(BD, 256)
KERNEL(AttentionGemmGroupedInt8Combine)(
    const device bfloat* partials,
    const device float* partial_maxs,
    const device float* partial_sums,
    device T* o,
    const device T* gate OPTIONAL(has_gate),
    const constant uzu::attention::AttnParams& params,
    const constant uint& grouped_rows,
    const constant uint& padded_rows,
    const constant uint& kv_heads,
    const constant uint& num_splits,
    const constant uint& gate_row_stride OPTIONAL(has_gate),
    const bool has_gate SPECIALIZE,
    threadgroup float rotation[BD],
    const uint grouped_row_idx GROUPS(grouped_rows),
    const uint kv_head_idx GROUPS(kv_heads),
    const uint column THREADS(BD)
) {
  const size_t row = size_t(kv_head_idx) * padded_rows + grouped_row_idx;
  const uint split_rows = kv_heads * padded_rows;
  const uint head_in_group = grouped_row_idx / params.q_len;
  const uint query_position = grouped_row_idx - head_in_group * params.q_len;
  const uint q_head_idx = kv_head_idx * params.gqa_factor + head_in_group;
  const device float* row_maxs = partial_maxs + row;
  const device float* row_sums = partial_sums + row;
  const device bfloat* row_means = partials + row * BD + column;

  float numerator = 0.0f;
  float denominator = 0.0f;
  constexpr uint SIMDGROUPS = BD / METAL_SIMD_SIZE;
  float global_max = -INFINITY;
  for (uint s = column; s < num_splits; s += BD) {
    if (row_sums[s * split_rows] > 0.0f) {
      global_max = metal::max(global_max, row_maxs[s * split_rows]);
    }
  }
  global_max = simd_max(global_max);
  if (column % METAL_SIMD_SIZE == 0) {
    rotation[column / METAL_SIMD_SIZE] = global_max;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  global_max = rotation[0];
  for (uint simdgroup = 1; simdgroup < SIMDGROUPS; ++simdgroup) {
    global_max = metal::max(global_max, rotation[simdgroup]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  threadgroup float* split_weights = rotation;
  for (uint chunk_start = 0; chunk_start < num_splits; chunk_start += BD) {
    const uint split = chunk_start + column;
    float weight = 0.0f;
    if (split < num_splits && row_sums[split * split_rows] > 0.0f) {
      weight = fast::exp2(row_maxs[split * split_rows] - global_max) * row_sums[split * split_rows];
    }
    split_weights[column] = weight;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint chunk_size = metal::min(num_splits - chunk_start, uint(BD));
    _Pragma("clang loop unroll_count(8)") for (uint i = 0; i < chunk_size; ++i) {
      const float split_weight = split_weights[i];
      denominator += split_weight;
      numerator += split_weight * float(row_means[size_t(chunk_start + i) * split_rows * BD]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float mean = denominator > 0.0f ? numerator / denominator : 0.0f;

  rotation[column] = mean;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (column >= METAL_SIMD_SIZE) {
    return;
  }
  constexpr ushort VALUES_PER_LANE = BD / METAL_SIMD_SIZE;
  float rotated[VALUES_PER_LANE];
  METAL_PRAGMA_UNROLL
  for (ushort value_index = 0; value_index < VALUES_PER_LANE; ++value_index) {
    rotated[value_index] = rotation[value_index * METAL_SIMD_SIZE + column];
  }
  simdgroup_hadamard_transform<VALUES_PER_LANE, METAL_SIMD_SIZE>(ushort(column), rotated);
  device T* output_row = o + size_t(q_head_idx) * params.o_strides[1] + size_t(query_position) * params.o_strides[2];
  METAL_PRAGMA_UNROLL
  for (ushort value_index = 0; value_index < VALUES_PER_LANE; ++value_index) {
    const uint channel = value_index * METAL_SIMD_SIZE + column;
    T result = static_cast<T>(rotated[value_index] * uzu::kv_layout::rotation_sign(channel));
    if (has_gate) {
      result = attention_apply_gate(result, gate[query_position * gate_row_stride + q_head_idx * BD + channel]);
    }
    output_row[channel] = result;
  }
}
