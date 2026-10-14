#include <metal_stdlib>
#include "../../common/dsl.h"
#include "../../common/integral_constant.h"
#include "../../common/thread_context.h"
#include "../../matmul/common/fragment.h"
#include "../../matmul/common/mxu_fragment/ops.h"
#include "../../generated/attention.h"
#include "../../generated/trie.h"
#include "../kv_layout.h"
#include "util.h"

// Per 128-key step, slice simdgroup `s` scores key block `s` in full and publishes its u8 P codes and row stats to
// threadgroup memory; every simdgroup then accumulates its 64-channel P x V slice over all published blocks.

using namespace metal;
using namespace uzu::matmul;
using namespace uzu::attention;
using namespace uzu::trie;
namespace kv_layout = uzu::kv_layout;

// Bit tricks: a float in [2^23, 2^24) holds its integer part in the mantissa.
UZU_CONST float BIASED_INT_OFFSET = 0x1p23f;

// rint(x) for 0 <= x < 256, in one add.
METAL_FUNC uchar round_to_byte(const float x) { return uchar(as_type<uint>(x + 0x1.8p23f)); }

// BIASED_INT_OFFSET + i for 0 <= i < 2^23, in one or; the offset is subtracted after accumulation.
METAL_FUNC float biased_int_to_float(const int i) {
  return as_type<float>(as_type<uint>(i) | as_type<uint>(BIASED_INT_OFFSET));
}

// One exchange slot per step: every simdgroup's u8 P codes, `(max0, sum0, max1, sum1)` stats and probability sums.
template <uint HEAD_DIM>
struct AttentionGemmGroupedInt8Layout {
  using Grouped = AttentionGemmGroupedLayout<float, kv_layout::TILE_KEYS, HEAD_DIM, HEAD_DIM / kv_layout::SLICE_DIMS>;

  UZU_CONST int CODE_FLOATS = Grouped::SIMDGROUPS * Grouped::SCORE_TILE_ELEMENTS / int(sizeof(float));
  UZU_CONST int STAT_FLOATS = Grouped::SIMDGROUPS * Grouped::TILE_ROWS * 2;
  UZU_CONST int SLOT_FLOATS = CODE_FLOATS + STAT_FLOATS + Grouped::SIMDGROUPS * Grouped::TILE_ROWS;

  // Q stage: one uint4 per lane and fragment pair.
  UZU_CONST int QUERY_FRAGMENT_PAIRS = int(HEAD_DIM) / (2 * Grouped::TILE_ROWS);
  UZU_CONST int QUERY_STAGE_VECTORS = Grouped::BANDS * QUERY_FRAGMENT_PAIRS * METAL_SIMD_SIZE;
};

// Lane-local (column, row) index of the element at fragment coordinate (row, col); constant once unrolled.
template <class Ops>
METAL_FUNC ushort2 lane_index(const short2 position, const short row, const short col) {
  const short lane_col = col - position.x;
  return ushort2(
      ushort(lane_col / Ops::FRAGMENT_COLS * Ops::THREAD_ELEMENT_COLS + lane_col % Ops::FRAGMENT_COLS),
      ushort((row - position.y) / Ops::THREAD_ELEMENT_ROW_STRIDE)
  );
}

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
    const device TrieNode* trie OPTIONAL(is_trie),
    device bfloat* partials,
    device float* partial_maxs,
    device float* partial_sums,
    const constant AttnParams& params,
    const constant uint& m_tiles,
    const constant uint& batch_count,
    const constant uint& num_splits,
    const constant uint& split_base,
    const constant uint& split_count,
    const constant uint& trie_nodes,
    const bool is_trie SPECIALIZE,
    const uint32_t num_kv_heads SPECIALIZE,
    threadgroup float p_exchange[2 * AttentionGemmGroupedInt8Layout<HEAD_DIM>::SLOT_FLOATS],
    threadgroup ulong trie_masks[32],
    threadgroup uint4 query_stage[AttentionGemmGroupedInt8Layout<HEAD_DIM>::QUERY_STAGE_VECTORS],
    const ThreadContext thread_context,
    const uint tile_split_flat GROUPS(m_tiles * split_count),
    const uint kv_head_idx GROUPS(num_kv_heads),
    const uint batch_idx GROUPS(batch_count),
    const uint lid THREADS(ATTENTION_GQA_SIMDGROUPS * METAL_SIMD_SIZE)
) {
  using Int8Layout = AttentionGemmGroupedInt8Layout<HEAD_DIM>;
  using Layout = typename Int8Layout::Grouped;
  using Ops = typename Layout::Ops;
  using AccumType = float;

  constexpr uint D_SLICES = HEAD_DIM / kv_layout::SLICE_DIMS;

  constexpr int KEY_COL_FRAGMENTS = int(kv_layout::TILE_KEYS) / Ops::FRAGMENT_COLS;
  constexpr int HEAD_DIM_FRAGMENTS_PER_SLICE = Layout::SLICE_COLS / Ops::FRAGMENT_ROWS;
  constexpr int VALUE_COL_FRAGMENTS = Layout::SLICE_COLS / Ops::FRAGMENT_COLS;
  constexpr int PANEL_COL_FRAGMENTS = 2;
  constexpr int PANELS = VALUE_COL_FRAGMENTS / PANEL_COL_FRAGMENTS;
  constexpr int ROWS_PER_LANE = Ops::THREAD_ELEMENT_ROWS;
  constexpr int LANE_COLS = KEY_COL_FRAGMENTS * Ops::THREAD_ELEMENT_COLS;
  constexpr uint STAT_ROWS = uint(Layout::TILE_ROWS / ROWS_PER_LANE);

  static_assert(KEY_COL_FRAGMENTS == 2 && VALUE_COL_FRAGMENTS == 4, "32 keys x 64 channels per simdgroup");
  static_assert(ROWS_PER_LANE == 2, "stats pack two rows per lane");

  // Online softmax in log2 units. The running max starts finite so the first rescale is exp2(-huge) = 0, not NaN;
  // masked scores underflow to 0.
  constexpr AccumType INITIAL_MAX = -1e30f;
  constexpr AccumType MASKED_SCORE = -1e9f;
  // Rescale only when a block max exceeds the running max by more than this (block weights stay <= 2^8).
  constexpr AccumType RESCALE_THRESHOLD = 8.0f;

  // P is stored as u8: p_code = rint(255 * exp2(score - block_max)), so p = p_code / 255.
  constexpr AccumType CODE_MAX = 255.0f;
  constexpr AccumType LOG2_CODE_MAX = 7.9943534368588578f; // log2(255)

  using QueryCodeFragment = OperandFragment<int8_t, 1, 1, Ops>;
  using KeyCodeFragment = OperandFragment<int8_t, 1, KEY_COL_FRAGMENTS, Ops, ReadTranspose>;
  using ScoreFragment = Fragment<AccumType, 1, KEY_COL_FRAGMENTS, Ops>;
  using IntScoreFragment = Fragment<int, 1, KEY_COL_FRAGMENTS, Ops>;
  using OutputFragment = Fragment<AccumType, 1, VALUE_COL_FRAGMENTS, Ops>;
  using ProbabilityCodeFragment = Fragment<uchar, 1, KEY_COL_FRAGMENTS, Ops>;
  using ValuePanelFragment = OperandFragment<uchar, KEY_COL_FRAGMENTS, PANEL_COL_FRAGMENTS, Ops>;
  using IntPanelFragment = Fragment<int, 1, PANEL_COL_FRAGMENTS, Ops>;

  const ushort lane = ushort(thread_context.simd_lane_id);
  const ushort simdgroup_index = ushort(thread_context.simdgroup_index);
  const ushort band = simdgroup_index / ushort(D_SLICES);
  const ushort slice = simdgroup_index % ushort(D_SLICES);

  const uint m_tile_idx = tile_split_flat % m_tiles;
  const uint split_idx = split_base + tile_split_flat / m_tiles;

  const uint grouped_rows = params.gqa_factor * params.q_len;
  const uint row_base = m_tile_idx * uint(Layout::BLOCK_ROWS) + uint(band) * uint(Layout::TILE_ROWS);

  const uint query_row_stride = params.q_strides[2];
  const uint key_row_stride = params.k_strides[2];
  const uint value_row_stride = params.v_strides[2];
  const uint slice_offset = uint(slice) * uint(Layout::SLICE_COLS);

  const size_t query_group_base = size_t(batch_idx) * params.q_strides[0] + size_t(kv_head_idx) * params.q_strides[1];
  q_codes += query_group_base + size_t(row_base) * query_row_stride;
  k_codes +=
      size_t(batch_idx) * params.k_strides[0] + uint(kv_layout::TILE_KEYS) * size_t(kv_head_idx) * params.k_strides[1];
  k_codes += int(lane) * kv_layout::LANE_BYTES;
  const device uchar* v_tile_codes =
      v_codes +
      (size_t(batch_idx) * params.v_strides[0] +
       size_t(kv_layout::TILE_KEYS) * (size_t(kv_head_idx) * params.v_strides[1] + slice_offset)) +
      size_t(lane) * kv_layout::LANE_BYTES;

  // Padding rows reread the last real row.
  int query_offset_row0 = 0;
  int query_offset_row1 = 0;
  {
    const short2 query_position = QueryCodeFragment::get_position(lane);
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

  OutputFragment output;
  output.clear();

  AccumType max_score[ROWS_PER_LANE] = {INITIAL_MAX, INITIAL_MAX};
  AccumType sum_score[ROWS_PER_LANE] = {0.0f, 0.0f};
  AccumType output_bias[ROWS_PER_LANE] = {0.0f, 0.0f};

  const uint prefix_length = params.q_off;
  const uint key_length = params.k_len;
  const short2 score_position = ScoreFragment::get_position(lane);
  // This lane's two rows within the band.
  const uint band_rows[ROWS_PER_LANE] = {
      uint(score_position.y),
      uint(score_position.y) + uint(Ops::THREAD_ELEMENT_ROW_STRIDE)
  };
  const uint block_band_base = uint(band) * uint(Layout::TILE_ROWS);
  // Causal or trie mask per row.
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
      const TrieNode node = trie[node_index];
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

  const uint key_block_stride = uint(kv_layout::TILE_KEYS) * key_row_stride;
  const uint value_block_stride = uint(kv_layout::TILE_KEYS) * value_row_stride;

  // This split's key blocks.
  const uint block_begin = (split_idx * params.nk) / num_splits;
  const uint tile_first_row = m_tile_idx * uint(Layout::BLOCK_ROWS);
  const uint tile_last_row = min(tile_first_row + uint(Layout::BLOCK_ROWS) - 1u, grouped_rows - 1u);
  const uint max_query_index = (tile_first_row / params.q_len == tile_last_row / params.q_len)
                                   ? (tile_last_row % params.q_len)
                                   : (params.q_len - 1u);
  const uint block_end_candidate = ((split_idx + 1u) * params.nk) / num_splits;
  const uint block_end =
      min(block_end_candidate,
          (prefix_length + max_query_index + uint(kv_layout::TILE_KEYS)) / uint(kv_layout::TILE_KEYS));
  const uint tail_keys = params.k_rem;
  const uint block_count = block_end > block_begin ? block_end - block_begin : 0u;
  const bool owns_tail_block = (tail_keys != 0u) && (block_count > 0u) && (block_end == params.nk);

  k_codes += size_t(block_begin) * key_block_stride;
  v_tile_codes += size_t(block_begin) * value_block_stride;
  const uint first_key = block_begin * uint(kv_layout::TILE_KEYS);

  const AccumType score_scale = AccumType(params.scale) * AccumType(M_LOG2E_F);

  const uint scale_block_stride = uint(kv_layout::TILE_KEYS) * num_kv_heads;
  const size_t split_scale_offset = kv_layout::Int8KVShape{HEAD_DIM, num_kv_heads}.scale_index(first_key, kv_head_idx);
  const device T* key_scales_base = k_scales + split_scale_offset;
  const device T* value_scales_base = v_scales + split_scale_offset;

  AccumType query_row_scale[ROWS_PER_LANE];
  METAL_PRAGMA_UNROLL
  for (int r = 0; r < ROWS_PER_LANE; ++r) {
    const uint query_row = min(row_base + band_rows[r], grouped_rows - 1u);
    query_row_scale[r] = AccumType(q_scales[query_group_base / HEAD_DIM + query_row]) * score_scale;
  }

  threadgroup uchar* code_slot = reinterpret_cast<threadgroup uchar*>(p_exchange);
  threadgroup float* stat_slot = p_exchange + Int8Layout::CODE_FLOATS;
  const ushort band_first_simdgroup = band * ushort(int(D_SLICES));

  threadgroup uchar* const lane_codes =
      code_slot + uint(band_first_simdgroup) * uint(Layout::SCORE_TILE_ELEMENTS) + uint(lane) * kv_layout::LANE_BYTES;
  threadgroup float4* const lane_stats = reinterpret_cast<threadgroup float4*>(stat_slot) +
                                         uint(band_first_simdgroup) * STAT_ROWS + uint(score_position.y);
  threadgroup float2* const lane_probability_sums =
      reinterpret_cast<threadgroup float2*>(stat_slot + Int8Layout::STAT_FLOATS) +
      uint(band_first_simdgroup) * STAT_ROWS + uint(score_position.y);
  uint slot_offset = 0u;
  uint scale_offset = uint(slice) * scale_block_stride + uint(score_position.x);

  // Each slice simdgroup stages its depth slice of the band's Q.
  threadgroup uint4* const staged_query =
      query_stage + uint(band) * uint(Int8Layout::QUERY_FRAGMENT_PAIRS * METAL_SIMD_SIZE) + uint(lane);
  {
    constexpr int SLICE_PAIRS = HEAD_DIM_FRAGMENTS_PER_SLICE / 2;
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
    threadgroup float2* slot_probability_sums = lane_probability_sums + slot_offset / 2u;
    threadgroup uchar* slot_codes = lane_codes + 4u * slot_offset;

    if (publishes_block) {
      const uint own_block = kb + uint(slice);
      const bool is_tail_block = owns_tail_block && (own_block + 1u == block_count);
      const uint key_base = first_key + own_block * uint(kv_layout::TILE_KEYS);
      const device int8_t* key_tile_base = k_codes + own_block * key_block_stride;

      ScoreFragment score;

      // Q x K^T (int8).
      IntScoreFragment int_score;
      int_score.clear();
      for (int depth_slice = 0; depth_slice < int(D_SLICES); ++depth_slice) {
        METAL_PRAGMA_UNROLL
        for (int slice_fragment = 0; slice_fragment < HEAD_DIM_FRAGMENTS_PER_SLICE; slice_fragment += 2) {
          const int depth_fragment = depth_slice * HEAD_DIM_FRAGMENTS_PER_SLICE + slice_fragment;
          using KeyVector = typename KeyCodeFragment::ThreadVectorType;
          using QueryVector = typename QueryCodeFragment::ThreadVectorType;
          using QueryPairFragment = OperandFragment<int8_t, 1, 2, Ops>;
          using KeyPairFragment = OperandFragment<int8_t, 2, 2, Ops, ReadTranspose>;
          QueryPairFragment query_fragments;
          const uint4 query_pair = staged_query[(depth_fragment / 2) * METAL_SIMD_SIZE];
          query_fragments.fragment_data[0] = as_type<QueryVector>(query_pair.xy);
          query_fragments.fragment_data[1] = as_type<QueryVector>(query_pair.zw);
          KeyPairFragment key_fragments;
          METAL_PRAGMA_UNROLL
          for (int pair_index = 0; pair_index < 2; ++pair_index) {
            const uint4 key_chunk = *reinterpret_cast<const device uint4*>(
                key_tile_base + (depth_fragment + pair_index) * int(kv_layout::FRAGMENT_BYTES)
            );
            key_fragments.fragment_data[pair_index] = as_type<KeyVector>(uint2(key_chunk.x, key_chunk.y));
            key_fragments.fragment_data[2 + pair_index] = as_type<KeyVector>(uint2(key_chunk.z, key_chunk.w));
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
        for (int fragment_col = 0; fragment_col < KEY_COL_FRAGMENTS; ++fragment_col) {
          METAL_PRAGMA_UNROLL
          for (int col = 0; col < Ops::THREAD_ELEMENT_COLS; ++col) {
            const int lane_col = fragment_col * Ops::THREAD_ELEMENT_COLS + col;
            key_scale[lane_col] = AccumType(key_scale_lane[fragment_col * Ops::FRAGMENT_COLS + col]);
            const float v_scale = float(value_scale_lane[fragment_col * Ops::FRAGMENT_COLS + col]);
            // A never-written slot gives (0, 0), so masked keys stay finite.
            const bool written = v_scale >= kv_layout::MIN_VALUE_SCALE;
            log2_value_scale[lane_col] = written ? metal::precise::log2(v_scale) : 0.0f;
            inverse_value_scale[lane_col] = written ? metal::precise::divide(1.0f, v_scale) : 0.0f;
          }
        }
      }
      ScoreFragment::zip_for_each_coord(
          lane,
          [&](short row, short col, thread AccumType& value, thread int& int_value) UZU_ALWAYS_INLINE {
            const ushort2 index = lane_index<Ops>(score_position, row, col);
            value = metal::fma(
                AccumType(int_value) * key_scale[index.x],
                query_row_scale[index.y],
                log2_value_scale[index.x]
            );
          },
          score,
          int_score
      );

      // Apply mask.
      if (!MASK_FREE && (is_tail_block || (key_base + uint(kv_layout::TILE_KEYS) > prefix_length))) {
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

      // Quantize P to u8 codes; row max, code sum and probability sum.
      AccumType block_max[ROWS_PER_LANE];
      AccumType code_sum[ROWS_PER_LANE];
      ProbabilityCodeFragment probability_codes;
      score.row_reduce(block_max, -INFINITY, [](AccumType a, AccumType b) UZU_ALWAYS_INLINE {
        return metal::max(a, b);
      });
      ScoreFragment::zip_for_each_coord(
          lane,
          [&](short row, short col, thread AccumType& value, thread uchar& code) UZU_ALWAYS_INLINE {
            const ushort2 index = lane_index<Ops>(score_position, row, col);
            value = metal::rint(fast::exp2(value - (block_max[index.y] - LOG2_CODE_MAX)));
            code = round_to_byte(value);
          },
          score,
          probability_codes
      );
      score.row_reduce(code_sum, AccumType(0), [](AccumType a, AccumType b) UZU_ALWAYS_INLINE { return a + b; });
      AccumType probability_sum[ROWS_PER_LANE];
      score.map_coords(lane, [&](short row, short col, AccumType value) UZU_ALWAYS_INLINE {
        return value * inverse_value_scale[lane_index<Ops>(score_position, row, col).x];
      });
      score.row_reduce(probability_sum, AccumType(0), [](AccumType a, AccumType b) UZU_ALWAYS_INLINE { return a + b; });

      // Publish.
      *reinterpret_cast<threadgroup uint4*>(slot_codes + uint(slice) * uint(Layout::SCORE_TILE_ELEMENTS)) =
          uint4(as_type<uint2>(probability_codes.fragment_data[0]), as_type<uint2>(probability_codes.fragment_data[1]));
      if (score_position.x == 0) {
        slot_stats[uint(slice) * STAT_ROWS] = float4(block_max[0], code_sum[0], block_max[1], code_sum[1]);
        slot_probability_sums[uint(slice) * STAT_ROWS] = float2(probability_sum[0], probability_sum[1]);
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
      const float2 packed_probability_sums = slot_probability_sums[published_block * STAT_ROWS];
      const AccumType probability_sum[ROWS_PER_LANE] = {packed_probability_sums.x, packed_probability_sums.y};
      AccumType alpha[ROWS_PER_LANE];
      METAL_PRAGMA_UNROLL
      for (int r = 0; r < ROWS_PER_LANE; ++r) {
        alpha[r] = fast::exp2(block_max[r] - max_score[r]) * (1.0f / CODE_MAX);
        sum_score[r] += alpha[r] * probability_sum[r];
        output_bias[r] += alpha[r] * AccumType(kv_layout::V_CODE_BIAS) * code_sum[r];
        output_bias[r] += alpha[r] * BIASED_INT_OFFSET;
      }

      ProbabilityCodeFragment probability_codes;
      const uint4 packed_codes =
          *reinterpret_cast<const threadgroup uint4*>(slot_codes + published_block * Layout::SCORE_TILE_ELEMENTS);
      probability_codes.fragment_data[0] = as_type<typename ProbabilityCodeFragment::ThreadVectorType>(packed_codes.xy);
      probability_codes.fragment_data[1] = as_type<typename ProbabilityCodeFragment::ThreadVectorType>(packed_codes.zw);

      const uint value_block_index = kb + published_block;
      thread AccumType* accumulated = output.elements();
      // Each 32-channel panel is merged into the output before the next panel's V is loaded. The panel is a column
      // sub-range of `output`, so the shape-matched Fragment helpers do not apply.
      using ValueVector = typename ValuePanelFragment::ThreadVectorType;
      const device uchar* value_block = v_tile_codes + value_block_index * value_block_stride;
      METAL_PRAGMA_UNROLL
      for (int panel = 0; panel < PANELS; ++panel) {
        ValuePanelFragment value_panel;
        METAL_PRAGMA_UNROLL
        for (int key_fragment = 0; key_fragment < KEY_COL_FRAGMENTS; ++key_fragment) {
          const uint4 tile = *reinterpret_cast<const device uint4*>(
              value_block + (key_fragment * PANELS + panel) * kv_layout::FRAGMENT_BYTES
          );
          value_panel.fragment_data[key_fragment * PANEL_COL_FRAGMENTS] = as_type<ValueVector>(tile.xy);
          value_panel.fragment_data[key_fragment * PANEL_COL_FRAGMENTS + 1] = as_type<ValueVector>(tile.zw);
        }
        simdgroup_barrier(mem_flags::mem_none);
        IntPanelFragment product;
        product.clear();
        fragment_mma(product, probability_codes, value_panel);
        thread int* int_product = product.elements();
        METAL_PRAGMA_UNROLL
        for (ushort row = 0; row < Ops::THREAD_ELEMENT_ROWS; ++row) {
          const AccumType block_weight = alpha[row];
          METAL_PRAGMA_UNROLL
          for (ushort panel_col = 0; panel_col < PANEL_COL_FRAGMENTS; ++panel_col) {
            METAL_PRAGMA_UNROLL
            for (ushort col = 0; col < Ops::THREAD_ELEMENT_COLS; ++col) {
              const ushort panel_element = panel_col * Ops::ELEMENTS_PER_THREAD + row * Ops::THREAD_ELEMENT_COLS + col;
              const ushort element = (panel * PANEL_COL_FRAGMENTS + panel_col) * Ops::ELEMENTS_PER_THREAD +
                                     row * Ops::THREAD_ELEMENT_COLS + col;
              const AccumType value_dot = biased_int_to_float(int_product[panel_element]);
              accumulated[element] = metal::fma(block_weight, value_dot, accumulated[element]);
            }
          }
        }
      }
    }

    // Double-buffered: the next step writes the other slot, so one barrier per step suffices.
    slot_offset = uint(Int8Layout::SLOT_FLOATS) - slot_offset;
    kb += step_blocks;
    scale_offset += step_blocks * scale_block_stride;
  }

  // Normalize and store the split partials.
  output.map_rows(output_bias, [](AccumType value, AccumType bias) UZU_ALWAYS_INLINE { return value - bias; });

  const uint tile_index =
      (((split_idx - split_base) * batch_count + batch_idx) * num_kv_heads + kv_head_idx) * m_tiles + m_tile_idx;
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
