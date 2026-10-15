#include <metal_stdlib>
#include "../../common/dsl.h"
#include "../../common/conversion.h"
#include "../../common/thread_context.h"
#include "../../matmul/common/fragment.h"
#include "../../generated/attention.h"
#include "../../generated/trie.h"
#include "../kv_layout.h"
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

template <class Ops, int DEPTH_FRAGMENTS>
METAL_FUNC void prepare_queries(
    const device int8_t* query_codes,
    threadgroup uint4* staged_query,
    uint query_row_base,
    uint num_query_rows,
    uint query_row_stride,
    uint lane,
    uint channel_slice
) {
  // Padding rows reread the last real row.
  const short2 query_position = Ops::get_position(lane);
  const uint query_row = query_row_base + uint(query_position.y);
  const uint last_query_row = num_query_rows - 1u;
  const device int8_t* query_row0 =
      query_codes + size_t(min(query_row, last_query_row)) * query_row_stride + uint(query_position.x);
  const device int8_t* query_row1 =
      query_codes + size_t(min(query_row + uint(Ops::THREAD_ELEMENT_ROW_STRIDE), last_query_row)) * query_row_stride +
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
struct Int8Softmax {
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

  METAL_FUNC Int8Softmax() { output.clear(); }

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

  METAL_FUNC void normalize() thread {
    output.map_rows(value_bias, [](AccumType value, AccumType bias) UZU_ALWAYS_INLINE { return value - bias; });

    AccumType inverse_denominator[ROWS_PER_LANE];
    METAL_PRAGMA_UNROLL
    for (int row = 0; row < ROWS_PER_LANE; ++row) {
      inverse_denominator[row] = denominator[row] > AccumType(0) ? AccumType(1) / denominator[row] : AccumType(1);
    }
    output.map_rows(inverse_denominator, [](AccumType value, AccumType inverse_sum) UZU_ALWAYS_INLINE {
      return value * inverse_sum;
    });
  }
};

template <typename ScaleType, uint HEAD_DIM>
VARIANTS(ScaleType, bfloat)
VARIANTS(HEAD_DIM, 256)
KERNEL(AttentionGemmGroupedInt8)(
    const device int8_t* query_codes,
    const device ScaleType* query_scales,
    const device int8_t* key_codes,
    const device ScaleType* key_scales,
    const device uchar* value_codes,
    const device ScaleType* value_scales,
    const device uzu::trie::TrieNode* trie OPTIONAL(is_trie),
    device bfloat* partial_values,
    device float* partial_maxima,
    device float* partial_denominators,
    const constant uzu::attention::AttnParams& params,
    const constant uint& query_tiles,
    const constant uint& num_splits,
    const constant uint& split_base,
    const constant uint& split_count,
    const constant uint& trie_nodes,
    const bool mask_free SPECIALIZE,
    const bool is_trie SPECIALIZE,
    const uint32_t num_kv_heads SPECIALIZE,
    threadgroup float probability_exchange[2 * Int8Scratch<HEAD_DIM>::SLOT_FLOATS],
    threadgroup ulong trie_masks[Int8Scratch<HEAD_DIM>::TRIE_MASKS],
    threadgroup uint4 query_stage[Int8Scratch<HEAD_DIM>::QUERY_VECTORS],
    const ThreadContext thread_context,
    const uint tile_split_index GROUPS(query_tiles * split_count),
    const uint kv_head_index GROUPS(num_kv_heads),
    const uint batch_index GROUPS(1),
    const uint thread_index THREADS(ATTENTION_GQA_SIMDGROUPS * METAL_SIMD_SIZE)
) {
  using Layout = AttentionGemmGroupedLayout<uzu::kv_layout::TILE_KEYS, HEAD_DIM, HEAD_DIM / uzu::kv_layout::SLICE_DIMS>;
  using Scratch = Int8Scratch<HEAD_DIM>;
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

  const uint num_query_rows = params.gqa_factor * params.q_len;
  const uint query_row_base = query_tile_index * uint(Layout::BLOCK_ROWS) + uint(query_band) * uint(Layout::TILE_ROWS);

  const uint channel_offset = uint(channel_slice) * uint(Layout::SLICE_COLS);

  const size_t query_group_offset =
      size_t(batch_index) * params.q_strides[0] + size_t(kv_head_index) * params.q_strides[1];
  query_codes += query_group_offset;
  key_codes += size_t(batch_index) * params.k_strides[0] +
               uint(uzu::kv_layout::TILE_KEYS) * size_t(kv_head_index) * params.k_strides[1];
  key_codes += int(lane) * uzu::kv_layout::LANE_BYTES;
  const device uchar* value_tiles =
      value_codes +
      (size_t(batch_index) * params.v_strides[0] +
       size_t(uzu::kv_layout::TILE_KEYS) * (size_t(kv_head_index) * params.v_strides[1] + channel_offset)) +
      size_t(lane) * uzu::kv_layout::LANE_BYTES;

  Int8Softmax<Fragment<AccumType, 1, VALUE_FRAGMENTS, Ops>> softmax;

  const uint prefix_length = params.q_off;
  const uint key_length = params.k_len;
  const short2 score_position = ScoreFragment::get_position(lane);
  const uint lane_rows[ROWS_PER_LANE] = {
      uint(score_position.y),
      uint(score_position.y) + uint(Ops::THREAD_ELEMENT_ROW_STRIDE)
  };
  const uint query_band_offset = uint(query_band) * uint(Layout::TILE_ROWS);
  uint query_positions[ROWS_PER_LANE];
  if (!mask_free) {
    METAL_PRAGMA_UNROLL
    for (ushort row = 0; row < ROWS_PER_LANE; ++row) {
      query_positions[row] = (query_row_base + lane_rows[row]) % params.q_len;
    }
    if (is_trie) {
      ulong row_trie_mask[ROWS_PER_LANE] = {};
      for (uint node_index = 0; node_index < trie_nodes; ++node_index) {
        const uzu::trie::TrieNode node = trie[node_index];
        METAL_PRAGMA_UNROLL
        for (ushort row = 0; row < ROWS_PER_LANE; ++row) {
          if (query_positions[row] >= node.trie_start && query_positions[row] <= node.trie_end) {
            row_trie_mask[row] |= (1ul << node_index);
          }
        }
      }
      if (score_position.x == 0) {
        METAL_PRAGMA_UNROLL
        for (ushort row = 0; row < ROWS_PER_LANE; ++row) {
          trie_masks[query_band_offset + lane_rows[row]] = row_trie_mask[row];
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  const uint key_block_stride = uint(uzu::kv_layout::TILE_KEYS) * params.k_strides[2];
  const uint value_block_stride = uint(uzu::kv_layout::TILE_KEYS) * params.v_strides[2];

  const uint first_block = (split_index * params.nk) / num_splits;
  const uint tile_first_row = query_tile_index * uint(Layout::BLOCK_ROWS);
  const uint tile_last_row = min(tile_first_row + uint(Layout::BLOCK_ROWS) - 1u, num_query_rows - 1u);
  const uint last_query_position = (tile_first_row / params.q_len == tile_last_row / params.q_len)
                                       ? (tile_last_row % params.q_len)
                                       : (params.q_len - 1u);
  const uint split_end_block = ((split_index + 1u) * params.nk) / num_splits;
  const uint end_block =
      min(split_end_block,
          (prefix_length + last_query_position + uint(uzu::kv_layout::TILE_KEYS)) / uint(uzu::kv_layout::TILE_KEYS));
  const uint block_count = end_block > first_block ? end_block - first_block : 0u;
  const bool has_tail_block = (params.k_rem != 0u) && (block_count > 0u) && (end_block == params.nk);

  key_codes += size_t(first_block) * key_block_stride;
  value_tiles += size_t(first_block) * value_block_stride;
  const uint first_key = first_block * uint(uzu::kv_layout::TILE_KEYS);

  const AccumType log2_attention_scale = AccumType(params.scale) * AccumType(M_LOG2E_F);

  const uint scales_per_block = uint(uzu::kv_layout::TILE_KEYS) * num_kv_heads;
  const size_t first_scale_index =
      uzu::kv_layout::Int8KVShape{HEAD_DIM, num_kv_heads}.scale_index(first_key, kv_head_index);
  const device ScaleType* block_key_scales = key_scales + first_scale_index;
  const device ScaleType* block_value_scales = value_scales + first_scale_index;

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
  uint block_scale_offset = uint(channel_slice) * scales_per_block + uint(score_position.x);

  threadgroup uint4* const staged_query =
      query_stage + uint(query_band) * uint(Scratch::QUERY_VECTORS_PER_BAND) + uint(lane);
  auto compute_probabilities = [&](uint block_index,
                                   uint channel_slice,
                                   threadgroup float4* slot_stats,
                                   threadgroup float2* slot_sums,
                                   threadgroup uchar* slot_codes) UZU_ALWAYS_INLINE {
    const uint key_block = block_index + uint(channel_slice);
    const bool is_tail_block = has_tail_block && (key_block + 1u == block_count);
    const uint key_base = first_key + key_block * uint(uzu::kv_layout::TILE_KEYS);
    const device int8_t* block_keys = key_codes + key_block * key_block_stride;

    ScoreFragment scores;

    // Q x K^T (int8).
    Fragment<int, 1, KEY_FRAGMENTS, Ops> score;
    score.clear();
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
          const uint4 packed_keys = *reinterpret_cast<const device uint4*>(
              block_keys + (depth_fragment + pair_index) * int(uzu::kv_layout::FRAGMENT_BYTES)
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
    {
      const device ScaleType* lane_key_scales = block_key_scales + block_scale_offset;
      const device ScaleType* lane_value_scales = block_value_scales + block_scale_offset;
      METAL_PRAGMA_UNROLL
      for (int fragment_col = 0; fragment_col < KEY_FRAGMENTS; ++fragment_col) {
        METAL_PRAGMA_UNROLL
        for (int column = 0; column < Ops::THREAD_ELEMENT_COLS; ++column) {
          const int lane_col = fragment_col * Ops::THREAD_ELEMENT_COLS + column;
          const int scale_col = fragment_col * Ops::FRAGMENT_COLS + column;
          const bool valid = mask_free || key_base + uint(score_position.x) + uint(scale_col) < key_length;
          key_scale_values[lane_col] = AccumType(lane_key_scales[scale_col]);
          const float value_scale = float(lane_value_scales[scale_col]);
          log2_value_scales[lane_col] = valid ? metal::precise::log2(value_scale) : 0.0f;
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

    if (!mask_free && (is_tail_block || (key_base + uint(uzu::kv_layout::TILE_KEYS) > prefix_length))) {
      ulong trie_row_masks[ROWS_PER_LANE] = {0ul, 0ul};
      if (is_trie) {
        METAL_PRAGMA_UNROLL
        for (ushort row = 0; row < ROWS_PER_LANE; ++row) {
          trie_row_masks[row] = trie_masks[query_band_offset + lane_rows[row]];
        }
      }
      map_lane_rows(scores, score_position, [&](ushort lane_row, short column, AccumType value) UZU_ALWAYS_INLINE {
        const uint key = key_base + uint(column);
        bool visible = key < key_length;
        if (is_trie) {
          visible = visible &&
                    (key < prefix_length || ((trie_row_masks[lane_row] >> ulong(key - prefix_length)) & 1ul) != 0ul);
        } else {
          visible = visible && key <= prefix_length + query_positions[lane_row];
        }
        return visible ? value : -metal::numeric_limits<AccumType>::infinity();
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
                               threadgroup uchar* slot_codes) UZU_ALWAYS_INLINE {
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

      const uint value_block_index = block_index + published_block;
      thread AccumType* output_elements = softmax.output.elements();
      // Accumulate V panels directly into their output columns.
      using ValueVector = typename Ops::template ThreadVector<uchar>;
      const device uchar* block_values = value_tiles + value_block_index * value_block_stride;
      METAL_PRAGMA_UNROLL
      for (int panel = 0; panel < VALUE_PANELS; ++panel) {
        OperandFragment<uchar, KEY_FRAGMENTS, FRAGMENTS_PER_PANEL, Ops> value_fragments;
        METAL_PRAGMA_UNROLL
        for (int key_fragment = 0; key_fragment < KEY_FRAGMENTS; ++key_fragment) {
          const uint4 packed_values = *reinterpret_cast<const device uint4*>(
              block_values + (key_fragment * VALUE_PANELS + panel) * uzu::kv_layout::FRAGMENT_BYTES
          );
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

    if (uint(channel_slice) < blocks_in_step) {
      compute_probabilities(block_index, channel_slice, slot_stats, slot_sums, slot_codes);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    softmax.rescale(slot_stats, blocks_in_step, STAT_ROWS);

    accumulate_output(block_index, blocks_in_step, slot_stats, slot_sums, slot_codes);

    // Alternate slots to avoid a second barrier.
    slot_offset = uint(Scratch::SLOT_FLOATS) - slot_offset;
    block_index += blocks_in_step;
    block_scale_offset += blocks_in_step * scales_per_block;
  }

  softmax.normalize();

  const uint partial_tile_index =
      ((split_index - split_base + batch_index) * num_kv_heads + kv_head_index) * query_tiles + query_tile_index;
  const size_t partial_offset = size_t(partial_tile_index) * Layout::BLOCK_ROWS * HEAD_DIM +
                                size_t(query_band) * Layout::TILE_ROWS * HEAD_DIM + channel_offset;
  softmax.output.store(lane, partial_values + partial_offset, int(HEAD_DIM));

  if (channel_slice == 0 && score_position.x == 0) {
    METAL_PRAGMA_UNROLL
    for (int row = 0; row < ROWS_PER_LANE; ++row) {
      const uint row_index = partial_tile_index * uint(Layout::BLOCK_ROWS) + query_band_offset + lane_rows[row];
      partial_maxima[row_index] = softmax.running_max[row];
      partial_denominators[row_index] = softmax.denominator[row];
    }
  }
}
