#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../matmul/common/fragment.h"
#include "../matmul/common/mxu_fragment/ops.h"
#include "../matmul/common/simdgroup_fragment_ops.h"
#include "kv_layout.h"

using namespace metal;
using namespace uzu::matmul;
namespace kv_layout = uzu::kv_layout;

#define THREADS_PER_TG 256

UZU_CONST uint NUM_SIMDGROUPS = THREADS_PER_TG / METAL_SIMD_SIZE;

template <uint NUM_HEADS, uint HEAD_DIM, bool USE_MXU>
VARIANTS(NUM_HEADS, 4)
VARIANTS(HEAD_DIM, 128)
VARIANTS(USE_MXU, false, true)
KERNEL(AttentionBlockScore)(
    const device int8_t* query_codes,
    const device bfloat* query_scales,
    const device int8_t* key_codes,
    const device bfloat* key_scales,
    device half* scores,
    const constant uint& first_row,
    const constant uint& row_count,
    const constant uint& query_rows,
    const constant uint& prefix_tokens,
    const constant uint& cached_block_row,
    const constant uint& rows_per_group,
    const constant uint& blocks_per_group,
    const constant uint& candidate_blocks,
    const ThreadContext thread_context,
    const uint row_group_index GROUPS(row_count.div_ceil(rows_per_group)),
    const uint block_group_index GROUPS(candidate_blocks.div_ceil(blocks_per_group)),
    const uint thread_index THREADS(THREADS_PER_TG)
) {
  using Ops = metal::conditional_t<USE_MXU, MxuFragmentOps<>, SimdgroupFragmentOps>;
  constexpr int DEPTH_FRAGMENTS = HEAD_DIM / Ops::FRAGMENT_COLS;
  constexpr uint KEY_FRAGMENTS = USE_MXU ? 2 : 1;
  constexpr uint KEY_TILE = KEY_FRAGMENTS * Ops::FRAGMENT_COLS;
  constexpr int SLICE_FRAGMENTS = kv_layout::SLICE_DIMS / Ops::FRAGMENT_COLS;
  constexpr int DEPTH_SLICES = HEAD_DIM / kv_layout::SLICE_DIMS;
  constexpr ushort LANE_ROWS = 2;
  using Code = typename Ops::template ThreadVector<int8_t>;
  using Queries =
      metal::conditional_t<USE_MXU, Fragment<int8_t, 1, DEPTH_FRAGMENTS, Ops>, Fragment<half, DEPTH_FRAGMENTS, 1, Ops>>;
  using Keys = metal::conditional_t<
      USE_MXU,
      OperandFragment<int8_t, DEPTH_FRAGMENTS, KEY_FRAGMENTS, Ops, ReadTranspose>,
      Fragment<half, KEY_FRAGMENTS, DEPTH_FRAGMENTS, Ops>>;
  using Dots =
      metal::conditional_t<USE_MXU, Fragment<int, 1, KEY_FRAGMENTS, Ops>, Fragment<float, KEY_FRAGMENTS, 1, Ops>>;

  const uint cached_blocks = prefix_tokens / kv_layout::COMPRESSION_RATIO;
  // A suffix block's key is at its last token's row.
  const auto key_slot = [&](uint candidate_block) {
    return candidate_block < cached_blocks ? cached_block_row + candidate_block
                                           : (candidate_block + 1) * kv_layout::COMPRESSION_RATIO - 1 - prefix_tokens;
  };
  const uint query_bands = rows_per_group / Ops::FRAGMENT_ROWS;
  const uint query_band = thread_context.simdgroup_index % query_bands;
  const uint simdgroup_in_band = thread_context.simdgroup_index / query_bands;
  const uint simdgroups_per_band = NUM_SIMDGROUPS / query_bands;
  const uint query_row_base = row_group_index * rows_per_group + query_band * Ops::FRAGMENT_ROWS;
  if (query_row_base >= row_count)
    return;

  const short2 position = Ops::get_position(thread_context.simd_lane_id);
  const auto load_slice_fragments = [&](const device int8_t* first_row_codes,
                                        const device int8_t* second_row_codes,
                                        ushort slice,
                                        thread uint2(&fragments)[SLICE_FRAGMENTS]) UZU_ALWAYS_INLINE {
    const uint4 first_row_words = *reinterpret_cast<const device uint4*>(
        first_row_codes + slice * int(kv_layout::SLICE_DIMS) + position.x * int(kv_layout::LANE_COLS)
    );
    const uint4 second_row_words = *reinterpret_cast<const device uint4*>(
        second_row_codes + slice * int(kv_layout::SLICE_DIMS) + position.x * int(kv_layout::LANE_COLS)
    );
    METAL_PRAGMA_UNROLL
    for (ushort slice_fragment = 0; slice_fragment < SLICE_FRAGMENTS; ++slice_fragment)
      fragments[slice_fragment] = uint2(first_row_words[slice_fragment], second_row_words[slice_fragment]);
  };
  const auto key_column_in_tile = [](ushort column) UZU_ALWAYS_INLINE {
    return (column / Ops::THREAD_ELEMENT_COLS) * Ops::FRAGMENT_COLS + column % Ops::THREAD_ELEMENT_COLS;
  };
  uint lane_rows[LANE_ROWS];
  METAL_PRAGMA_UNROLL
  for (ushort i = 0; i < LANE_ROWS; ++i)
    lane_rows[i] = query_row_base + (USE_MXU ? position.y + i * Ops::THREAD_ELEMENT_ROW_STRIDE : position.x + i);
  const auto load_head = [&](ushort head,
                             thread Queries& queries,
                             thread float (&scales)[LANE_ROWS]) UZU_ALWAYS_INLINE {
    const size_t head_rows = size_t(head) * query_rows + first_row;
    const device int8_t* first_row_codes = query_codes + (head_rows + min(lane_rows[0], row_count - 1)) * HEAD_DIM;
    const device int8_t* second_row_codes = query_codes + (head_rows + min(lane_rows[1], row_count - 1)) * HEAD_DIM;
    if constexpr (USE_MXU) {
      METAL_PRAGMA_UNROLL
      for (ushort slice = 0; slice < DEPTH_SLICES; ++slice) {
        uint2 fragments[SLICE_FRAGMENTS];
        load_slice_fragments(first_row_codes, second_row_codes, slice, fragments);
        METAL_PRAGMA_UNROLL
        for (ushort slice_fragment = 0; slice_fragment < SLICE_FRAGMENTS; ++slice_fragment)
          queries.fragment_data[slice * SLICE_FRAGMENTS + slice_fragment] = as_type<Code>(fragments[slice_fragment]);
      }
    } else {
      METAL_PRAGMA_UNROLL
      for (ushort depth = 0; depth < DEPTH_FRAGMENTS; ++depth) {
        const int offset = depth * Ops::FRAGMENT_ROWS + position.y;
        queries.fragment_data[depth] = half2(half(first_row_codes[offset]), half(second_row_codes[offset]));
      }
    }
    METAL_PRAGMA_UNROLL
    for (ushort i = 0; i < LANE_ROWS; ++i)
      scales[i] = float(query_scales[head_rows + min(lane_rows[i], row_count - 1)]);
  };
  // MXU reloads one head at a time to stay in registers; simd keeps all heads resident.
  Queries resident_queries[USE_MXU ? 1 : NUM_HEADS];
  float resident_scales[USE_MXU ? 1 : NUM_HEADS][LANE_ROWS];
  if constexpr (!USE_MXU) {
    METAL_PRAGMA_UNROLL
    for (ushort head = 0; head < NUM_HEADS; ++head)
      load_head(head, resident_queries[head], resident_scales[head]);
  }

  const uint tile_end = min((block_group_index + 1) * blocks_per_group, candidate_blocks + KEY_TILE - 1) / KEY_TILE;
  for (uint key_tile = block_group_index * blocks_per_group / KEY_TILE + simdgroup_in_band; key_tile < tile_end;
       key_tile += simdgroups_per_band) {
    const uint key_base = key_tile * KEY_TILE;
    // Load keys: MXU zips two rows, simd reads two bytes.
    Keys keys;
    METAL_PRAGMA_UNROLL
    for (ushort key_fragment = 0; key_fragment < KEY_FRAGMENTS; ++key_fragment) {
      const uint key_candidate = key_base + key_fragment * Ops::FRAGMENT_COLS + position.y;
      const device int8_t* first_row_codes = key_codes + key_slot(min(key_candidate, candidate_blocks - 1)) * HEAD_DIM;
      if constexpr (USE_MXU) {
        const device int8_t* second_row_codes =
            key_codes + key_slot(min(key_candidate + Ops::THREAD_ELEMENT_ROW_STRIDE, candidate_blocks - 1)) * HEAD_DIM;
        METAL_PRAGMA_UNROLL
        for (ushort slice = 0; slice < DEPTH_SLICES; ++slice) {
          uint2 fragments[SLICE_FRAGMENTS];
          load_slice_fragments(first_row_codes, second_row_codes, slice, fragments);
          METAL_PRAGMA_UNROLL
          for (ushort slice_fragment = 0; slice_fragment < SLICE_FRAGMENTS; ++slice_fragment)
            keys.fragment_data[key_fragment * DEPTH_FRAGMENTS + slice * SLICE_FRAGMENTS + slice_fragment] =
                as_type<Code>(fragments[slice_fragment]);
        }
      } else {
        METAL_PRAGMA_UNROLL
        for (ushort depth = 0; depth < DEPTH_FRAGMENTS; ++depth) {
          keys.fragment_data[key_fragment * DEPTH_FRAGMENTS + depth] = half2(
              as_type<char2>(
                  *reinterpret_cast<const device ushort*>(first_row_codes + depth * Ops::FRAGMENT_COLS + position.x)
              )
          );
        }
      }
    }
    Fragment<float, Dots::ROW_FRAGMENTS, Dots::COL_FRAGMENTS, Ops> head_sums;
    head_sums.clear();
    const auto add_head = [&](thread Queries& queries, thread float (&scales)[LANE_ROWS]) UZU_ALWAYS_INLINE {
      Dots dots;
      dots.clear();
      if constexpr (USE_MXU)
        fragment_mma(dots, queries, keys);
      else
        fragment_mma(dots, keys, queries);
      head_sums.for_each_lane(
          [&](ushort row, ushort column, thread float& head_sum, thread auto& dot) UZU_ALWAYS_INLINE {
            const float factor = scales[USE_MXU ? row : column];
            // factor > 0, so factor * relu(dot) = relu(factor * dot).
            head_sum = fma(factor, max(float(dot), 0.0f), head_sum);
          },
          dots
      );
    };
    if constexpr (USE_MXU) {
      METAL_PRAGMA_NO_UNROLL
      for (ushort head = 0; head < NUM_HEADS; ++head) {
        Queries queries;
        float scales[LANE_ROWS];
        load_head(head, queries, scales);
        add_head(queries, scales);
      }
    } else {
      METAL_PRAGMA_UNROLL
      for (ushort head = 0; head < NUM_HEADS; ++head)
        add_head(resident_queries[head], resident_scales[head]);
    }
    if constexpr (USE_MXU) {
      if (key_base + KEY_TILE <= cached_blocks) {
        float key_scale_values[KEY_FRAGMENTS * Ops::THREAD_ELEMENT_COLS];
        METAL_PRAGMA_UNROLL
        for (ushort column = 0; column < KEY_FRAGMENTS * Ops::THREAD_ELEMENT_COLS; ++column)
          key_scale_values[column] =
              float(key_scales[cached_block_row + key_base + position.x + key_column_in_tile(column)]);
        device half* score_tile = scores + size_t(lane_rows[0]) * candidate_blocks + key_base + position.x;
        const size_t score_row_stride = size_t(Ops::THREAD_ELEMENT_ROW_STRIDE) * candidate_blocks;
        head_sums.for_each_lane([&](ushort row, ushort column, thread float& head_sum) UZU_ALWAYS_INLINE {
          if (lane_rows[row] < row_count)
            score_tile[row * score_row_stride + key_column_in_tile(column)] = half(head_sum * key_scale_values[column]);
        });
        continue;
      }
    }
    head_sums.for_each_lane([&](ushort row, ushort column, thread float& head_sum) UZU_ALWAYS_INLINE {
      const uint key = key_base + (USE_MXU ? position.x + key_column_in_tile(column) : position.y);
      const uint query_row = lane_rows[USE_MXU ? row : column];
      if (query_row >= row_count || key >= candidate_blocks)
        return;
      const bool visible = key < (prefix_tokens + first_row + query_row + 1) / kv_layout::COMPRESSION_RATIO;
      scores[size_t(query_row) * candidate_blocks + key] =
          half(visible ? head_sum * float(key_scales[key_slot(key)]) : -INFINITY);
    });
  }
}
