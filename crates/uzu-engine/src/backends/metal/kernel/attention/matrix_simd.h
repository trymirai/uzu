#pragma once

#include "../matmul/common/fragment.h"
#include "../matmul/common/simdgroup_fragment_ops.h"
#include "kv_layout.h"
#include "mask.h"
#include "matrix_layout.h"
#include "rotated_output.h"

using namespace uzu::matmul;

// Vectors a thread loads at once; a sparse thread's are adjacent and share one key row.
constexpr constant uint GATHER_GROUP = 2;

template <uint HEAD_DIM, uint BK, uint NUM_SIMDGROUPS, bool IS_SPARSE, typename Code>
METAL_FUNC void gather_vectors(
    const device Code* codes,
    uzu::kv_layout::KVShape shape,
    const threadgroup int* rows,
    uint first,
    thread uint4 (&words)[GATHER_GROUP]
) {
  constexpr uint VECTORS_PER_ROW = HEAD_DIM / 16;
  constexpr uint VECTORS = BK * VECTORS_PER_ROW;
  constexpr uint SLOT = IS_SPARSE ? 1 : NUM_SIMDGROUPS * METAL_SIMD_SIZE;
  const int first_row = first < VECTORS ? rows[first / VECTORS_PER_ROW] : uzu::kv_layout::NO_KEY_SLOT;
  METAL_PRAGMA_UNROLL
  for (uint g = 0; g < GATHER_GROUP; ++g) {
    const uint vector = first + g * SLOT;
    const int row = IS_SPARSE || g == 0
                        ? first_row
                        : (vector < VECTORS ? rows[vector / VECTORS_PER_ROW] : uzu::kv_layout::NO_KEY_SLOT);
    words[g] = row >= 0 ? *reinterpret_cast<const device uint4*>(
                              codes + shape.row_code_index(uint(row), vector % VECTORS_PER_ROW * 16)
                          )
                        : uint4(0);
  }
}

template <uint HEAD_DIM, uint BK, bool IS_VALUE, uint NUM_SIMDGROUPS, bool IS_SPARSE, typename Code>
METAL_FUNC void stage_int8_tile(
    const device Code* codes,
    threadgroup half* tile,
    uzu::kv_layout::KVShape shape,
    uint first_vector,
    const threadgroup int* rows,
    thread uint4 (&words)[GATHER_GROUP]
) {
  constexpr uint ROW_STRIDE = HEAD_DIM + 8;
  constexpr uint VECTORS_PER_ROW = HEAD_DIM / 16;
  constexpr uint VECTORS = BK * VECTORS_PER_ROW;
  constexpr uint THREADS = NUM_SIMDGROUPS * METAL_SIMD_SIZE;
  constexpr uint SLOT = IS_SPARSE ? 1 : THREADS;
  METAL_PRAGMA_NO_UNROLL
  for (uint first = first_vector; first < VECTORS; first += GATHER_GROUP * THREADS) {
    if (first != first_vector)
      gather_vectors<HEAD_DIM, BK, NUM_SIMDGROUPS, IS_SPARSE>(codes, shape, rows, first, words);
    METAL_PRAGMA_UNROLL
    for (uint g = 0; g < GATHER_GROUP; ++g) {
      const uint vector = first + g * SLOT;
      if (vector < VECTORS) {
        if constexpr (IS_VALUE)
          words[g] ^= uint4(0x80808080u);
        METAL_PRAGMA_UNROLL
        for (ushort chunk = 0; chunk < 4; ++chunk) {
          *reinterpret_cast<threadgroup half4*>(
              tile + vector / VECTORS_PER_ROW * ROW_STRIDE + vector % VECTORS_PER_ROW * 16 + chunk * 4
          ) = half4(as_type<char4>(words[g][chunk]));
        }
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

template <typename ScaleType, uint HEAD_DIM, uint NUM_SIMDGROUPS, bool IS_SPARSE>
METAL_FUNC void attention_matrix_simd(
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
    device bfloat* output_rows,
    const device bfloat* gate,
    uint gate_row_stride,
    const device bfloat* sinks,
    const constant uzu::attention::AttnParams& params,
    uint query_tiles,
    uint num_splits,
    uint trie_nodes,
    bool is_trie,
    bool is_split,
    bool has_gate,
    bool has_sinks,
    const device int* selected_blocks,
    uint blocks_per_query,
    uint compression_ratio,
    uint num_kv_heads,
    threadgroup float* exchange,
    threadgroup half* kv_tile,
    threadgroup ulong* row_masks,
    ThreadContext thread_context,
    uint tile_split_index,
    uint kv_head_index
) {
  using Scratch = AttentionMatrixScratch<HEAD_DIM, false, NUM_SIMDGROUPS, IS_SPARSE>;
  constexpr uint BK = Scratch::SIMD_KEYS;
  constexpr uint SLICES = Scratch::SIMD_SLICES;
  constexpr uint BLOCK_ROWS = NUM_SIMDGROUPS / SLICES * 8;
  using Layout = AttentionMatrixLayout<BK, HEAD_DIM, SLICES, false>;
  using Ops = SimdgroupFragmentOps;
  constexpr uint THREADS = NUM_SIMDGROUPS * METAL_SIMD_SIZE;
  static_assert(
      !IS_SPARSE || BK * HEAD_DIM / 16 <= GATHER_GROUP * THREADS,
      "the sparse tile is one group of vectors per thread"
  );
  constexpr uint DEPTH_FRAGMENTS = Layout::SLICE_COLS / Ops::FRAGMENT_ROWS;
  constexpr uint KEY_FRAGMENTS = BK / Ops::FRAGMENT_COLS;
  constexpr uint VALUE_FRAGMENTS = Layout::SLICE_COLS / Ops::FRAGMENT_COLS;
  using Query = Fragment<half, 1, 1, Ops>;
  using Keys = Fragment<half, 1, KEY_FRAGMENTS, Ops, ReadTranspose>;
  using Scores = Fragment<float, 1, KEY_FRAGMENTS, Ops>;
  using Value = Fragment<half, 1, 1, Ops>;
  using Output = Fragment<float, 1, VALUE_FRAGMENTS, Ops>;

  const ushort lane = ushort(thread_context.simd_lane_id);
  const uint simdgroup = thread_context.simdgroup_index;
  const uint band = simdgroup / SLICES;
  const uint slice = simdgroup % SLICES;
  const uint query_tile = tile_split_index % query_tiles;
  const uint split = tile_split_index / query_tiles;
  const uint grouped_rows = params.gqa_factor * params.q_len;
  // A sparse tile is one query position's heads, q_len rows apart.
  const uint query_row_step = IS_SPARSE ? params.q_len : 1u;
  const uint row_base = IS_SPARSE ? query_tile + band * Ops::FRAGMENT_ROWS * params.q_len
                                  : query_tile * BLOCK_ROWS + band * Ops::FRAGMENT_ROWS;
  const uint last_query_row = IS_SPARSE ? (params.gqa_factor - 1u) * params.q_len + query_tile : grouped_rows - 1u;
  const uint channel_base = slice * Layout::SLICE_COLS;
  const short2 position = Ops::get_position(lane);
  const uint query_row = min(row_base + uint(position.y) * query_row_step, last_query_row);
  const uint query_position = query_row % params.q_len;
  const size_t query_group = size_t(kv_head_index) * grouped_rows;
  const float query_scale = float(query_scales[query_group + query_row]) * params.scale * M_LOG2E_F;
  const uzu::kv_layout::KVShape shape{HEAD_DIM, num_kv_heads, uzu::kv_layout::KVLayout::RowMajor};

  Query queries[DEPTH_FRAGMENTS];
  METAL_PRAGMA_UNROLL
  for (ushort depth = 0; depth < DEPTH_FRAGMENTS; ++depth) {
    queries[depth].fragment_at(0, 0) = half2(
        float2(*reinterpret_cast<const device char2*>(
            query_codes + (query_group + query_row) * HEAD_DIM + channel_base + depth * Ops::FRAGMENT_COLS + position.x
        )) *
        query_scale
    );
  }

  const SparseKeys sparse_keys(
      selected_blocks,
      blocks_per_query,
      compression_ratio,
      trie,
      trie_nodes,
      params.q_off,
      query_tile,
      IS_SPARSE && is_trie
  );
  const uint prefix_length = params.q_off;
  const uint ring_offset = is_kv_cache_ring ? ring_params.ring_offset : 0u;
  const KeyWindow window = KeyWindow::make(is_causal, is_sliding_window, sliding_window_size);
  const uint tile_first = query_tile * BLOCK_ROWS;
  const uint tile_last = min(tile_first + BLOCK_ROWS - 1u, grouped_rows - 1u);
  const VisibleKeyBlocks visible = VisibleKeyBlocks::make<BK>(
      window,
      is_causal,
      prefix_length,
      ring_offset,
      IS_SPARSE || !(is_kv_cache_ring || is_sliding_window),
      tile_query_range(tile_first, tile_last, params.q_len, is_trie)
  );
  const uint total =
      IS_SPARSE ? div_ceil(sparse_keys.key_count(), BK) : visible.count_until(div_ceil(params.k_len, BK));
  const uint2 split_blocks = visible.split(total, split, num_splits);
  const uint oldest_visible =
      window.oldest_visible(prefix_length, is_trie && is_sliding_window ? trie[query_position].height : query_position);
  Output output;
  output.clear();
  float running_max = -INFINITY;
  float denominator = 0.0f;
  threadgroup float* score_slot = exchange + simdgroup * Layout::SCORE_TILE_ELEMENTS;
  threadgroup int* chunk_rows = reinterpret_cast<threadgroup int*>(row_masks);
  threadgroup float* chunk_key_scales = reinterpret_cast<threadgroup float*>(chunk_rows + Scratch::CHUNK_KEYS);
  threadgroup float* chunk_value_scales = chunk_key_scales + Scratch::CHUNK_KEYS;

  for (uint block = split_blocks.x; block < split_blocks.y; ++block) {
    const uint key_base = visible.block_at(block) * BK;
    const uint thread_index = simdgroup * METAL_SIMD_SIZE + lane;
    const uint first_vector = thread_index * (IS_SPARSE ? GATHER_GROUP : 1);
    const uint chunk_offset = (block - split_blocks.x) % (Scratch::CHUNK_KEYS / BK) * BK;
    if (chunk_offset == 0) {
      for (uint k = thread_index; k < Scratch::CHUNK_KEYS; k += THREADS) {
        const uint chunk_block = block + k / BK;
        const uint key = visible.block_at(chunk_block) * BK + k % BK;
        uint slot = uzu::kv_layout::NO_KEY_SLOT;
        if (chunk_block < split_blocks.y) {
          slot = IS_SPARSE ? sparse_keys.slot(key) : key;
        }
        const int row = slot < params.k_len ? int(shape.scale_index(slot, kv_head_index)) : uzu::kv_layout::NO_KEY_SLOT;
        chunk_rows[k] = row;
        chunk_key_scales[k] = row >= 0 ? float(key_scales[row]) : 0.0f;
        chunk_value_scales[k] = row >= 0 ? float(value_scales[row]) : 0.0f;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint4 words[GATHER_GROUP];
    gather_vectors<HEAD_DIM, BK, NUM_SIMDGROUPS, IS_SPARSE>(
        key_codes,
        shape,
        chunk_rows + chunk_offset,
        first_vector,
        words
    );
    stage_int8_tile<HEAD_DIM, BK, false, NUM_SIMDGROUPS, IS_SPARSE>(
        key_codes,
        kv_tile,
        shape,
        first_vector,
        chunk_rows + chunk_offset,
        words
    );
    // Load the value tile's first vectors now so their latency hides behind the scores.
    gather_vectors<HEAD_DIM, BK, NUM_SIMDGROUPS, IS_SPARSE>(
        value_codes,
        shape,
        chunk_rows + chunk_offset,
        first_vector,
        words
    );
    Scores scores;
    scores.clear();
    METAL_PRAGMA_UNROLL
    for (ushort depth = 0; depth < DEPTH_FRAGMENTS; ++depth) {
      Keys keys;
      keys.load_from(lane, fragment_source(kv_tile + channel_base + depth * Ops::FRAGMENT_ROWS, int(HEAD_DIM + 8)));
      fragment_mma(scores, queries[depth], keys);
    }
    if constexpr (SLICES > 1) {
      scores.store(lane, score_slot, int(BK));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    METAL_PRAGMA_UNROLL
    for (ushort other = 1; other < SLICES; ++other) {
      Scores partial;
      partial.load_from(
          lane,
          fragment_source(exchange + (band * SLICES + (slice + other) % SLICES) * Layout::SCORE_TILE_ELEMENTS, int(BK))
      );
      scores.for_each_lane([&](ushort, ushort, thread float& score, thread float& value) { score += value; }, partial);
    }
    stage_int8_tile<HEAD_DIM, BK, true, NUM_SIMDGROUPS, IS_SPARSE>(
        value_codes,
        kv_tile,
        shape,
        first_vector,
        chunk_rows + chunk_offset,
        words
    );
    scores.map_coords(lane, [&](short, short col, float score) {
      return score * chunk_key_scales[chunk_offset + col];
    });
    if (IS_SPARSE || is_sliding_window || key_base + BK > prefix_length) {
      scores.for_each_lane([&](ushort, ushort lane_col, thread float& score) {
        const uint col = lane_col / Ops::THREAD_ELEMENT_COLS * Ops::FRAGMENT_COLS + uint(position.x) +
                         lane_col % Ops::THREAD_ELEMENT_COLS;
        bool visible;
        if constexpr (IS_SPARSE) {
          visible = chunk_rows[chunk_offset + col] >= 0;
        } else {
          const uint key = key_base + col;
          uint key_position = position_of_key(key, params.k_len, prefix_length, ring_offset);
          RowRange seeing_rows = RowRange::all();
          if (is_trie && key >= prefix_length && key < params.k_len) {
            const auto node = trie[key - prefix_length];
            seeing_rows = RowRange::make(node.trie_start, node.trie_end);
            if (is_sliding_window) {
              key_position = prefix_length + node.height;
            }
          }
          visible = window.sees(oldest_visible, key_position) &&
                    (!(is_trie && is_causal) || seeing_rows.contains(query_position));
        }
        score = visible ? score : -INFINITY;
      });
    }
    float block_max;
    scores.row_reduce(&block_max, -INFINITY, [](float a, float b) { return metal::max(a, b); });
    const float next_max = metal::max(running_max, block_max);
    const float rescale = isfinite(running_max) ? fast::exp2(running_max - next_max) : 0.0f;
    scores.map([&](float score) { return isfinite(next_max) ? fast::exp2(score - next_max) : 0.0f; });
    float block_sum;
    scores.row_reduce(&block_sum, 0.0f, [](float a, float b) { return a + b; });
    denominator = denominator * rescale + block_sum;
    running_max = next_max;
    output.map([&](float value) { return value * rescale; });
    scores.map_coords(lane, [&](short, short col, float score) {
      return score * chunk_value_scales[chunk_offset + col];
    });

    METAL_PRAGMA_UNROLL
    for (ushort key_fragment = 0; key_fragment < KEY_FRAGMENTS; ++key_fragment) {
      METAL_PRAGMA_UNROLL
      for (ushort col = 0; col < VALUE_FRAGMENTS; ++col) {
        Value value;
        value.load_from(
            lane,
            fragment_source(
                kv_tile + key_fragment * Ops::FRAGMENT_ROWS * (HEAD_DIM + 8) + channel_base + col * Ops::FRAGMENT_COLS,
                int(HEAD_DIM + 8)
            )
        );
        SimdgroupMMA<float, 8, 8, float, half>::mma(
            output.fragment_at(0, col),
            scores.fragment_at(0, key_fragment),
            value.fragment_at(0, 0),
            output.fragment_at(0, col)
        );
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if constexpr (SLICES == 1) {
    if (!IS_SPARSE && !is_split) {
      threadgroup bfloat* tile_values = reinterpret_cast<threadgroup bfloat*>(kv_tile);
      threadgroup float2* tile_stats = reinterpret_cast<threadgroup float2*>(row_masks);
      output.store(lane, tile_values + band * Ops::FRAGMENT_ROWS * HEAD_DIM, int(HEAD_DIM));
      if (position.x == 0) {
        tile_stats[band * Ops::FRAGMENT_ROWS + position.y] = float2(running_max, denominator);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      constexpr ushort VALUES_PER_LANE = HEAD_DIM / METAL_SIMD_SIZE;
      for (uint row = simdgroup; row < BLOCK_ROWS; row += NUM_SIMDGROUPS) {
        const uint grouped_row = tile_first + row;
        if (grouped_row >= grouped_rows) {
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
            output_rows,
            gate,
            params,
            kv_head_index,
            grouped_row,
            gate_row_stride,
            has_gate
        );
      }
      return;
    }
  }
  const uint partial_row = (split * num_kv_heads + kv_head_index) * query_tiles * BLOCK_ROWS + row_base;
  output.store(lane, partial_values + size_t(partial_row) * HEAD_DIM + channel_base, int(query_row_step * HEAD_DIM));
  if (slice == 0 && position.x == 0) {
    partial_maxima[partial_row + position.y * query_row_step] = running_max;
    partial_denominators[partial_row + position.y * query_row_step] = denominator;
  }
}
