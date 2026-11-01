#pragma once

#include "../matmul/common/fragment.h"
#include "../matmul/common/simdgroup_fragment_ops.h"
#include "kv_layout.h"
#include "mask.h"
#include "layout.h"

template <uint HEAD_DIM, uint BK, bool IS_VALUE, uint NUM_SIMDGROUPS, typename Code>
METAL_FUNC void stage_int8_tile(
    const device Code* codes,
    threadgroup half* tile,
    uzu::kv_layout::Int8KVShape shape,
    uint key_base,
    uint key_length,
    uint kv_head,
    uint thread_index
) {
  constexpr uint ROW_STRIDE = HEAD_DIM + 8;
  constexpr uint VECTORS = BK * HEAD_DIM / 16;
  const device uint4* rows = reinterpret_cast<const device uint4*>(codes + shape.k_code_index(key_base, kv_head, 0));
  for (uint vector = thread_index; vector < VECTORS; vector += NUM_SIMDGROUPS * METAL_SIMD_SIZE) {
    const uint row = vector / (HEAD_DIM / 16);
    const uint channel = vector % (HEAD_DIM / 16) * 16;
    uint4 words = key_base + row < key_length ? rows[vector] : uint4(0);
    if constexpr (IS_VALUE)
      words ^= uint4(0x80808080u);
    METAL_PRAGMA_UNROLL
    for (ushort chunk = 0; chunk < 4; ++chunk) {
      const half4 values = key_base + row < key_length ? half4(as_type<char4>(words[chunk])) : half4(0);
      *reinterpret_cast<threadgroup half4*>(tile + row * ROW_STRIDE + channel + chunk * 4) = values;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

template <typename ScaleType, uint HEAD_DIM, uint NUM_SIMDGROUPS>
METAL_FUNC void attention_int8_simd(
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
    const constant uzu::attention::AttnParams& params,
    uint query_tiles,
    uint num_splits,
    uint trie_nodes,
    bool is_trie,
    uint num_kv_heads,
    threadgroup float* exchange,
    threadgroup half* kv_tile,
    ThreadContext thread_context,
    uint tile_split_index,
    uint kv_head_index
) {
  using namespace uzu::matmul;
  constexpr uint BK = HEAD_DIM == 512 ? 16 : 32;
  constexpr uint SLICES = HEAD_DIM < 128 ? 1 : HEAD_DIM / 128;
  constexpr uint BLOCK_ROWS = NUM_SIMDGROUPS / SLICES * 8;
  using Layout = AttentionGemmGroupedLayout<BK, HEAD_DIM, SLICES, false>;
  using Ops = SimdgroupFragmentOps;
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
  const uint row_base = query_tile * BLOCK_ROWS + band * Ops::FRAGMENT_ROWS;
  const uint channel_base = slice * Layout::SLICE_COLS;
  const short2 position = Ops::get_position(lane);
  const uint query_row = min(row_base + uint(position.y), grouped_rows - 1u);
  const uint query_position = query_row % params.q_len;
  const size_t query_group = size_t(kv_head_index) * grouped_rows;
  const float query_scale = float(query_scales[query_group + query_row]) * params.scale * M_LOG2E_F;
  const uzu::kv_layout::Int8KVShape shape{HEAD_DIM, num_kv_heads, false};

  Query queries[DEPTH_FRAGMENTS];
  METAL_PRAGMA_UNROLL
  for (ushort depth = 0; depth < DEPTH_FRAGMENTS; ++depth) {
    queries[depth].clear();
    queries[depth].map_coords(lane, [&](short row, short col, half) {
      const uint q = min(row_base + uint(row), grouped_rows - 1u);
      return half(query_codes[(query_group + q) * HEAD_DIM + channel_base + depth * Ops::FRAGMENT_COLS + col]);
    });
  }

  ulong trie_mask = 0ul;
  if (is_trie) {
    for (uint node_index = 0; node_index < min(trie_nodes, 64u); ++node_index) {
      const auto node = trie[node_index];
      if (query_position >= node.trie_start && query_position <= node.trie_end) {
        trie_mask |= 1ul << node_index;
      }
    }
  }

  const uint num_blocks = (params.k_len + BK - 1) / BK;
  const uint block_begin = split * num_blocks / num_splits;
  const uint suffix_position = is_kv_cache_ring ? ring_params.ring_length : params.q_off;
  const uint logical_query = suffix_position + (is_trie ? trie[query_position].height : query_position);
  // Trie indices are preorder: an ancestor never follows its descendant.
  const uint tile_first = query_tile * BLOCK_ROWS;
  const uint tile_last = min(tile_first + BLOCK_ROWS - 1u, grouped_rows - 1u);
  const uint last_query =
      tile_first / params.q_len == tile_last / params.q_len ? tile_last % params.q_len : params.q_len - 1u;
  const uint block_end = is_causal ? min((split + 1u) * num_blocks / num_splits, (params.q_off + last_query + BK) / BK)
                                   : (split + 1u) * num_blocks / num_splits;
  Output output;
  output.clear();
  float running_max = -INFINITY;
  float denominator = 0.0f;
  threadgroup float* score_slot = exchange + simdgroup * Layout::SCORE_TILE_ELEMENTS;

  for (uint block = block_begin; block < block_end; ++block) {
    const uint key_base = block * BK;
    stage_int8_tile<HEAD_DIM, BK, false, NUM_SIMDGROUPS>(
        key_codes,
        kv_tile,
        shape,
        key_base,
        params.k_len,
        kv_head_index,
        simdgroup * METAL_SIMD_SIZE + lane
    );
    Scores scores;
    scores.clear();
    METAL_PRAGMA_UNROLL
    for (ushort depth = 0; depth < DEPTH_FRAGMENTS; ++depth) {
      Keys keys;
      keys.load_from(lane, fragment_source(kv_tile + channel_base + depth * Ops::FRAGMENT_ROWS, int(HEAD_DIM + 8)));
      fragment_mma(scores, queries[depth], keys);
    }
    // Sum the head-dimension slices. All bands share the exchange allocation.
    scores.store(lane, score_slot, int(BK));
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
    stage_int8_tile<HEAD_DIM, BK, true, NUM_SIMDGROUPS>(
        value_codes,
        kv_tile,
        shape,
        key_base,
        params.k_len,
        kv_head_index,
        simdgroup * METAL_SIMD_SIZE + lane
    );
    // The prefix is visible to every query. Keep its tiles on a uniform path
    // without per-element causal/trie checks, including in the final split.
    if (!is_kv_cache_ring && !is_sliding_window && key_base + BK <= params.q_off) {
      scores.map_coords(lane, [&](short, short col, float score) {
        return score * query_scale * float(key_scales[shape.scale_index(key_base + uint(col), kv_head_index)]);
      });
    } else {
      scores.map_coords(lane, [&](short, short col, float score) {
        const uint key = key_base + uint(col);
        bool visible = key < params.k_len;
        if (visible) {
          if (!is_kv_cache_ring && !is_sliding_window && is_causal && trie_nodes <= 64) {
            if (key >= params.q_off) {
              visible =
                  is_trie ? ((trie_mask >> (key - params.q_off)) & 1ul) != 0ul : key <= params.q_off + query_position;
            }
          } else {
            visible = should_use_key(
                ring_params,
                trie,
                sliding_window_size,
                query_position,
                params.q_off,
                suffix_position,
                logical_query,
                key,
                is_kv_cache_ring,
                is_causal,
                is_trie,
                is_sliding_window
            );
          }
        }
        return visible ? score * query_scale * float(key_scales[shape.scale_index(key, kv_head_index)]) : -INFINITY;
      });
    }
    float block_max[1];
    scores.row_reduce(block_max, -INFINITY, [](float a, float b) { return metal::max(a, b); });
    const float next_max = metal::max(running_max, block_max[0]);
    // An entirely masked split contributes zero, including at prefix zero.
    const float rescale = isfinite(running_max) ? fast::exp2(running_max - next_max) : 0.0f;
    scores.map([&](float score) { return isfinite(next_max) ? fast::exp2(score - next_max) : 0.0f; });
    float block_sum[1];
    scores.row_reduce(block_sum, 0.0f, [](float a, float b) { return a + b; });
    denominator = denominator * rescale + block_sum[0];
    running_max = next_max;
    output.map([&](float value) { return value * rescale; });

    // Keep scaling in float: BF16 KV scales need not fit in half precision.
    scores.map_coords(lane, [&](short, short col, float probability) {
      const uint key = key_base + uint(col);
      return key < params.k_len ? probability * float(value_scales[shape.scale_index(key, kv_head_index)]) : 0.0f;
    });
    METAL_PRAGMA_UNROLL
    for (ushort key_fragment = 0; key_fragment < KEY_FRAGMENTS; ++key_fragment) {
      // Interleave independent output accumulators instead of issuing all
      // dependent updates for one output panel consecutively.
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

  const float inverse_sum = denominator > 0.0f ? 1.0f / denominator : 0.0f;
  output.map([&](float value) { return value * inverse_sum; });
  const uint partial_row = (split * num_kv_heads + kv_head_index) * query_tiles * BLOCK_ROWS + row_base;
  output.store(lane, partial_values + size_t(partial_row) * HEAD_DIM + channel_base, int(HEAD_DIM));
  if (slice == 0 && position.x == 0) {
    partial_maxima[partial_row + position.y] = running_max;
    partial_denominators[partial_row + position.y] = denominator;
  }
}
