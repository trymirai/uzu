#include <metal_stdlib>
#include <metal_simdgroup_matrix>
#include "../common/dsl.h"
#include "confidence_head.h"

using namespace metal;

#define CH_HEAD_DIM_MAX 64u
// 8 x 8 tiles of a head's dimensions.
#define CH_HEAD_TILES (CH_HEAD_DIM_MAX / 8u)
// A threadgroup takes a tile of eight queries; its simdgroups split the keys (blocks of 8 x CH_KEY_TILES keys, dealt
// round-robin) and merge their partial softmaxes at the end.
#define CH_KEY_SPLIT 4u
#define CH_ATTENTION_THREADS (CH_KEY_SPLIT * METAL_SIMD_SIZE)
// Key tiles of eight between two updates of the running softmax.
#define CH_KEY_TILES 4u
#define CH_RELATIONS 6u

// The cells of an 8 x 8 simdgroup matrix a lane holds in thread_elements(): one row and two adjacent columns, the first
// of them returned (Apple's layout, as in uzu's matmul SimdgroupMMA). A row's eight columns sit on the lanes l, l ^ 1,
// l ^ 8, l ^ 9.
METAL_FUNC ushort2 confidence_head_matrix_cell(ushort lane) {
  const ushort quad = lane / 4;
  return ushort2((quad & 4) + ((lane / 2) % 4), (quad & 2) * 2 + (lane % 2) * 2);
}

// The two cells a lane holds, as uzu's matmul reads them.
METAL_FUNC thread float2& confidence_head_cells(thread simdgroup_float8x8& matrix) {
  return reinterpret_cast<thread float2&>(matrix.thread_elements());
}

// Attention of the head's tree layers: every token attends to every present token with a learned bias per head and
// pair relation (self, parent, child, ancestor two or more edges up, descendant likewise, sibling); an absent query
// reads the present keys without biases. A simdgroup takes eight queries of one head and goes over the keys in tiles of
// eight: Q K^T and P V are 8 x 8 matrix products, the biases, the mask and the softmax act on the score tile's cells in
// the lanes' registers (online over blocks of CH_KEY_TILES tiles, the output rescaled as the maximum grows). The
// threadgroup's simdgroups take every CH_KEY_SPLIT-th block and merge their maxima, totals and outputs at the end. Tiles
// past the last token are read as zeros. The CPU twin is the reference.
PUBLIC KERNEL(ConfidenceHeadTreeAttention)(
    const device float* qkv,
    const device float* relation_weights,
    const device uint* token_parents,
    const device uint* token_depths,
    const device uint* token_present,
    const device uint* token_ancestors,
    device float* output,
    constant uint& token_count,
    const uint heads SPECIALIZE,
    const uint head_dim SPECIALIZE,
    const uint relations SPECIALIZE,
    const uint max_depth SPECIALIZE,
    threadgroup float partial_max[CH_KEY_SPLIT * 8],
    threadgroup float partial_total[CH_KEY_SPLIT * 8],
    threadgroup float partial_output[CH_KEY_SPLIT * 8 * CH_HEAD_DIM_MAX],
    const ThreadContext thread_context,
    const uint group GROUPS(heads * token_count.div_ceil(8)),
    const uint lid THREADS(CH_ATTENTION_THREADS)
) {
  const uint tiles = (token_count + 7u) / 8u;
  const uint head = group / tiles;
  const uint first = (group % tiles) * 8u;
  const uint split = thread_context.simdgroup_index;
  const uint width = heads * head_dim, levels = max_depth + 1u, stride = 3u * width;
  const float scale = 1.0f / sqrt(float(head_dim));
  const ushort2 cell = confidence_head_matrix_cell(ushort(thread_context.simd_lane_id));
  float relation_weight[CH_RELATIONS];
  for (uint relation = 0; relation < CH_RELATIONS; ++relation) {
    relation_weight[relation] = relation < relations ? relation_weights[head * relations + relation] : 0.0f;
  }

  // This lane's query: the row of its cells in every tile.
  const uint q = first + cell.x;
  const bool query_valid = q < token_count;
  const bool query_present = query_valid && token_present[q] != 0u;
  const uint parent = query_valid ? token_parents[q] : CH_NONE;
  const uint depth = query_valid ? token_depths[q] : 0u;
  simdgroup_float8x8 query_tiles[CH_HEAD_TILES];
  METAL_PRAGMA_UNROLL
  for (uint tile = 0; tile < CH_HEAD_TILES; ++tile) {
    const uint column = head * head_dim + 8u * tile + cell.y;
    confidence_head_cells(query_tiles[tile]) =
        query_valid && 8u * tile < head_dim ? *reinterpret_cast<const device float2*>(qkv + q * stride + column)
                                            : float2(0.0f);
  }

  simdgroup_float8x8 accumulated[CH_HEAD_TILES];
  METAL_PRAGMA_UNROLL
  for (uint tile = 0; tile < CH_HEAD_TILES; ++tile) {
    accumulated[tile] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  }
  float running_max = -INFINITY, running_total = 0.0f;
  for (uint block = split * 8u * CH_KEY_TILES; block < token_count; block += CH_KEY_SPLIT * 8u * CH_KEY_TILES) {
    // Scores of the block's key tiles: Q K^T, with K^T read cell by cell (row = dimension, column = key).
    simdgroup_float8x8 scores[CH_KEY_TILES];
    float2 cells[CH_KEY_TILES];
    float block_max = -INFINITY;
    METAL_PRAGMA_UNROLL
    for (uint key_tile = 0; key_tile < CH_KEY_TILES; ++key_tile) {
      const uint keys = block + 8u * key_tile;
      scores[key_tile] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
      if (keys < token_count) {
        const uint key_first = keys + cell.y;
        METAL_PRAGMA_UNROLL
        for (uint tile = 0; tile < CH_HEAD_TILES; ++tile) {
          if (8u * tile < head_dim) {
            const uint column = width + head * head_dim + 8u * tile + cell.x;
            simdgroup_float8x8 key_tile_matrix;
            confidence_head_cells(key_tile_matrix) = float2(
                key_first < token_count ? qkv[key_first * stride + column] : 0.0f,
                key_first + 1u < token_count ? qkv[(key_first + 1u) * stride + column] : 0.0f
            );
            simdgroup_multiply_accumulate(scores[key_tile], query_tiles[tile], key_tile_matrix, scores[key_tile]);
          }
        }
      }
      // The biases and the mask on this lane's two cells: query q, keys key_first and key_first + 1.
      const float2 raw = confidence_head_cells(scores[key_tile]);
      METAL_PRAGMA_UNROLL
      for (uint element = 0; element < 2u; ++element) {
        const uint key = keys + cell.y + element;
        const bool key_present = key < token_count && token_present[key] != 0u;
        float score = key_present ? raw[element] * scale : -INFINITY;
        if (key_present && query_present) {
          const uint key_parent = token_parents[key], key_depth = token_depths[key];
          const bool ancestor =
              key_depth + 2u <= depth && token_ancestors[q * levels + min(key_depth, levels - 1u)] == key;
          const bool descendant =
              depth + 2u <= key_depth && token_ancestors[key * levels + min(depth, levels - 1u)] == q;
          score += select(0.0f, relation_weight[0], q == key) + select(0.0f, relation_weight[1], parent == key) +
                   select(0.0f, relation_weight[2], key_parent == q) + select(0.0f, relation_weight[3], ancestor) +
                   select(0.0f, relation_weight[4], descendant) +
                   select(0.0f, relation_weight[5], parent != CH_NONE && parent == key_parent && q != key);
        }
        cells[key_tile][element] = score;
        block_max = max(block_max, score);
      }
    }
    // Online softmax over the row's four lanes: rescale what was accumulated under the old maximum. A block without
    // present keys changes nothing.
    block_max = max(block_max, simd_shuffle_xor(block_max, ushort(1)));
    block_max = max(block_max, simd_shuffle_xor(block_max, ushort(8)));
    const float new_max = max(running_max, block_max);
    const float rescale = running_max == -INFINITY ? 0.0f : exp(running_max - new_max);
    float block_total = 0.0f;
    METAL_PRAGMA_UNROLL
    for (uint key_tile = 0; key_tile < CH_KEY_TILES; ++key_tile) {
      const float2 weights = select(exp(cells[key_tile] - new_max), float2(0.0f), cells[key_tile] == -INFINITY);
      confidence_head_cells(scores[key_tile]) = weights;
      block_total += weights.x + weights.y;
    }
    block_total += simd_shuffle_xor(block_total, ushort(1));
    block_total += simd_shuffle_xor(block_total, ushort(8));
    running_total = running_total * rescale + block_total;
    running_max = new_max;
    METAL_PRAGMA_UNROLL
    for (uint tile = 0; tile < CH_HEAD_TILES; ++tile) {
      confidence_head_cells(accumulated[tile]) *= rescale;
    }
    // P V: V read cell by cell (row = key, column = dimension); keys past the last token are zeros.
    METAL_PRAGMA_UNROLL
    for (uint key_tile = 0; key_tile < CH_KEY_TILES; ++key_tile) {
      const uint key = block + 8u * key_tile + cell.x;
      if (block + 8u * key_tile < token_count) {
        METAL_PRAGMA_UNROLL
        for (uint tile = 0; tile < CH_HEAD_TILES; ++tile) {
          if (8u * tile < head_dim) {
            simdgroup_float8x8 value_tile;
            confidence_head_cells(value_tile) =
                key < token_count
                    ? *reinterpret_cast<const device float2*>(
                          qkv + key * stride + 2u * width + head * head_dim + 8u * tile + cell.y
                      )
                    : float2(0.0f);
            simdgroup_multiply_accumulate(accumulated[tile], scores[key_tile], value_tile, accumulated[tile]);
          }
        }
      }
    }
  }
  // This simdgroup's partial softmax: its row's maximum and total (one lane of the row writes them) and its cells.
  if (cell.y == 0u) {
    partial_max[split * 8u + cell.x] = running_max;
    partial_total[split * 8u + cell.x] = running_total;
  }
  METAL_PRAGMA_UNROLL
  for (uint tile = 0; tile < CH_HEAD_TILES; ++tile) {
    if (8u * tile < head_dim) {
      *reinterpret_cast<threadgroup float2*>(
          partial_output + (split * 8u + cell.x) * CH_HEAD_DIM_MAX + 8u * tile + cell.y
      ) = confidence_head_cells(accumulated[tile]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // The merge: each thread takes output cells of the tile, rescaling every split by exp(its maximum - the maximum).
  for (uint position = lid; position < 8u * head_dim; position += CH_ATTENTION_THREADS) {
    const uint row = position / head_dim, dimension = position % head_dim;
    if (first + row >= token_count) {
      continue;
    }
    float maximum = -INFINITY;
    for (uint part = 0; part < CH_KEY_SPLIT; ++part) {
      maximum = max(maximum, partial_max[part * 8u + row]);
    }
    float total = 0.0f, value = 0.0f;
    for (uint part = 0; part < CH_KEY_SPLIT; ++part) {
      const float part_max = partial_max[part * 8u + row];
      const float factor = part_max == -INFINITY ? 0.0f : exp(part_max - maximum);
      total += partial_total[part * 8u + row] * factor;
      value += partial_output[(part * 8u + row) * CH_HEAD_DIM_MAX + dimension] * factor;
    }
    output[(first + row) * width + head * head_dim + dimension] = value / total;
  }
}
