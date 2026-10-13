#include "../../common/dsl.h"
#include "../../common/defines.h"
#include "../common/mxu_fragment/ops.h"

using namespace metal;

#define TILE_CODES_THREADS 1024u

template <uint BITS, uint WORDS_PER_THREAD>
VARIANTS(BITS, 4, 8)
VARIANTS(WORDS_PER_THREAD, 32, 72, 160)
KERNEL(GemmTileCodes)(
    device uint* codes,
    const constant uint& row_words,
    const constant uint& row_block_count,
    const uint row_block GROUPS(row_block_count),
    const uint thread_index THREADS(TILE_CODES_THREADS)
) {
  using Ops = uzu::matmul::MxuFragmentOps<>;
  constexpr uint CHUNK_BYTES = BITS == 4 ? 16 : 32;
  constexpr uint COL_FRAGMENTS = BITS == 4 ? 1 : 2;
  constexpr uint LANE_WORDS = 2 * Ops::THREAD_ELEMENT_ROWS * COL_FRAGMENTS;
  const uint block_words = 2 * Ops::FRAGMENT_ROWS * row_words;
  device uint* block = codes + size_t(row_block) * block_words;

  uint words[WORDS_PER_THREAD];
  METAL_PRAGMA_UNROLL
  for (uint slot = 0; slot < WORDS_PER_THREAD; ++slot) {
    const uint index = slot * TILE_CODES_THREADS + thread_index;
    if (index < block_words) {
      const uint lane = (index / LANE_WORDS) % METAL_SIMD_SIZE;
      const uint word = index % LANE_WORDS;
      const uint fragment_row = word / (Ops::THREAD_ELEMENT_ROWS * COL_FRAGMENTS);
      const uint row_slot = word / COL_FRAGMENTS % Ops::THREAD_ELEMENT_ROWS;
      const uint fragment_column = word % COL_FRAGMENTS;
      const short2 position = Ops::get_position(ushort(lane));
      const uint row = uint(position.y) + fragment_row * Ops::FRAGMENT_ROWS + row_slot * Ops::THREAD_ELEMENT_ROW_STRIDE;
      const uint byte = index / (METAL_SIMD_SIZE * LANE_WORDS) * CHUNK_BYTES + uint(position.x) +
                        fragment_column * Ops::FRAGMENT_COLS;
      words[slot] = block[row * row_words + byte / 4];
    }
  }
  threadgroup_barrier(mem_flags::mem_device);
  METAL_PRAGMA_UNROLL
  for (uint slot = 0; slot < WORDS_PER_THREAD; ++slot) {
    const uint index = slot * TILE_CODES_THREADS + thread_index;
    if (index < block_words) {
      block[index] = words[slot];
    }
  }
}
