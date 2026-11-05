#include "../../common/dsl.h"
#include "../../common/defines.h"
#include "common/mxu_tiled_layout.h"

using namespace metal;

using uzu::matmul::MXU_TILED_PREPARE_THREADS;

template <uint BITS>
static METAL_FUNC uint codes_in_activation_order(const device uint* source, const uint column) {
  constexpr uint BYTE_BITS = 8;
  constexpr uint CODES_PER_BYTE = BYTE_BITS / BITS;
  const uint source_byte = column / CODES_PER_BYTE;
  uint packed = 0;
  METAL_PRAGMA_UNROLL
  for (uint source_word = 0; source_word < CODES_PER_BYTE; ++source_word) {
    const uint codes = source[source_word] >> (BYTE_BITS * source_byte);
    METAL_PRAGMA_UNROLL
    for (uint code = 0; code < CODES_PER_BYTE; ++code) {
      packed |= ((codes >> (BITS * code)) & ((1u << BITS) - 1)) << (BYTE_BITS * code + BITS * source_word);
    }
  }
  return packed;
}

template <uint BITS, uint WORDS_PER_THREAD>
VARIANTS(BITS, 2, 4, 8)
VARIANTS(WORDS_PER_THREAD, 32, 72, 160)
KERNEL(GemmPrepareBLayout)(
    device uint* weights,
    const constant uint& row_words,
    const constant uint& row_block_count,
    const uint row_block GROUPS(row_block_count),
    const uint thread_index THREADS(MXU_TILED_PREPARE_THREADS)
) {
  using Layout = uzu::gemm::MxuTiledLayout<BITS>;
  using Ops = typename Layout::Ops;
  constexpr uint ROW_SLOT_STRIDE = BITS == 2 ? 1 : Layout::COL_FRAGMENTS;
  constexpr uint COLUMN_STRIDE = BITS == 2 ? Ops::THREAD_ELEMENT_ROWS : 1;
  const uint block_words = Layout::BLOCK_ROWS * row_words;
  device uint* block = weights + size_t(row_block) * block_words;

  uint words[WORDS_PER_THREAD];
  METAL_PRAGMA_UNROLL
  for (uint slot = 0; slot < WORDS_PER_THREAD; ++slot) {
    const uint index = slot * MXU_TILED_PREPARE_THREADS + thread_index;
    if (index < block_words) {
      const uint chunk = index / (METAL_SIMD_SIZE * Layout::LANE_WORDS);
      const uint word = index % Layout::LANE_WORDS;
      const short2 position = Ops::get_position(ushort(index / Layout::LANE_WORDS % METAL_SIMD_SIZE));
      const uint fragment_row = word / (Ops::THREAD_ELEMENT_ROWS * Layout::COL_FRAGMENTS);
      const uint row_slot = word / ROW_SLOT_STRIDE % Ops::THREAD_ELEMENT_ROWS;
      const uint fragment_column = word / COLUMN_STRIDE % Layout::COL_FRAGMENTS;
      const uint row = uint(position.y) + fragment_row * Ops::FRAGMENT_ROWS + row_slot * Ops::THREAD_ELEMENT_ROW_STRIDE;
      const device uint* source = block + row * row_words +
                                  (chunk * Layout::COL_FRAGMENTS + fragment_column) * Ops::FRAGMENT_COLS / sizeof(uint);
      words[slot] = BITS == 2 ? codes_in_activation_order<BITS>(source, uint(position.x))
                              : source[uint(position.x) / sizeof(uint)];
    }
  }
  threadgroup_barrier(mem_flags::mem_device);
  METAL_PRAGMA_UNROLL
  for (uint slot = 0; slot < WORDS_PER_THREAD; ++slot) {
    const uint index = slot * MXU_TILED_PREPARE_THREADS + thread_index;
    if (index < block_words) {
      block[index] = words[slot];
    }
  }
}
