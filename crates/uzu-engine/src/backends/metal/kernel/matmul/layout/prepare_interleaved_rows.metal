#include "../../common/dsl.h"
#include "../../common/defines.h"
#include "../../generated/matmul.h"
#include "interleaved_rows_layout.h"

using namespace metal;

using uzu::gemm::InterleavedRowsLayout;
using uzu::matmul::INTERLEAVED_ROWS_RELAYOUT_THREADS;

// Relays a strip's row-major W4 codes into InterleavedRowsLayout in place, through registers.
template <uint WORDS_PER_THREAD>
VARIANTS(WORDS_PER_THREAD, 16, 48, 96)
KERNEL(PrepareInterleavedRows)(
    device uint* weights,
    const constant uint& row_words,
    const constant uint& strip_count,
    const uint strip GROUPS(strip_count),
    const uint thread_index THREADS(INTERLEAVED_ROWS_RELAYOUT_THREADS)
) {
  const uint strip_words = InterleavedRowsLayout::STRIP_ROWS * row_words;
  device uint* codes = weights + size_t(strip) * strip_words;
  uint words[WORDS_PER_THREAD];
  METAL_PRAGMA_UNROLL
  for (uint slot = 0; slot < WORDS_PER_THREAD; ++slot) {
    const uint index = slot * INTERLEAVED_ROWS_RELAYOUT_THREADS + thread_index;
    if (index < strip_words) {
      words[slot] = codes[InterleavedRowsLayout::row_major_word_in_strip(index, row_words)];
    }
  }
  threadgroup_barrier(mem_flags::mem_device);
  METAL_PRAGMA_UNROLL
  for (uint slot = 0; slot < WORDS_PER_THREAD; ++slot) {
    const uint index = slot * INTERLEAVED_ROWS_RELAYOUT_THREADS + thread_index;
    if (index < strip_words) {
      codes[index] = words[slot];
    }
  }
}
