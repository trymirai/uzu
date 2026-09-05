#pragma once

#include <metal_stdlib>

#include "../../common/microfloat.h"

using namespace metal;

namespace uzu {
namespace gemm {

template <
    typename T,
    short THREADGROUP_TILE_ROWS,
    short THREADGROUP_TILE_COLS,
    short DESTINATION_LEADING_DIMENSION,
    short THREADGROUP_SIZE,
    short GROUP_SIZE>
struct Mxfp4BlockLoader {
  static_assert(THREADGROUP_TILE_COLS % 2 == 0, "MXFP4 tiles must contain complete packed bytes");
  static_assert(GROUP_SIZE == 16 || GROUP_SIZE == 32, "MXFP4 groups contain 16 or 32 values");
  static_assert(THREADGROUP_TILE_COLS == GROUP_SIZE, "MXFP4 staging tiles must span one scale group");

  UZU_CONST short PACK_FACTOR = 2;
  UZU_CONST short PACKED_COLS = THREADGROUP_TILE_COLS / PACK_FACTOR;
  UZU_CONST short PACK_COUNT = THREADGROUP_TILE_ROWS * PACKED_COLS;
  UZU_CONST short READS_PER_THREAD = PACK_COUNT < THREADGROUP_SIZE ? 1 : PACK_COUNT / THREADGROUP_SIZE;
  UZU_CONST bool TILE_HAS_IDLE_THREADS = PACK_COUNT < THREADGROUP_SIZE;
  static_assert(
      PACK_COUNT < THREADGROUP_SIZE || PACK_COUNT % THREADGROUP_SIZE == 0,
      "MXFP4 tile packs must divide evenly among threads"
  );
  static_assert(
      TILE_HAS_IDLE_THREADS || PACKED_COLS % READS_PER_THREAD == 0,
      "each MXFP4 thread must remain within one weight row"
  );

  const short thread_row;
  const short thread_col;
  uint k_base;

  threadgroup T* destination;
  const device uint8_t* codes;
  const device uint8_t* scales;

  Mxfp4BlockLoader(
      const device uint8_t* codes_,
      const device uint8_t* scales_,
      const size_t code_row_stride_,
      const size_t scale_row_stride_,
      const uint k_base_,
      threadgroup T* destination_,
      ushort simdgroup_index,
      ushort simd_lane
  )
      : thread_row(READS_PER_THREAD * (simdgroup_index * 32 + simd_lane) / PACKED_COLS),
        thread_col((READS_PER_THREAD * (simdgroup_index * 32 + simd_lane)) % PACKED_COLS), k_base(k_base_),
        destination(destination_ + thread_row * DESTINATION_LEADING_DIMENSION + thread_col * PACK_FACTOR),
        codes(codes_ + size_t(thread_row) * code_row_stride_ + thread_col),
        scales(scales_ + size_t(thread_row) * scale_row_stride_) {}

  METAL_FUNC void decode_pack(const int pack, const float scale, const int valid_values = PACK_FACTOR) const {
    const uint packed = codes[pack];
    destination[pack * PACK_FACTOR] = T(decode_e2m1(packed & 0x0fu) * scale);
    destination[pack * PACK_FACTOR + 1] = valid_values == PACK_FACTOR ? T(decode_e2m1(packed >> 4u) * scale) : T(0);
  }

  METAL_FUNC void load_unsafe() const {
    if constexpr (TILE_HAS_IDLE_THREADS) {
      if (thread_row >= THREADGROUP_TILE_ROWS) {
        return;
      }
    }
    const float scale = decode_e8m0(scales[k_base / GROUP_SIZE]);
    for (int pack = 0; pack < READS_PER_THREAD; ++pack) {
      decode_pack(pack, scale);
    }
  }

  METAL_FUNC void load_safe(short2 source_dimensions) const {
    if constexpr (TILE_HAS_IDLE_THREADS) {
      if (thread_row >= THREADGROUP_TILE_ROWS) {
        return;
      }
    }
    if (thread_row >= source_dimensions.y) {
      for (int value = 0; value < READS_PER_THREAD * PACK_FACTOR; ++value) {
        destination[value] = T(0);
      }
      return;
    }

    const float scale = decode_e8m0(scales[k_base / GROUP_SIZE]);
    const int valid_packs = (source_dimensions.x + PACK_FACTOR - 1) / PACK_FACTOR;
    for (int pack = 0; pack < READS_PER_THREAD; ++pack) {
      const int pack_index = thread_col + pack;
      if (pack_index >= valid_packs) {
        destination[pack * PACK_FACTOR] = T(0);
        destination[pack * PACK_FACTOR + 1] = T(0);
        continue;
      }
      const int valid_values = min(int(PACK_FACTOR), int(source_dimensions.x) - pack_index * PACK_FACTOR);
      decode_pack(pack, scale, valid_values);
    }
  }

  METAL_FUNC void next() {
    codes += PACKED_COLS;
    k_base += THREADGROUP_TILE_COLS;
  }
};

} // namespace gemm
} // namespace uzu
