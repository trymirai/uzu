

#pragma once

#include "../defines.h"

///////////////////////////////////////////////////////////////////////////////
// Loading helper
///////////////////////////////////////////////////////////////////////////////

namespace steel {

template <typename T>
struct BlockLoader {
  // Tile params
  const short BROWS;
  const short dst_ld;
  const short vec_size;
  const short TROWS;

  // Leading dimension for src
  const int src_ld;
  const int tile_stride;

  // Thread location indices
  const short bi;
  const short bj;

  // threadgroup and device memory
  threadgroup T* dst;
  const device T* src;

  /* Constructor */
  METAL_FUNC BlockLoader(
      const device T* src_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id,
      ushort simd_lane_id,
      short BROWS_,
      short BCOLS_,
      short dst_ld_,
      short reduction_dim,
      short tgp_size
  )
      : BROWS(BROWS_), dst_ld(dst_ld_),
        vec_size((BCOLS_ * BROWS_) / tgp_size),
        TROWS(tgp_size / (BCOLS_ / ((BCOLS_ * BROWS_) / tgp_size))),
        src_ld(src_ld_),
        tile_stride(reduction_dim ? BCOLS_ : BROWS_ * src_ld_),
        bi(short(simd_group_id * 32 + simd_lane_id) /
           short(BCOLS_ / ((BCOLS_ * BROWS_) / tgp_size))),
        bj(short((BCOLS_ * BROWS_) / tgp_size) *
           (short(simd_group_id * 32 + simd_lane_id) %
            short(BCOLS_ / ((BCOLS_ * BROWS_) / tgp_size)))),
        dst(dst_ + bi * dst_ld_ + bj),
        src(src_ + bi * src_ld_ + bj) {}

  /* Load from device memory into threadgroup memory - without bound checking */
  METAL_FUNC void load_unsafe() const {
    for (short i = 0; i < BROWS; i += TROWS) {
      for (short j = 0; j < vec_size; j++) {
        dst[i * dst_ld + j] = src[i * src_ld + j];
      }
    }
  }

  /* Load from device memory into threadgroup memory - with bound checking */
  METAL_FUNC void load_safe(short2 src_tile_dim) const {
    src_tile_dim = src_tile_dim - short2(bj, bi);

    if (src_tile_dim.x <= 0 || src_tile_dim.y <= 0) {
      for (short i = 0; i < BROWS; i += TROWS) {
        for (short j = 0; j < vec_size; j++) {
          dst[i * dst_ld + j] = T(0);
        }
      }
      return;
    }

    for (short i = 0; i < BROWS; i += TROWS) {
      for (short j = 0; j < vec_size; j++) {
        bool valid = (i < src_tile_dim.y) && (j < src_tile_dim.x);
        T val = src[(valid ? i * src_ld + j : 0)];
        dst[i * dst_ld + j] = valid ? val : T(0);
      }
    }
  }

  /* Iteration helper */
  METAL_FUNC void next() { src += tile_stride; }
};

} // namespace steel
