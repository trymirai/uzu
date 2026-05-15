#include <metal_stdlib>
#include "../common/dsl.h"
#include "nf4_qmm_core.h"

// NF4 transposed QMM with constant-codebook lookup and 1-byte E4M3 FP8
// per-group scales (half the scale bytes of Nf4QmmConstant). Mirrors
// Nf4QmmConstant's VARIANTS/CONSTRAINT/signature exactly; only the `scales`
// buffer type and the per-group scale decode differ.
template <typename T, uint GROUP_SIZE, uint BM, uint BK, uint BN, uint WM, uint WN>
VARIANTS(T, bfloat)
VARIANTS(GROUP_SIZE, 64)
VARIANTS(BM, 8, 32, 64)
VARIANTS(BK, 32, 64)
VARIANTS(BN, 32, 64)
VARIANTS(WM, 1, 2)
VARIANTS(WN, 1, 2)
CONSTRAINT(
  (BM == 8  && BK == 32 && BN == 32 && WM == 1 && WN == 1) ||
  (BM == 32 && BK == 32 && BN == 32 && WM == 2 && WN == 2) ||
  (BM == 64 && BK == 64 && BN == 64 && WM == 2 && WN == 2))
CONSTRAINT(BK <= GROUP_SIZE)
KERNEL(Nf4QmmE4m3)(
    const device uint32_t* weights,
    const device uint8_t* scales,
    const device T* input,
    device T* output,
    const constant uint& in_vec_size,
    const constant uint& out_vec_size,
    const constant uint& batch_size,
    threadgroup T Xs[BM * (BK + 16 / sizeof(T))],
    threadgroup T Ws[BN * (BK + 16 / sizeof(T))],
    const uint out_block_idx GROUPS(out_vec_size.div_ceil(BN)),
    const uint batch_block_idx GROUPS(batch_size.div_ceil(BM)),
    const uint simd_lane THREADS(32),
    const uint simd_group THREADS(WM * WN)
) {
  Nf4LookupCtx<NF4_CONST> ctx;

  nf4_qmm_impl_e4m3<T, GROUP_SIZE, NF4_CONST, true, BM, BK, BN, WM, WN>(
      weights,
      scales,
      input,
      output,
      Xs,
      Ws,
      ctx,
      in_vec_size,
      out_vec_size,
      batch_size,
      out_block_idx,
      batch_block_idx,
      simd_group,
      simd_lane
  );
}
