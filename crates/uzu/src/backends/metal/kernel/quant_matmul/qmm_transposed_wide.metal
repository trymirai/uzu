#include <metal_stdlib>
#include "../common/dsl.h"
#include "quant_matmul.h"

// Wide QMM tile: BM=64, BK=32, BN=64.
// Processes 4x more output elements per threadgroup than the 32x32 variant.
// BK=32 satisfies the group_size >= BK constraint for all group sizes (32, 64,
// 128).
template <typename T, uint GROUP_SIZE, uint BITS>
VARIANTS(T, bfloat)
VARIANTS(GROUP_SIZE, 32, 64, 128)
VARIANTS(BITS, 4, 8)
PUBLIC KERNEL(QuantizedMatmulQmmTransposedWide)(
    const device uint32_t* weights,
    const device T* scales,
    const device uint8_t* zero_points OPTIONAL(use_zero_points),
    const device T* biases OPTIONAL(use_mlx_quant),
    const device T* input,
    device T* output,
    const constant uint& in_vec_size,
    const constant uint& out_vec_size,
    const constant uint& batch_size,
    threadgroup T Xs[64 * (32 + 16 / sizeof(T))],
    threadgroup T Ws[64 * (32 + 16 / sizeof(T))],
    const bool use_zero_points SPECIALIZE,
    const bool use_mlx_quant SPECIALIZE,
    const uint out_block_idx GROUPS(out_vec_size.div_ceil(64)),
    const uint batch_block_idx GROUPS(batch_size.div_ceil(64)),
    const uint simd_lane THREADS(32),
    const uint simd_group THREADS(4)
) {
  if (use_mlx_quant) {
    qmm_transposed_impl<T, GROUP_SIZE, BITS, true, 64, 32, 64, true>(
        weights,
        scales,
        zero_points,
        biases,
        input,
        output,
        Xs,
        Ws,
        in_vec_size,
        out_vec_size,
        batch_size,
        out_block_idx,
        batch_block_idx,
        simd_group,
        simd_lane
    );
  } else {
    qmm_transposed_impl<T, GROUP_SIZE, BITS, true, 64, 32, 64, false>(
        weights,
        scales,
        zero_points,
        biases,
        input,
        output,
        Xs,
        Ws,
        in_vec_size,
        out_vec_size,
        batch_size,
        out_block_idx,
        batch_block_idx,
        simd_group,
        simd_lane
    );
  }
}
