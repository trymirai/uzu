#include "../../common/dsl.h"
#include "../../common/soft_cap.h"
#include "../../generated/gemm.h"
#include "../../hadamard_transform/hadamard_transform.h"
#include "common/arguments.h"
#include "common/quant_metadata.h"
#include "common/tile.h"

using namespace metal;
using namespace uzu;
using namespace uzu::gemm;

#define GEMV_MXU_TILED_SIMDGROUPS 8u

template <uint GROUP_SIZE>
VARIANTS(GROUP_SIZE, 32, 64)
KERNEL(GemvMxuTiled)(
    const device uint4* b,
    const device bfloat* scales,
    const device bfloat* a,
    device bfloat* d,
    const device bfloat* output_bias
        OPTIONAL(output_transform.contains(GemmDTransform::BIAS)),
    const device int32_t* hadamard_factors
        OPTIONAL(output_transform.contains(GemmDTransform::RHT)),
    const constant uint& in_vec_size,
    const constant uint& out_vec_size,
    const constant float& ab_scale,
    const constant uint& scale_output_stride,
    const constant uint& scale_group_stride,
    const constant float& soft_cap
        OPTIONAL(output_transform.contains(GemmDTransform::SOFT_CAP)),
    const GemmDTransform output_transform SPECIALIZE,
    const bool signed_codes SPECIALIZE,
    threadgroup float shared_results[GEMV_MXU_TILED_SIMDGROUPS * 32],
    const uint block GROUPS(out_vec_size / 32),
    const uint lane THREADS(32),
    const uint k_slice THREADS(GEMV_MXU_TILED_SIMDGROUPS)
) {
  constexpr uint K_SPLIT = GEMV_MXU_TILED_SIMDGROUPS;
  using Tile = GemvTile<1, 32, 4, 1, K_SPLIT, K_SPLIT>;
  using Ops = GemvOperands<bfloat, bfloat, bfloat>;
  using Metadata = QuantMetadata<Tile, bfloat, bfloat, bfloat, GemmBPrologueKind::ScaleSymmetricDequant, 8>;
  constexpr uint CHUNK_VALUES = 32;
  constexpr uint SEGMENT_VECTORS = 2;
  constexpr uint BLOCK_CHUNK_VECTORS = 32 * SEGMENT_VECTORS;
  const Ops ops = {
      reinterpret_cast<const device uint32_t*>(b),
      scales,
      nullptr,
      nullptr,
      a,
      d,
      output_bias,
      hadamard_factors,
      nullptr
  };
  const GemvParams params = {
      in_vec_size,
      out_vec_size,
      scale_output_stride,
      scale_group_stride,
      0,
      0,
      1,
      ab_scale,
      soft_cap,
      output_transform,
      false,
      signed_codes,
      GemmBLayout::MxuTiled
  };

  const uint pair = lane % 16;
  const uint parity = lane / 16;
  const uint y = (pair & 3) | (pair >> 3 << 2);
  const uint half_index = pair >> 2 & 1;
  uint rows[Tile::ROWS_PER_LANE];
  METAL_PRAGMA_UNROLL
  for (uint slot = 0; slot < Tile::ROWS_PER_LANE; ++slot) {
    rows[slot] = block * 32 + y + 8 * slot;
  }
  const uint chunks = in_vec_size / CHUNK_VALUES;
  const device uint4* block_weights = b + block * (in_vec_size / CHUNK_VALUES) * BLOCK_CHUNK_VECTORS;

  float result[1][Tile::ROWS_PER_LANE] = {{0}};
#pragma clang loop unroll_count(2)
  for (uint chunk = 2 * k_slice + parity; chunk < chunks; chunk += 2 * K_SPLIT) {
    const device uint4* source = block_weights + chunk * BLOCK_CHUNK_VECTORS + pair * 2 * SEGMENT_VECTORS;
    uint4 packed[2 * SEGMENT_VECTORS];
    METAL_PRAGMA_UNROLL
    for (uint vector = 0; vector < 2 * SEGMENT_VECTORS; ++vector) {
      packed[vector] = source[vector];
    }
    Metadata metadata;
    metadata.load(chunk * CHUNK_VALUES / GROUP_SIZE, rows, ops, params);
    float partial[1][Tile::ROWS_PER_LANE] = {{0}};
    float input_sum[1] = {0};
    const device bfloat* input = a + chunk * CHUNK_VALUES + 8 * half_index;
    METAL_PRAGMA_UNROLL
    for (uint step = 0; step < 2; ++step) {
      float input_values[8];
      METAL_PRAGMA_UNROLL
      for (uint i = 0; i < 4; ++i) {
        input_values[i] = float(input[4 * step + i]);
        input_values[4 + i] = float(input[16 + 4 * step + i]);
      }
      METAL_PRAGMA_UNROLL
      for (uint slot = 0; slot < Tile::ROWS_PER_LANE; ++slot) {
        float weight_values[8];
        metadata.decode(packed, step * 4 + slot, slot, weight_values);
        METAL_PRAGMA_UNROLL
        for (uint i = 0; i < 8; ++i) {
          partial[0][slot] = fma(input_values[i], weight_values[i], partial[0][slot]);
        }
      }
    }
    metadata.fold(result, partial, input_sum);
  }

  METAL_PRAGMA_UNROLL
  for (uint slot = 0; slot < Tile::ROWS_PER_LANE; ++slot) {
    result[0][slot] += simd_shuffle_xor(result[0][slot], 4);
    result[0][slot] += simd_shuffle_xor(result[0][slot], 16);
    if ((lane & 20) == 0) {
      shared_results[k_slice * 32 + y + 8 * slot] = result[0][slot];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (k_slice != 0) {
    return;
  }

  float value = 0.0f;
  METAL_PRAGMA_UNROLL
  for (uint slice = 0; slice < K_SPLIT; ++slice) {
    value += shared_results[slice * 32 + lane];
  }
  const uint row = block * 32 + lane;
  const bool fused_hadamard = output_transform.contains(GemmDTransform::RHT);
  if (output_transform.contains(GemmDTransform::SCALE)) {
    value *= ab_scale;
  }
  if (!fused_hadamard && output_transform.contains(GemmDTransform::BIAS)) {
    value += float(output_bias[row]);
  }
  if (output_transform.contains(GemmDTransform::SOFT_CAP)) {
    value = apply_soft_cap(value, soft_cap);
  }
  bfloat output = bfloat(value);
  if (fused_hadamard) {
    output = simdgroup_output_random_hadamard_transform(ushort(lane), output, hadamard_factors[row]);
    if (output_transform.contains(GemmDTransform::BIAS)) {
      output += output_bias[row];
    }
  }
  d[row] = output;
}
