#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "hadamard_transform.h"

using namespace metal;

template <typename T>
VARIANTS(T, half, bfloat, float)
PUBLIC KERNEL(HadamardTransform)(
    device T* data,
    const device int32_t* factors,
    constant uint& hidden_dim,
    constant uint& batch_size,
    uint block_index GROUPS(hidden_dim.div_ceil(METAL_SIMD_SIZE)),
    uint batch_index GROUPS(batch_size),
    uint lane_index THREADS(METAL_SIMD_SIZE)
) {
  uint factor_index = block_index * METAL_SIMD_SIZE + lane_index;
  uint element_index = batch_index * hidden_dim + factor_index;

  data[element_index] = simdgroup_random_hadamard_transform(
      lane_index,
      data[element_index],
      factors[factor_index]
  );
}
