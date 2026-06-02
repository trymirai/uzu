#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../generated/hadamard_order.h"
#include "hadamard_transform.h"

using namespace metal;
using namespace uzu::hadamard_order;

template <typename T>
VARIANTS(T, bfloat, float)
PUBLIC KERNEL(HadamardTransform)(
    device T* data,
    const device int32_t* factors,
    constant uint& hidden_dim,
    constant uint& batch_size,
    const HadamardTransformOrder transform_order SPECIALIZE,
    uint block_index GROUPS(hidden_dim.div_ceil(METAL_SIMD_SIZE)),
    uint batch_index GROUPS(batch_size),
    uint lane_index THREADS(METAL_SIMD_SIZE)
) {
  uint factor_index = block_index * METAL_SIMD_SIZE + lane_index;
  uint element_index = batch_index * hidden_dim + factor_index;

  if (transform_order == HadamardTransformOrder::Input) {
    data[element_index] = simdgroup_input_random_hadamard_transform(
        lane_index,
        data[element_index],
        factors[factor_index]
    );
  } else {
    data[element_index] = simdgroup_output_random_hadamard_transform(
        lane_index,
        data[element_index],
        factors[factor_index]
    );
  }
}
