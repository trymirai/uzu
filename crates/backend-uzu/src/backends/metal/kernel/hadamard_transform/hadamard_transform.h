#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/thread_context.h"

using namespace metal;

static METAL_FUNC float simdgroup_hadamard_transform(
    ushort lane_index,
    float lane_value
) {
  for (ushort stride = 1; stride < METAL_SIMD_SIZE; stride <<= 1) {
    float partner_lane_value = simd_shuffle_xor(lane_value, stride);
    lane_value = (lane_index & stride) ? (partner_lane_value - lane_value)
                                       : (partner_lane_value + lane_value);
  }

  return lane_value / sqrt((float)METAL_SIMD_SIZE);
}

template <typename T>
static METAL_FUNC T simdgroup_random_hadamard_transform(
    ushort lane_index,
    T lane_value,
    int32_t lane_factor
) {
  return T(simdgroup_hadamard_transform(
      lane_index,
      float(lane_value) * float(lane_factor)
  ));
}
