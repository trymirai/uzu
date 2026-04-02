#include <metal_stdlib>
#include "../activation/activations.h"
#include "../common/dsl.h"
#include "../hadamard_transform/hadamard_transform.h"

using namespace uzu::activation_type;

template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(MlpGateActMul) (
    const device T* fused_up,
    device T* hidden,
    const device int32_t* hadamard_factors OPTIONAL(use_hadamard),
    const constant int& h,
    const constant int& m,
    const constant ActivationType& act_type,
    const bool use_hadamard SPECIALIZE,
    uint j AXIS(h, 64),
    uint row AXIS(m, 1)
) {
  int base = row * (2 * h);
  T up = fused_up[base + j];
  T gate = fused_up[base + h + j];
  T g = activate(gate, act_type);
  T result = T(float(up) * float(g));

  if (use_hadamard) {
    result = simdgroup_random_hadamard_transform(
        static_cast<ushort>(j % METAL_SIMD_SIZE),
        result,
        hadamard_factors[j]
    );
  }

  hidden[row * h + j] = result;
}
