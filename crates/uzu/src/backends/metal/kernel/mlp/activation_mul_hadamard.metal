#include <metal_stdlib>
#include "../activation/activations.h"
#include "../common/defines.h"
#include "../common/dsl.h"

using namespace uzu::activation_type;

// Fused MlpGateActMul + Hadamard transform for down_proj input.
// Computes up * activate(gate), multiplies by Hadamard factors, applies the
// Walsh-Hadamard butterfly via simd_shuffle_xor, writes to hidden buffer.
// Eliminates the standalone input Hadamard dispatch for down_proj.
// Requires h % 32 == 0 for aligned Hadamard blocks.
template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(MlpGateActMulHadamard)(
    const device T* fused_up,
    device T* hidden,
    const device T* hadamard_factors,
    const constant int& h,
    const constant int& m,
    const constant ActivationType& act_type,
    uint j AXIS(h, 64),
    uint row AXIS(m, 1)
) {
  float out_f = 0.0f;
  bool valid = (static_cast<int>(j) < h);

  if (valid) {
    int base = row * (2 * h);
    T up = fused_up[base + j];
    T gate = fused_up[base + h + j];
    T g = activate(gate, act_type);
    out_f = float(up) * float(g);
    out_f *= float(hadamard_factors[j]);
  }

  uint lane = j % METAL_SIMD_SIZE;

  for (uint stride = 1; stride < METAL_SIMD_SIZE; stride <<= 1) {
    float partner = simd_shuffle_xor(out_f, static_cast<ushort>(stride));
    out_f = (lane & stride) ? (partner - out_f) : (partner + out_f);
  }

  if (valid) {
    constexpr float normalization_factor = 1.0f / 5.656854249f;
    hidden[row * h + j] = T(out_f * normalization_factor);
  }
}
