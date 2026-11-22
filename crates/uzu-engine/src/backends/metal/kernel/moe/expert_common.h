#pragma once

#include <metal_stdlib>

#include "../common/defines.h"
#include "../generated/moe.h"

using namespace metal;

namespace uzu {
namespace moe {

// Values per 16-byte chunk of 4-bit codes, the unit one lane loads at a time.
UZU_CONST uint CHUNK_VALUES = 32;
// E2M1 codes decode to their value times 2^-14 and E4M3 scales to theirs times 2^-8 (see below).
UZU_CONST float NVFP4_DECODE_SCALE = 4194304.0f; // 2^22
UZU_CONST uint NVFP4_GROUP_SIZE = 16;
UZU_CONST uint AFFINE_GROUP_SIZE = 64;

template <typename T>
METAL_FUNC float4 load_float4(const device T* source) {
  return float4(*reinterpret_cast<const device vec<T, 4>*>(source));
}

// Eight consecutive activations as float, from one 16-byte load (bfloat or half) or two (float).
template <typename T>
METAL_FUNC void load_values8(const device T* source, thread float (&values)[8]) {
  if constexpr (metal::is_same_v<T, float>) {
    const device float4* vectors = reinterpret_cast<const device float4*>(source);
    const float4 low = vectors[0];
    const float4 high = vectors[1];
    METAL_PRAGMA_UNROLL
    for (uint i = 0; i < 4; ++i) {
      values[i] = low[i];
      values[4 + i] = high[i];
    }
  } else if constexpr (metal::is_same_v<T, bfloat>) {
    const uint4 packed = *reinterpret_cast<const device uint4*>(source);
    METAL_PRAGMA_UNROLL
    for (uint i = 0; i < 4; ++i) {
      values[2 * i] = as_type<float>(packed[i] << 16);
      values[2 * i + 1] = as_type<float>(packed[i] & 0xFFFF0000u);
    }
  } else {
    static_assert(metal::is_same_v<T, half>, "activations are float, bfloat or half");
    const uint4 packed = *reinterpret_cast<const device uint4*>(source);
    METAL_PRAGMA_UNROLL
    for (uint i = 0; i < 4; ++i) {
      const float2 pair = float2(as_type<half2>(packed[i]));
      values[2 * i] = pair.x;
      values[2 * i + 1] = pair.y;
    }
  }
}

// One 32-bit word holds eight E2M1 codes, repacked at load (nvfp4_repack_word in gpu_types/moe.rs). Pair k is
// (value k, value k + 4), each the E2M1 value times 2^-14: its three magnitude bits land in half bits [9, 12) and its
// sign in bit 15. Codes of magnitude 0.5 come out subnormal; converting them to float keeps them exact.
METAL_FUNC void decode_e2m1x8(const uint word, thread half2 (&pairs)[4]) {
  pairs[0] = as_type<half2>(word & 0x8E008E00u);
  pairs[1] = as_type<half2>((word << 3) & 0x8E008E00u);
  pairs[2] = as_type<half2>(((word << 6) & 0x0E000E00u) | ((word << 2) & 0x80008000u));
  pairs[3] = as_type<half2>(((word << 9) & 0x0E000E00u) | ((word << 1) & 0x80008000u));
}

// Two non-negative E4M3 bytes (low byte first) as float, each times 2^-8.
METAL_FUNC float2 decode_e4m3x2(const uint bytes) {
  return float2(
      float(as_type<half>(ushort((bytes << 7) & 0x3F80u))),
      float(as_type<half>(ushort((bytes >> 1) & 0x3F80u)))
  );
}

// One non-negative E4M3 byte as float, times 2^-8.
METAL_FUNC float decode_e4m3(const uint byte) { return float(as_type<half>(ushort((byte << 7) & 0x3F80u))); }

// One 32-bit word holds eight unsigned 4-bit codes, code i in bits [4i, 4i + 4). `low` pairs hold (q0, q4) and
// (q2, q6), `high` pairs (16 q1, 16 q5) and (16 q3, 16 q7): a nibble lands in the mantissa of 1024 and the
// subtraction is exact. (Masking the nibbles into subnormal halves saves integer work but converts slower.)
METAL_FUNC void decode_u4x8(const uint word, thread half2 (&low)[2], thread half2 (&high)[2]) {
  const half2 offset = half2(1024.0h);
  const uint shifted = word >> 8;
  low[0] = as_type<half2>((word & 0x000F000Fu) | 0x64006400u) - offset;
  high[0] = as_type<half2>((word & 0x00F000F0u) | 0x64006400u) - offset;
  low[1] = as_type<half2>((shifted & 0x000F000Fu) | 0x64006400u) - offset;
  high[1] = as_type<half2>((shifted & 0x00F000F0u) | 0x64006400u) - offset;
}

// Sum of code i * values[i] over the word's eight unsigned codes. `low_acc` collects the even codes, `high_acc`
// sixteen times the odd ones.
METAL_FUNC void dot_u4x8(
    const uint word,
    const thread float (&values)[8],
    thread float& low_acc,
    thread float& high_acc
) {
  half2 low[2];
  half2 high[2];
  decode_u4x8(word, low, high);
  low_acc = fma(float(low[0].x), values[0], low_acc);
  low_acc = fma(float(low[0].y), values[4], low_acc);
  low_acc = fma(float(low[1].x), values[2], low_acc);
  low_acc = fma(float(low[1].y), values[6], low_acc);
  high_acc = fma(float(high[0].x), values[1], high_acc);
  high_acc = fma(float(high[0].y), values[5], high_acc);
  high_acc = fma(float(high[1].x), values[3], high_acc);
  high_acc = fma(float(high[1].y), values[7], high_acc);
}

// An entry of the active-expert table the routing kernels write: the expert, its route count and its first route.
struct ActiveExpert {
  uint expert;
  uint count;
  uint first_route;
};

METAL_FUNC ActiveExpert decode_active_entry(const uint entry) {
  return ActiveExpert{
      entry & ((1u << MOE_ACTIVE_EXPERT_BITS) - 1),
      (entry >> MOE_ACTIVE_EXPERT_BITS) & ((1u << MOE_ACTIVE_COUNT_BITS) - 1),
      entry >> (MOE_ACTIVE_EXPERT_BITS + MOE_ACTIVE_COUNT_BITS),
  };
}

METAL_FUNC float expert_sigmoid(float x) { return 1.0f / (1.0f + precise::exp(-x)); }

// The gated expert activation in float32: `up` optionally soft-capped, times SiLU of `gate` whose linear factor is
// optionally soft-capped.
METAL_FUNC float expert_activation(
    float up,
    float gate,
    bool cap_up,
    float up_soft_cap,
    bool cap_gate,
    float gate_soft_cap,
    float activation_alpha
) {
  if (cap_up) {
    up = up_soft_cap * precise::tanh(up / up_soft_cap);
  }
  const float linear = cap_gate ? gate_soft_cap * precise::tanh(gate / gate_soft_cap) : gate;
  return up * linear * expert_sigmoid(activation_alpha * gate);
}

} // namespace moe
} // namespace uzu
