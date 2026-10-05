// AMDGPU version of uzu's `dequantize_contiguous` (matmul/common/quant_unpack.h), used by the staged GEMM
// loaders. The generic version dequantizes one code byte per call in bf16 arithmetic and stores values one
// by one: on RDNA that is byte loads, bf16<->f32 conversions around every operation and 2-byte LDS stores.
// Here a thread's run of codes is read as 32-bit words, dequantized in f32 and stored to LDS as one 16- to
// 64-byte vector. The AMD GEMM only runs with K a multiple of 8 (amdgpu/kernel/matmul/gemm.rs), which keeps
// the code words 4-byte and the staged rows 16-byte aligned.
#pragma once

#define UZU_DEQUANTIZE_CONTIGUOUS 1

namespace uzu {
namespace gemm {

template <typename U, int N, int bits>
inline void dequantize(const __global uchar* w, U scale, U bias, __local U* w_local, const bool signed_codes);

template <typename U, int N, int bits, int PACKS>
inline void dequantize_contiguous(
    const __global uchar* w,
    U scale,
    U bias,
    __local U* w_local,
    const bool signed_codes
) {
  // 4- and 8-bit codes pack into one byte (N = 2 or 1 values), so a run of PACKS packs is PACKS bytes.
  constexpr int BYTES = PACKS;
  constexpr int VALUES = PACKS * N;
  constexpr bool VECTORIZED = (bits == 4 || bits == 8) && N == 8 / bits &&
                              (BYTES == 4 || BYTES == 8 || BYTES == 16) && sizeof(U) == 2 &&
                              (VALUES == 8 || VALUES == 16 || VALUES == 32);
  if constexpr (VECTORIZED) {
    const float s = float(scale);
    const float b = float(bias);
    float values[VALUES];
#pragma unroll
    for (int word_index = 0; word_index < BYTES / 4; ++word_index) {
      uint word = reinterpret_cast<const __global uint*>(w)[word_index];
      if constexpr (bits == 4) {
        // signed nibbles are stored offset by 8, as the generic path's XOR with 0x88
        if (signed_codes) {
          word ^= 0x88888888u;
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          values[word_index * 8 + j] = s * float((word >> (4 * j)) & 15u) + b;
        }
      } else {
        // int8 codes: scale * (code + 128) + bias, as the generic path's adjusted bias
        if (signed_codes) {
          word ^= 0x80808080u;
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          values[word_index * 4 + j] = s * float((word >> (8 * j)) & 255u) + b;
        }
      }
    }
    typedef ushort __uzu_dequant_vector __attribute__((ext_vector_type(VALUES)));
    __uzu_dequant_vector out;
#pragma unroll
    for (int i = 0; i < VALUES; ++i) {
      out[i] = __builtin_bit_cast(ushort, U(values[i]));
    }
    *reinterpret_cast<__local __uzu_dequant_vector*>(w_local) = out;
  } else {
#pragma unroll
    for (int i = 0; i < PACKS; i++) {
      dequantize<U, N, bits>(w + i, scale, bias, w_local + i * N, signed_codes);
    }
  }
}

} // namespace gemm
} // namespace uzu
