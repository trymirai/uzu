#pragma once

#include "../../generated/unified_gemm.h"

namespace uzu {
namespace unified_gemm {

constexpr uint as_u32(BitsPerWeight bits) {
  return static_cast<uint>(bits);
}

constexpr uint weights_per_byte(BitsPerWeight bits) {
  return 8u / as_u32(bits);
}

constexpr uint packed_bytes_for_k(BitsPerWeight bits, uint k) {
  return k / weights_per_byte(bits);
}

static_assert(as_u32(BitsPerWeight::Bits4) == 4u, "BitsPerWeight::Bits4 discriminant mismatch");
static_assert(as_u32(BitsPerWeight::Bits8) == 8u, "BitsPerWeight::Bits8 discriminant mismatch");
static_assert(weights_per_byte(BitsPerWeight::Bits4) == 2u, "4-bit packs 2 weights per byte");
static_assert(weights_per_byte(BitsPerWeight::Bits8) == 1u, "8-bit packs 1 weight per byte");
static_assert(packed_bytes_for_k(BitsPerWeight::Bits4, 64u) == 32u, "64 4-bit weights fit in 32 bytes");
static_assert(packed_bytes_for_k(BitsPerWeight::Bits8, 64u) == 64u, "64 8-bit weights fit in 64 bytes");

} // namespace unified_gemm
} // namespace uzu
