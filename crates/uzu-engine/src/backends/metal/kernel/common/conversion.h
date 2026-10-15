#pragma once

#include <metal_stdlib>

#include "defines.h"

namespace uzu {

UZU_CONST float FLOAT_OFFSET = 0x1p23f;

// Round [0, 255] to nearest-even without a numeric conversion.
METAL_FUNC uchar round_to_byte(const float x) { return uchar(as_type<uint>(x + 0x1.8p23f)); }

// Returns float(i) + FLOAT_OFFSET for 0 <= i < 2^23; callers must remove the offset.
METAL_FUNC float float_with_offset(const uint i) { return as_type<float>(i | as_type<uint>(FLOAT_OFFSET)); }

// Widen two packed BF16 values, low half first.
METAL_FUNC float2 unpack_bfloat(const uint packed) {
  return as_type<float2>(uint2(packed << 16, packed & 0xffff0000u));
}

METAL_FUNC float4 bfloat_to_float(const bfloat4 value) { return as_type<float4>(uint4(as_type<ushort4>(value)) << 16); }

} // namespace uzu
