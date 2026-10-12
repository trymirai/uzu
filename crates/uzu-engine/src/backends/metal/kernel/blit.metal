#include <metal_stdlib>
#include "common/dsl.h"

#define BLIT_MAX_THREADS 16777216

KERNEL(BlitCopy)(
    const device uchar* src,
    device uchar* dst,
    constant uint64_t& size,
    const uint alignment SPECIALIZE,
    uint position AXIS(size.div_ceil(16).min(BLIT_MAX_THREADS), 256)
) {
  for (ulong index = position; index < (size + 15) / 16; index += BLIT_MAX_THREADS) {
    const ulong offset = index * 16;
    if (size - offset >= 16 && alignment == 0) {
      reinterpret_cast<device uint4*>(dst)[index] = reinterpret_cast<const device uint4*>(src)[index];
    } else if (size - offset >= 16 && (alignment & 3) == 0) {
      reinterpret_cast<device packed_uint4*>(dst)[index] = reinterpret_cast<const device packed_uint4*>(src)[index];
    } else if ((alignment & 3) == 0) {
      for (ulong word = offset / 4; word < size / 4; ++word) {
        reinterpret_cast<device uint*>(dst)[word] = reinterpret_cast<const device uint*>(src)[word];
      }
      for (ulong byte = size / 4 * 4; byte < size; ++byte) {
        dst[byte] = src[byte];
      }
    } else {
      for (ulong i = offset; i < min(offset + 16, size); ++i) {
        dst[i] = src[i];
      }
    }
  }
}

KERNEL(BlitFill)(
    device uint* dst,
    constant uint64_t& size,
    constant uint& value,
    const bool aligned SPECIALIZE,
    uint position AXIS(size.div_ceil(16).min(BLIT_MAX_THREADS), 256)
) {
  for (ulong index = position; index < (size + 15) / 16; index += BLIT_MAX_THREADS) {
    const ulong offset = index * 16;
    if (size - offset >= 16) {
      if (aligned) {
        reinterpret_cast<device uint4*>(dst)[index] = uint4(value);
      } else {
        reinterpret_cast<device packed_uint4*>(dst)[index] = uint4(value);
      }
    } else {
      for (ulong i = offset / 4; i < size / 4; ++i) {
        dst[i] = value;
      }
    }
  }
}
