#include <metal_stdlib>
#include "../../common/defines.h"
#include "../../common/dsl.h"

using namespace metal;

template <uint CHUNK_SIZE>
VARIANTS(CHUNK_SIZE, 32, 64)
KERNEL(DeltaNetChunkedCumsum)(
    device const float* log_decay,
    device float* g_out,
    constant const uint& num_v_heads,
    constant const uint& suffix_len,
    const uint chunk_idx GROUPS(suffix_len.div_ceil(CHUNK_SIZE)),
    const uint hv_idx GROUPS(num_v_heads),
    const uint lane THREADS(METAL_SIMD_SIZE)
) {
  const uint token_begin = chunk_idx * CHUNK_SIZE;
  const uint token_end = min(token_begin + CHUNK_SIZE, suffix_len);

  float carry = 0.0f;
  METAL_PRAGMA_UNROLL
  for (uint token_offset = 0; token_offset < CHUNK_SIZE; token_offset += METAL_SIMD_SIZE) {
    const uint token = token_begin + token_offset + lane;
    const bool valid = token < token_end;
    const uint input_offset = token * num_v_heads + hv_idx;
    const uint output_offset = hv_idx * suffix_len + token;
    const float value = valid ? log_decay[input_offset] : 0.0f;
    const float prefix = carry + simd_prefix_inclusive_sum(value);

    if (valid) {
      g_out[output_offset] = prefix;
    }

    carry += simd_sum(value);
  }
}
