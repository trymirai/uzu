#include <metal_stdlib>
#include "../../common/defines.h"
#include "../../common/dsl.h"

using namespace metal;

template <uint CHUNK_SIZE>
VARIANTS(CHUNK_SIZE, 16, 32, 64)
KERNEL(DeltaNetChunkedCumsum)(
    device const float* log_decay,
    device float* g_out,
    constant const uint& num_v_heads,
    constant const uint& suffix_len,
    const uint chunk_idx GROUPS(suffix_len.div_ceil(CHUNK_SIZE)),
    const uint hv_idx GROUPS(num_v_heads),
    const uint lane THREADS(METAL_SIMD_SIZE)
) {
  static_assert(CHUNK_SIZE <= 2 * METAL_SIMD_SIZE, "cumsum handles at most two tokens per lane");

  const uint token_begin = chunk_idx * CHUNK_SIZE;
  const uint token_end = min(token_begin + CHUNK_SIZE, suffix_len);

  const uint token0 = token_begin + lane;
  const bool valid0 = token0 < token_end;
  const uint input_offset0 = token0 * num_v_heads + hv_idx;
  const uint output_offset0 = hv_idx * suffix_len + token0;
  float value0 = valid0 ? log_decay[input_offset0] : 0.0f;
  const float prefix0 = simd_prefix_inclusive_sum(value0);
  if (valid0) {
    g_out[output_offset0] = prefix0;
  }

  if constexpr (CHUNK_SIZE > METAL_SIMD_SIZE) {
    const uint token1 = token_begin + METAL_SIMD_SIZE + lane;
    const bool valid1 = token1 < token_end;
    const uint input_offset1 = token1 * num_v_heads + hv_idx;
    const uint output_offset1 = hv_idx * suffix_len + token1;
    const float value1 = valid1 ? log_decay[input_offset1] : 0.0f;
    const float prefix1 = simd_sum(value0) + simd_prefix_inclusive_sum(value1);

    if (valid1) {
      g_out[output_offset1] = prefix1;
    }
  }
}
