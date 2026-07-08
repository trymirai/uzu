#include <metal_stdlib>
#include "../../common/defines.h"
#include "../../common/dsl.h"

using namespace metal;

KERNEL(DeltaNetChunkedCumsum)(
    device const float* log_decay,
    device float* g_out,
    constant const uint& num_v_heads,
    constant const uint& suffix_len,
    constant const uint& chunk_size,
    const uint chunk_idx GROUPS(suffix_len.div_ceil(chunk_size)),
    const uint hv_idx GROUPS(num_v_heads),
    const uint lane THREADS(METAL_SIMD_SIZE)
) {
  if (lane != 0) {
    return;
  }

  const uint token_begin = chunk_idx * chunk_size;
  const uint token_end = min(token_begin + chunk_size, suffix_len);
  float acc = 0.0f;
  for (uint token = token_begin; token < token_end; ++token) {
    const uint offset = token * num_v_heads + hv_idx;
    acc += log_decay[offset];
    g_out[offset] = acc;
  }
}
