#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"

using namespace metal;

#define CHUNKED_APPLY_THREADS 128

static_assert(CHUNKED_APPLY_THREADS % METAL_SIMD_SIZE == 0, "thread count must be a multiple of simd size");

template <uint HEAD_K_DIM, uint CHUNK_SIZE>
VARIANTS(HEAD_K_DIM, 128)
VARIANTS(CHUNK_SIZE, 16, 32, 64)
PUBLIC KERNEL(DeltaNetChunkedStateA2DecayScale)(
    device const float* g,
    device float* decay_scale,
    constant const uint& num_v_heads,
    constant const uint& suffix_len,
    const uint chunk_idx GROUPS(suffix_len.div_ceil(CHUNK_SIZE)),
    const uint hv_idx GROUPS(num_v_heads),
    const uint tid THREADS(CHUNKED_APPLY_THREADS)
) {
  const uint token_base = chunk_idx * CHUNK_SIZE;
  const uint valid_tokens = token_base < suffix_len ? min(uint(CHUNK_SIZE), suffix_len - token_base) : 0u;
  if (valid_tokens == 0) {
    return;
  }

  const float g_last = g[(token_base + valid_tokens - 1) * num_v_heads + hv_idx];
  device float* dst_chunk = decay_scale + (chunk_idx * num_v_heads + hv_idx) * CHUNK_SIZE;

  for (uint local_t = tid; local_t < CHUNK_SIZE; local_t += CHUNKED_APPLY_THREADS) {
    if (local_t >= valid_tokens) {
      dst_chunk[local_t] = 0.0f;
      continue;
    }
    const uint token = token_base + local_t;
    dst_chunk[local_t] = fast::exp(g_last - g[token * num_v_heads + hv_idx]);
  }
}
