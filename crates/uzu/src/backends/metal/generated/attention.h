// Auto-generated from gpu_types/attention.rs - do not edit manually
#pragma once

#ifndef UZU_ATTENTION_H
#define UZU_ATTENTION_H

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;

namespace uzu {
namespace attention {
#else
#include <stdint.h>
#endif

/**Parameters for the GEMM attention kernel. */
/**All strides are in **elements**, not bytes. */
/**Q/K/V/O are treated as 3D tensors with strides for: */
/**- `[0]` batch */
/**- `[1]` head (or kv-head for K/V) */
/**- `[2]` sequence (row stride for [seq, head_dim]) */
typedef struct {
  int64_t q_strides[3];
  int64_t k_strides[3];
  int64_t v_strides[3];
  int64_t o_strides[3];
  int32_t gqa_factor;
  float scale;
  /**Query length (suffix length) */
  int32_t q_len;
  /**Key length (prefix + suffix) */
  int32_t k_len;
  /**Absolute offset of the first query token in the full key sequence. */
  /**For LLM decode/prefill this is the segment prefix length. */
  int32_t q_off;
  /**Query tiling: q_len / BQ */
  int32_t nq_aligned;
  /**Query tiling: q_len % BQ */
  int32_t q_rem;
  /**Key tiling: ceil(k_len / BK) */
  int32_t nk;
  /**Key tiling: k_len / BK */
  int32_t nk_aligned;
  /**Key tiling: k_len % BK */
  int32_t k_rem;
} AttnParams;

/**Parameters describing the additive attention mask. */
/**Mask is treated as a 2D matrix [qL, kL] with: */
/**- per-batch base offset (`m_strides[0]`) */
/**- per-head base offset (`m_strides[1]`) */
/**- row stride in elements (`m_strides[2]`) */
/**Column stride is assumed to be 1. */
typedef struct {
  int64_t m_strides[3];
} AttnMaskParams;

#ifdef __METAL_VERSION__
} // namespace attention
} // namespace uzu
#endif

#endif // UZU_ATTENTION_H
