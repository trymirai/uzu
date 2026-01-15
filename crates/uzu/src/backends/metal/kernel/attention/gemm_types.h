// Shared types for Gemm attention kernel - used by both Metal shaders and Rust
// (via bindgen). This header must be C-compatible for bindgen to parse it.
//
// NOTE: Only wrap namespaces for Metal builds. For C/bindgen builds, keep
// everything in the global namespace.
//
// SPDX-License-Identifier: MIT

#pragma once

// Metal and C have different type systems
// __METAL_VERSION__ is defined when compiling with the Metal shader compiler
#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;

namespace uzu {
namespace attention {

#else
#include <stdint.h>
#endif

///////////////////////////////////////////////////////////////////////////////
// Gemm Attention Parameters
///////////////////////////////////////////////////////////////////////////////

/// Parameters for the Gemm attention kernel.
///
/// All strides are in **elements**, not bytes.
/// Q/K/V/O are treated as 3D tensors with strides for:
/// - [0] batch
/// - [1] head (or kv-head for K/V)
/// - [2] sequence (row stride for [seq, head_dim])
struct AttnParams {
  int64_t q_strides[3];
  int64_t k_strides[3];
  int64_t v_strides[3];
  int64_t o_strides[3];

  int gqa_factor;
  float scale;

  // Query length (suffix length) and key length (prefix + suffix)
  int q_len;
  int k_len;

  // Absolute offset of the first query token in the full key sequence.
  // For LLM decode/prefill this is the segment prefix length.
  int q_off;

  // Query tiling metadata (BQ tiles)
  int nq_aligned; // q_len / BQ
  int q_rem;      // q_len % BQ

  // Key tiling metadata (BK tiles)
  int nk;         // ceil(k_len / BK)
  int nk_aligned; // k_len / BK
  int k_rem;      // k_len % BK
};

/// Parameters describing the additive attention mask.
///
/// Mask is treated as a 2D matrix [qL, kL] with:
/// - per-batch base offset (M_strides[0])
/// - per-head base offset  (M_strides[1])
/// - row stride in elements (M_strides[2])
/// Column stride is assumed to be 1.
struct AttnMaskParams {
  int64_t m_strides[3];
};

// Close Metal namespace
#ifdef __METAL_VERSION__
} // namespace attention
} // namespace uzu
#endif
