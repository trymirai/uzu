#include <metal_stdlib>

#include "../common/defines.h"
#include "../common/dsl.h"
#include "expert_common.h"

using namespace metal;
using namespace uzu::moe;

#define LOGITS_BLOCK 32
#define LOGITS_BK 32
#define LOGITS_STRIDE (LOGITS_BK + 4)
#define LOGITS_SIMDGROUPS 4

// Router logits of a large batch in float32: logits[token, expert] = router_weights[expert] . input[token]
// (+ router_biases[expert]). A threadgroup computes a 32 x 32 block on 8 x 8 simdgroup matrices; each of its four
// simdgroups owns a 16 x 16 quarter. Requires experts % 32 == 0 and model_dim % 32 == 0.
template <typename T>
VARIANTS(T, bfloat)
PUBLIC KERNEL(MoeRouterLogits)(
    const device T* input,
    const device float* router_weights,
    const device float* router_biases OPTIONAL(has_router_biases),
    device float* logits,
    const constant uint& tokens,
    const constant uint& model_dim,
    const constant uint& experts,
    const bool has_router_biases SPECIALIZE,
    threadgroup float input_tile[LOGITS_BLOCK * LOGITS_STRIDE],
    threadgroup float weight_tile[LOGITS_BLOCK * LOGITS_STRIDE],
    threadgroup float output_tile[LOGITS_BLOCK * LOGITS_BLOCK],
    const uint token_block GROUPS(tokens.div_ceil(LOGITS_BLOCK)),
    const uint expert_block GROUPS(experts / LOGITS_BLOCK),
    const uint lane THREADS(METAL_SIMD_SIZE),
    const uint simdgroup THREADS(LOGITS_SIMDGROUPS)
) {
  const uint thread_index = simdgroup * METAL_SIMD_SIZE + lane;
  const uint first_token = token_block * LOGITS_BLOCK;
  const uint first_expert = expert_block * LOGITS_BLOCK;
  const uint quarter_row = (simdgroup / 2) * 16;
  const uint quarter_col = (simdgroup % 2) * 16;

  simdgroup_float8x8 acc[2][2];
  METAL_PRAGMA_UNROLL
  for (uint i = 0; i < 2; ++i) {
    METAL_PRAGMA_UNROLL
    for (uint j = 0; j < 2; ++j) {
      acc[i][j] = simdgroup_float8x8(0.0f);
    }
  }

  // Each of the 128 threads stages 8 inputs and 8 weights per step: row thread / 4, columns 8 * (thread % 4).
  const uint stage_row = thread_index / 4;
  const uint stage_col = 8 * (thread_index % 4);
  const uint token = first_token + stage_row;
  const bool token_valid = token < tokens;
  const device T* input_row = input + size_t(min(token, tokens - 1)) * model_dim + stage_col;
  const device float* weight_row = router_weights + size_t(first_expert + stage_row) * model_dim + stage_col;

  for (uint k = 0; k < model_dim; k += LOGITS_BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float values[8];
    load_values8(input_row + k, values);
    const float4 w0 = *reinterpret_cast<const device float4*>(weight_row + k);
    const float4 w1 = *reinterpret_cast<const device float4*>(weight_row + k + 4);
    threadgroup float* input_destination = input_tile + stage_row * LOGITS_STRIDE + stage_col;
    threadgroup float* weight_destination = weight_tile + stage_row * LOGITS_STRIDE + stage_col;
    METAL_PRAGMA_UNROLL
    for (uint i = 0; i < 8; ++i) {
      input_destination[i] = token_valid ? values[i] : 0.0f;
    }
    *reinterpret_cast<threadgroup float4*>(weight_destination) = w0;
    *reinterpret_cast<threadgroup float4*>(weight_destination + 4) = w1;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    METAL_PRAGMA_UNROLL
    for (uint kk = 0; kk < LOGITS_BK; kk += 8) {
      simdgroup_float8x8 a[2];
      simdgroup_float8x8 b[2];
      METAL_PRAGMA_UNROLL
      for (uint i = 0; i < 2; ++i) {
        simdgroup_load(a[i], input_tile + (quarter_row + 8 * i) * LOGITS_STRIDE + kk, LOGITS_STRIDE);
        // The weight tile is [expert][k]; loading it transposed gives the [k][expert] right operand.
        simdgroup_load(
            b[i],
            weight_tile + (quarter_col + 8 * i) * LOGITS_STRIDE + kk,
            LOGITS_STRIDE,
            ulong2(0, 0),
            true
        );
      }
      METAL_PRAGMA_UNROLL
      for (uint i = 0; i < 2; ++i) {
        METAL_PRAGMA_UNROLL
        for (uint j = 0; j < 2; ++j) {
          simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
      }
    }
  }

  METAL_PRAGMA_UNROLL
  for (uint i = 0; i < 2; ++i) {
    METAL_PRAGMA_UNROLL
    for (uint j = 0; j < 2; ++j) {
      simdgroup_store(
          acc[i][j],
          output_tile + (quarter_row + 8 * i) * LOGITS_BLOCK + quarter_col + 8 * j,
          LOGITS_BLOCK
      );
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint index = thread_index; index < LOGITS_BLOCK * LOGITS_BLOCK; index += LOGITS_SIMDGROUPS * METAL_SIMD_SIZE) {
    const uint row = index / LOGITS_BLOCK;
    const uint col = index % LOGITS_BLOCK;
    if (first_token + row < tokens) {
      float logit = output_tile[index];
      if (has_router_biases) {
        logit += router_biases[first_expert + col];
      }
      logits[size_t(first_token + row) * experts + first_expert + col] = logit;
    }
  }
}
