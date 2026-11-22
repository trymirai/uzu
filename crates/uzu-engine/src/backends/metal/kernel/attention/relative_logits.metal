#include <metal_stdlib>
#include "../common/defines.h"
#include "../common/dsl.h"
#include "../generated/log_scaling.h"
#include "../generated/trie.h"

using namespace metal;

#define RELATIVE_LOGITS_THREADS 256u
#define RELATIVE_LOGITS_MAX_DIM 32u

// Each token's relative states mix bias-versus-distance profiles into one attention logit bias per distance:
// logits[head][token][distance] = sum_c r[c] * profiles[c][distance], where r is the head's `dim` states, scaled by
// `clip_scales`, clipped to an L2 norm of `clip_threshold` and optionally log scaled by position. One threadgroup per
// token and head.
template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(AttentionRelativeLogits)(
    device const T* states, // [token, state_row_stride], head h at h * dim
    device const float* clip_scales, // [head, dim]
    device const float* profiles, // [dim, extent]
    device bfloat* logits, // [head, token, extent]
    const device uzu::trie::TrieNode* trie OPTIONAL(has_trie),
    constant uint& token_count,
    constant uint& state_row_stride,
    constant float& clip_threshold,
    const constant uzu::log_scaling::LogScalingParams& log_scaling OPTIONAL(has_log_scaling),
    const uint num_heads SPECIALIZE,
    const uint dim SPECIALIZE,
    const uint extent SPECIALIZE,
    const bool has_log_scaling SPECIALIZE,
    const bool has_trie SPECIALIZE,
    threadgroup float shared_states[RELATIVE_LOGITS_MAX_DIM],
    const uint token GROUPS(token_count),
    const uint head GROUPS(num_heads),
    const uint thread_index THREADS(RELATIVE_LOGITS_THREADS)
) {
  if (thread_index < METAL_SIMD_SIZE) {
    const uint channel = thread_index;
    const float value =
        channel < dim
            ? float(states[size_t(token) * state_row_stride + head * dim + channel]) * clip_scales[head * dim + channel]
            : 0.0f;
    const float norm = metal::sqrt(simd_sum(value * value));
    float scale = metal::min(clip_threshold / (norm + 1e-6f), 1.0f);
    if (has_log_scaling) {
      const uint position = log_scaling.position_offset + (has_trie ? trie[token].height : token);
      scale *= 1.0f + log_scaling.alpha * metal::log(metal::max(float(position + 1) / log_scaling.n_floor, 1.0f));
    }
    if (channel < dim) {
      shared_states[channel] = value * scale;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  device bfloat* row = logits + (size_t(head) * token_count + token) * extent;
  for (uint distance = thread_index; distance < extent; distance += RELATIVE_LOGITS_THREADS) {
    float sum = 0.0f;
    for (uint channel = 0; channel < dim; ++channel) {
      sum += shared_states[channel] * profiles[channel * extent + distance];
    }
    row[distance] = bfloat(sum);
  }
}
