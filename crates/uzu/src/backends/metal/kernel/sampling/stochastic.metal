#include <metal_stdlib>
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../common/threadgroup_reduce.h"
#include "../rng.metal"

using namespace metal;

// ── Tuning constants
// ──────────────────────────────────────────────────────────
#define BLOCK_SIZE 1024
#define BLOCK_SIZE_IN_SIMDS (BLOCK_SIZE / 32) // 32

#define BITS_IN_U32 32u

// Gumbel access layout; must match gumbel.rs for PARD compatibility.
#define G_GRAIN_SIZE 64
#define G_WORDS_PER_OFFSET 4

// Number of retained candidates after the noisy local reductions.
#define N_CANDIDATES (BLOCK_SIZE_IN_SIMDS * 2) // 64

// ── Threadgroup shared memory
// ─────────────────────────────────────────────────
//   buffer[BLOCK_SIZE]   float[1024]   4096 B
//     Pass A: raking max reduce scratch.
//     Pass B: 2 × N_CANDIDATES Candidate entries for candidates + merge sort
//     scratch.
//   max_logit                   float            4 B
//   Total: 4100 B  (limit 32 KB)

struct SharedState {
  float buffer[BLOCK_SIZE];
  float max_logit;
};

// All fields are float: bfloat is a storage-only type in Metal and does not
// support simd_max / fmax / threadgroup_raking_reduce_max.
struct Candidate {
  float noisy;
  float logit;
  uint idx;
};

#define FUSED_SHARED_SIZE (BLOCK_SIZE + 1) // 1025 floats = 4100 bytes

// Stochastic sampling
//
// Pass A computes max_logit for stable exp/min_p thresholds.
// Pass B adds index-based Gumbel noise, keeps local top-2 per thread, then
// reduces to N_CANDIDATES candidates in threadgroup memory.
// Thread 0 sorts those candidates by logit, applies top_k/top_p/min_p within
// the retained set, and returns the highest-noisy surviving token.

// Index-based Gumbel noise matching gumbel.rs::gumbel_float(seed, revidx(idx)).
inline float gumbel_noise_at(uint64_t seed, uint token_idx) {
  const uint BLOCKGRAIN = BLOCK_SIZE * G_GRAIN_SIZE; // 65536
  uint global_idx = token_idx / BLOCKGRAIN;
  uint idx_in_bg = token_idx % BLOCKGRAIN;
  uint grain_idx = idx_in_bg / BLOCK_SIZE; // [0, G_GRAIN_SIZE)
  uint local_idx = idx_in_bg % BLOCK_SIZE;

  uint rng_offset = (global_idx * BLOCK_SIZE + local_idx) *
                        (G_GRAIN_SIZE + G_WORDS_PER_OFFSET - 1) /
                        G_WORDS_PER_OFFSET +
                    grain_idx / G_WORDS_PER_OFFSET;
  uint rng_word = grain_idx % G_WORDS_PER_OFFSET;

  float u = uniform_float_stateless(seed, rng_offset, rng_word);
  return -fast::log(-fast::log(u));
}

// Bottom-up merge sort of Candidate[N_CANDIDATES] by .logit descending.
// Uses `scratch` as a ping-pong buffer.
inline void merge_sort_by_logit_desc(
    threadgroup Candidate* src,
    threadgroup Candidate* scratch
) {
  uint pass = 0u;
  for (uint width = 2u; width < N_CANDIDATES; width <<= 1u, ++pass) {
    threadgroup Candidate* s = (pass & 1u) ? scratch : src;
    threadgroup Candidate* d = (pass & 1u) ? src : scratch;
    for (uint lo = 0u; lo < N_CANDIDATES; lo += width << 1u) {
      uint mid = lo + width, hi = lo + (width << 1u);
      for (uint i = lo, j = mid, k = lo; k < hi; ++k) {
        bool pick_left = (i < mid) && (j >= hi || s[i].logit >= s[j].logit);
        d[k] = pick_left ? s[i++] : s[j++];
      }
    }
  }
  for (uint i = 0u; i < N_CANDIDATES; ++i)
    src[i] = scratch[i];
}

// Returns the scaled logit, or -INFINITY if masked.
template <typename T>
inline float masked_logit(
    device const T* logits,
    device const uint32_t* bitmask,
    uint batch_start,
    uint batch_idx,
    uint bitmask_stride,
    bool has_bitmask,
    uint token_idx
) {
  if (has_bitmask) {
    uint bitmask_idx = batch_idx * bitmask_stride + (token_idx / BITS_IN_U32);
    if (!((bitmask[bitmask_idx] >> (token_idx % BITS_IN_U32)) & 0b1))
      return -INFINITY;
  }
  return float(logits[batch_start + token_idx]);
}

template <typename T>
VARIANTS(T, float, half, bfloat)
PUBLIC KERNEL(Stochastic)(
    device const T*          logits,
    device const uint64_t*   batch_seeds,
    device uint32_t*         sampled_tokens,
    device const uint32_t*   bitmask        OPTIONAL(has_bitmask),
    threadgroup float         shared[FUSED_SHARED_SIZE],
    constant uint&            batch_size,
    constant uint&            vocab_size,
    constant float&           temperature,
    constant uint&            top_k,
    constant float&           top_p,
    constant float&           min_p,
    const ThreadContext       simd,
    uint batch_idx            GROUPS(batch_size),
    uint thread_idx           THREADS(BLOCK_SIZE),
    const bool has_bitmask    SPECIALIZE
) {
  const uint lane_idx = simd.simdgroup_index;
  const uint batch_start = batch_idx * vocab_size;
  const uint64_t seed = batch_seeds[batch_idx];
  const float inv_temp = 1.0f / temperature;
  const uint bitmask_stride = (vocab_size + BITS_IN_U32 - 1u) / BITS_IN_U32;

  threadgroup SharedState* state = (threadgroup SharedState*)shared;
  threadgroup float* buffer = state->buffer;

  // Pass A: max_logit
  float local_max = -INFINITY;
  for (uint idx = thread_idx; idx < vocab_size; idx += BLOCK_SIZE) {
    float v = float(
        T(masked_logit(
              logits,
              bitmask,
              batch_start,
              batch_idx,
              bitmask_stride,
              has_bitmask,
              idx
          ) *
          inv_temp)
    );
    if (!isinf(v))
      local_max = fmax(local_max, v);
  }

  float max_logit = threadgroup_raking_reduce_max<BLOCK_SIZE>(
      local_max,
      buffer,
      (ushort)thread_idx
  );

  if (max_logit == -INFINITY) {
    if (thread_idx == 0u)
      sampled_tokens[batch_idx] = 0u;
    return;
  }

  if (thread_idx == 0u)
    state->max_logit = max_logit;

  // Pass B: Gumbel + per-thread top-2
  float best_noisy0 = -INFINITY, best_logit0 = -INFINITY;
  uint best_idx0 = 0u;
  float best_noisy1 = -INFINITY, best_logit1 = -INFINITY;
  uint best_idx1 = 0u;

  for (uint idx = thread_idx; idx < vocab_size; idx += BLOCK_SIZE) {
    // Keep T precision for compatibility with the previous path.
    T logit =
        T(masked_logit(
              logits,
              bitmask,
              batch_start,
              batch_idx,
              bitmask_stride,
              has_bitmask,
              idx
          ) *
          inv_temp);
    if (!isinf(logit)) {
      float noisy = logit + T(gumbel_noise_at(seed, idx));
      if (noisy > best_noisy0) {
        best_noisy1 = best_noisy0;
        best_logit1 = best_logit0;
        best_idx1 = best_idx0;
        best_noisy0 = noisy;
        best_logit0 = logit;
        best_idx0 = idx;
      } else if (noisy > best_noisy1) {
        best_noisy1 = noisy;
        best_logit1 = logit;
        best_idx1 = idx;
      }
    }
  }

  // Simd reduce: per-simd top-2 -> buffer
  {
    // Pass 1: best candidate per simd.
    float sg_max0 = simd_max(best_noisy0);
    uint winner_lane0 =
        simd_min(best_noisy0 >= sg_max0 ? lane_idx : 0xFFFFFFFFu);
    float sg_logit0 = simd_broadcast(best_logit0, (ushort)winner_lane0);
    uint sg_idx0 = simd_broadcast(best_idx0, (ushort)winner_lane0);

    // Pass 2: winner yields its 2nd best; everyone else yields its 1st.
    float runner_up_noisy =
        (lane_idx == winner_lane0) ? best_noisy1 : best_noisy0;
    float runner_up_logit =
        (lane_idx == winner_lane0) ? best_logit1 : best_logit0;
    uint runner_up_idx = (lane_idx == winner_lane0) ? best_idx1 : best_idx0;

    float sg_max1 = simd_max(runner_up_noisy);
    uint winner_lane1 =
        simd_min(runner_up_noisy >= sg_max1 ? lane_idx : 0xFFFFFFFFu);
    float sg_logit1 = simd_broadcast(runner_up_logit, (ushort)winner_lane1);
    uint sg_idx1 = simd_broadcast(runner_up_idx, (ushort)winner_lane1);

    if (lane_idx == 0u) {
      threadgroup Candidate* buf = (threadgroup Candidate*)buffer;
      uint base = simd.threadgroup_index * 2u;
      // Pre-sort the pair by logit for the merge pass below.
      Candidate c0 = {sg_max0, sg_logit0, sg_idx0};
      Candidate c1 = {sg_max1, sg_logit1, sg_idx1};
      buf[base + 0u] = (c0.logit >= c1.logit) ? c0 : c1;
      buf[base + 1u] = (c0.logit >= c1.logit) ? c1 : c0;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Filter retained candidates.
  if (thread_idx == 0u) {
    threadgroup Candidate* buf = (threadgroup Candidate*)buffer;
    const float max_logit = state->max_logit;

    // Scratch at buf + N_CANDIDATES stays within the shared buffer.
    merge_sort_by_logit_desc(buf, buf + N_CANDIDATES);

    uint k_limit = (top_k > 0u && top_k <= N_CANDIDATES) ? top_k : N_CANDIDATES;

    float sum_exp_K = 0.0f;
    for (uint i = 0u; i < k_limit; i++)
      sum_exp_K += fast::exp(buf[i].logit - max_logit);

    float threshold_top_k = (top_k > 0u && top_k <= N_CANDIDATES)
                                ? buf[k_limit - 1u].logit
                                : -INFINITY;

    float threshold_top_p = -INFINITY;
    if (top_p < 1.0f) {
      float cum = 0.0f;
      float target = top_p * sum_exp_K;
      for (uint i = 0u; i < k_limit; i++) {
        cum += fast::exp(buf[i].logit - max_logit);
        if (cum >= target) {
          threshold_top_p = buf[i].logit;
          break;
        }
      }
    }

    float threshold_min_p =
        (min_p > 0.0f) ? max_logit + fast::log(min_p) : -INFINITY;
    float threshold =
        fmax(fmax(threshold_top_k, threshold_top_p), threshold_min_p);

    float winner_noisy = -INFINITY;
    uint winner_idx = 0u;
    for (uint i = 0u; i < N_CANDIDATES; i++) {
      if (buf[i].logit >= threshold &&
          (buf[i].noisy > winner_noisy ||
           (buf[i].noisy == winner_noisy && buf[i].idx < winner_idx))) {
        winner_noisy = buf[i].noisy;
        winner_idx = buf[i].idx;
      }
    }
    sampled_tokens[batch_idx] = winner_idx;
  }
}
