#include <metal_stdlib>
#include <metal_simdgroup>
#include "../definitions.metal"

#define SIMD_WIDTH 32
#define SEQUENCE_BLOCK_SIZE 32
#define HEAD_BLOCK_SIZE 32

template <typename T, uint head_dim>
VARIANTS(T, float, half, bfloat)
VARIANTS(head_dim, 64, 128, 256)
KERNEL(AttentionSinglePass)(
    const device T* queries,
    const device T* keys,
    const device T* values,
    device T* out,
    const constant uint& gqa_factor,
    const constant uint& sequence_length,
    const constant uint& k_head_stride,
    const constant uint& k_seq_stride,
    const constant uint& v_head_stride,
    const constant uint& v_seq_stride,
    const constant float& scale,
    const device T* fmask OPTIONAL(float_mask),
    const constant uint& mask_kv_seq_stride OPTIONAL(has_mask),
    const constant uint& mask_q_seq_stride OPTIONAL(has_mask),
    const constant uint& mask_head_stride OPTIONAL(has_mask),
    const device float* sinks OPTIONAL(has_sinks),
    const constant uint& num_heads,
    const constant uint& suffix_length,
    threadgroup float shared_max_scores[SEQUENCE_BLOCK_SIZE * HEAD_BLOCK_SIZE],
    threadgroup float shared_sum_exp_scores[SEQUENCE_BLOCK_SIZE * HEAD_BLOCK_SIZE],
    threadgroup float shared_outputs[SEQUENCE_BLOCK_SIZE * HEAD_BLOCK_SIZE],
    const bool float_mask SPECIALIZE,
    const bool has_mask SPECIALIZE,
    const bool has_sinks SPECIALIZE,
    const bool do_causal SPECIALIZE,
    const uint head_idx GROUPS(num_heads),
    const uint q_seq_idx GROUPS(suffix_length),
    const uint tid THREADS(1024)
) {
  const uint3 tpg = {num_heads, suffix_length, 1};
  const uint simd_gid = tid / SIMD_WIDTH;
  const uint simd_lid = tid % SIMD_WIDTH;
  constexpr bool bool_mask = false;
  constexpr bool query_transposed = false;

  constexpr uint value_dim = head_dim;
  constexpr uint qk_elements_per_thread = head_dim / HEAD_BLOCK_SIZE;
  constexpr uint value_elements_per_thread = value_dim / HEAD_BLOCK_SIZE;
  uint inner_k_stride = SEQUENCE_BLOCK_SIZE * int(k_seq_stride);
  uint inner_v_stride = SEQUENCE_BLOCK_SIZE * int(v_seq_stride);

  typedef float U;

  thread U q[qk_elements_per_thread];
  thread U k[qk_elements_per_thread];
  thread U o[value_elements_per_thread];

  const uint kv_head_idx = head_idx / gqa_factor;
  const uint o_offset = q_seq_idx * tpg.x + head_idx;
  const uint q_offset = query_transposed ? tpg.x * q_seq_idx + head_idx
                                         : head_idx * tpg.y + q_seq_idx;

  queries += q_offset * head_dim + simd_lid * qk_elements_per_thread;
  keys += kv_head_idx * k_head_stride + simd_gid * k_seq_stride +
          simd_lid * qk_elements_per_thread;
  values += kv_head_idx * v_head_stride + simd_gid * v_seq_stride +
            simd_lid * value_elements_per_thread;

  if (bool_mask) {
    // bmask += head_idx * mask_head_stride + simd_gid * mask_kv_seq_stride +
    //         q_seq_idx * mask_q_seq_stride;
  }
  if (float_mask) {
    fmask += head_idx * mask_head_stride + simd_gid * mask_kv_seq_stride +
             q_seq_idx * mask_q_seq_stride;
  }

  out += o_offset * value_dim + simd_gid * value_elements_per_thread;

  // Read the query and 0 the output accumulator
  for (int i = 0; i < qk_elements_per_thread; i++) {
    q[i] = static_cast<U>(scale) * queries[i];
  }
  for (int i = 0; i < value_elements_per_thread; i++) {
    o[i] = 0;
  }

  U max_score = -INFINITY;
  U sum_exp_score = 0;
  if (has_sinks && simd_gid == 0) {
    const int num_q_heads = static_cast<int>(tpg.x);
    int q_head_idx = head_idx % num_q_heads;
    max_score = static_cast<U>(sinks[q_head_idx]);
    sum_exp_score = 1;
  }

  // For each key
  for (uint i = simd_gid; i < sequence_length; i += SEQUENCE_BLOCK_SIZE) {
    bool use_key = true;
    if (do_causal) {
      use_key = i <= (sequence_length - tpg.y + q_seq_idx);
    } else if (bool_mask) {
      // use_key = bmask[0];
    }

    if (use_key) {
      // Read the key
      for (uint j = 0; j < qk_elements_per_thread; j++) {
        k[j] = keys[j];
      }

      // Compute the i-th score
      U score = 0;
      for (uint j = 0; j < qk_elements_per_thread; j++) {
        score += q[j] * k[j];
      }
      score = simd_sum(score);
      if (float_mask) {
        score += max(-1e9f, static_cast<U>(fmask[0]));
      }

      // Update the accumulators
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Update the output accumulator
      for (int j = 0; j < value_elements_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * values[j];
      }
    }

    // Move the pointers to the next kv
    keys += inner_k_stride;
    values += inner_v_stride;
    if (bool_mask) {
      // bmask += SEQUENCE_BLOCK_SIZE * mask_kv_seq_stride;
    }
    if (float_mask) {
      fmask += SEQUENCE_BLOCK_SIZE * mask_kv_seq_stride;
    }
  }

  // Each thread has a partial part of the output so we need to combine them.
  if (simd_lid == 0) {
    shared_max_scores[simd_gid] = max_score;
    shared_sum_exp_scores[simd_gid] = sum_exp_score;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  max_score = shared_max_scores[simd_lid];
  U new_max = simd_max(max_score);
  U factor = fast::exp(max_score - new_max);
  sum_exp_score = simd_sum(shared_sum_exp_scores[simd_lid] * factor);

  // Now we need to aggregate all the outputs
  for (int i = 0; i < value_elements_per_thread; i++) {
    shared_outputs[simd_lid * HEAD_BLOCK_SIZE + simd_gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(
               shared_outputs[simd_gid * HEAD_BLOCK_SIZE + simd_lid] * factor
           ) /
           sum_exp_score;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // And write the output
  if (simd_lid == 0) {
    for (int i = 0; i < value_elements_per_thread; i++) {
      out[i] = static_cast<T>(o[i]);
    }
  }
}