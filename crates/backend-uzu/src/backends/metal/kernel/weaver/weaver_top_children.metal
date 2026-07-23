#include <metal_stdlib>
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../common/top_k.h"
#include "weaver_frontier.h"

using namespace metal;

METAL_FUNC bool weaver_better(uint score, uint token, uint index, uint best_score, uint best_token, uint best_index) {
  return score > best_score ||
         (score == best_score && (token < best_token || (token == best_token && index < best_index)));
}

PUBLIC KERNEL(WeaverTopChildren)(
    const device bfloat* residual_logits,
    const device float* candidate_scores,
    const device uint* candidate_ids,
    device uint* output_ids,
    device float* output_logprobs,
    constant uint& rows,
    constant uint& candidates,
    constant uint& children,
    threadgroup float reduce_float[weaver::TOP_CHILDREN_SIMDGROUPS],
    threadgroup uint reduce_score[weaver::TOP_CHILDREN_SIMDGROUPS],
    threadgroup uint reduce_token[weaver::TOP_CHILDREN_SIMDGROUPS],
    threadgroup uint reduce_index[weaver::TOP_CHILDREN_SIMDGROUPS],
    threadgroup float& logit_max,
    threadgroup float& log_sum,
    threadgroup uint& winner_score,
    threadgroup uint& winner_token,
    threadgroup uint& winner_index,
    const ThreadContext thread_context,
    const uint row GROUPS(rows),
    const uint lid THREADS(weaver::TOP_CHILDREN_THREADS)
) {
  if (candidates == 0 || candidates > weaver::CANDIDATES_MAX || children == 0 || children > candidates) {
    return;
  }

  const uint base = row * candidates;
  const uint first_index = lid;
  const uint second_index = lid + weaver::TOP_CHILDREN_THREADS;
  const bool first_valid = first_index < candidates;
  const bool second_valid = second_index < candidates;
  const float first_logit =
      first_valid ? candidate_scores[base + first_index] + float(residual_logits[base + first_index]) : -INFINITY;
  const float second_logit =
      second_valid ? candidate_scores[base + second_index] + float(residual_logits[base + second_index]) : -INFINITY;
  const uint first_token = first_valid ? uint(candidate_ids[base + first_index]) : 0xffffffffu;
  const uint second_token = second_valid ? uint(candidate_ids[base + second_index]) : 0xffffffffu;
  const uint first_score = first_valid ? top_k_score_key(first_logit) : 0u;
  const uint second_score = second_valid ? top_k_score_key(second_logit) : 0u;
  bool first_active = first_valid;
  bool second_active = second_valid;

  const float local_max = fmax(first_logit, second_logit);
  const float simd_maximum = simd_max(local_max);
  if (thread_context.simd_lane_id == 0) {
    reduce_float[thread_context.simdgroup_index] = simd_maximum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_context.simdgroup_index == 0) {
    const float group_value = thread_context.simd_lane_id < weaver::TOP_CHILDREN_SIMDGROUPS
                                  ? reduce_float[thread_context.simd_lane_id]
                                  : -INFINITY;
    const float maximum = simd_max(group_value);
    if (thread_context.simd_lane_id == 0) {
      logit_max = maximum;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float local_sum =
      (first_valid ? exp(first_logit - logit_max) : 0.0f) + (second_valid ? exp(second_logit - logit_max) : 0.0f);
  const float simd_total = simd_sum(local_sum);
  if (thread_context.simd_lane_id == 0) {
    reduce_float[thread_context.simdgroup_index] = simd_total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_context.simdgroup_index == 0) {
    const float group_value = thread_context.simd_lane_id < weaver::TOP_CHILDREN_SIMDGROUPS
                                  ? reduce_float[thread_context.simd_lane_id]
                                  : 0.0f;
    const float total = simd_sum(group_value);
    if (thread_context.simd_lane_id == 0) {
      log_sum = log(total) + logit_max;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint child = 0; child < children; ++child) {
    uint local_score = 0u;
    uint local_token = 0xffffffffu;
    uint local_index = 0xffffffffu;
    if (first_active) {
      local_score = first_score;
      local_token = first_token;
      local_index = first_index;
    }
    if (second_active &&
        weaver_better(second_score, second_token, second_index, local_score, local_token, local_index)) {
      local_score = second_score;
      local_token = second_token;
      local_index = second_index;
    }

    const uint simd_score = simd_max(local_score);
    const uint simd_token = simd_min(local_score == simd_score ? local_token : 0xffffffffu);
    const uint simd_index =
        simd_min(local_score == simd_score && local_token == simd_token ? local_index : 0xffffffffu);
    if (thread_context.simd_lane_id == 0) {
      reduce_score[thread_context.simdgroup_index] = simd_score;
      reduce_token[thread_context.simdgroup_index] = simd_token;
      reduce_index[thread_context.simdgroup_index] = simd_index;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (thread_context.simdgroup_index == 0) {
      const bool lane_valid = thread_context.simd_lane_id < weaver::TOP_CHILDREN_SIMDGROUPS;
      const uint group_score = lane_valid ? reduce_score[thread_context.simd_lane_id] : 0u;
      const uint selected_score = simd_max(group_score);
      const uint group_token =
          lane_valid && group_score == selected_score ? reduce_token[thread_context.simd_lane_id] : 0xffffffffu;
      const uint selected_token = simd_min(group_token);
      const uint group_index = lane_valid && group_score == selected_score && group_token == selected_token
                                   ? reduce_index[thread_context.simd_lane_id]
                                   : 0xffffffffu;
      const uint selected_index = simd_min(group_index);
      if (thread_context.simd_lane_id == 0) {
        winner_score = selected_score;
        winner_token = selected_token;
        winner_index = selected_index;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0) {
      const float winner_logit = top_k_score_from_key(winner_score);
      output_ids[row * children + child] = winner_token;
      output_logprobs[row * children + child] = winner_logit - log_sum;
    }
    first_active = first_active && first_index != winner_index;
    second_active = second_active && second_index != winner_index;
  }
}
