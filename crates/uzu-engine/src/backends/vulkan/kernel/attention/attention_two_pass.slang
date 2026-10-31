import definitions;
import common.activation;
import attention.attention;
import attention.mask;
import common.workgroup_reduction;
import generated.attention_two_pass;
import generated.ring;
import generated.trie;

// Pass 1: block b of ATTENTION_TWO_PASS_BLOCKS takes keys b, b + ATTENTION_TWO_PASS_BLOCKS, ...; each simdgroup holds one query head, lane l owning
// HEAD_DIM / 32 consecutive dimensions, and writes its block's FP32 partial output, sum and maximum, partial
// (head * suffix_length + query) * blocks + block. Pass 2: per head and query, invocation i merges dimension i of the
// `blocks` blocks, then applies the gate.

static const uint TWO_PASS_SIMDGROUPS = 8;

[[Kernel]]
[[Precondition("new", "HEAD_DIM == 64 || HEAD_DIM == 128 || HEAD_DIM == 256 || HEAD_DIM == 512")]]
[[Precondition("encode", "gqa_factor > 0")]]
[[Precondition("encode", "suffix_length <= sequence_length")]]
func AttentionTwoPass1<T: IModelFloat>(
  Ptr<T, Access.Read> queries,
  Ptr<T, Access.Read> keys,
  Ptr<T, Access.Read> values,
  Ptr<float, Access.ReadWrite> out,
  Ptr<float, Access.ReadWrite> sums,
  Ptr<float, Access.ReadWrite> maxs,
  uint gqa_factor,
  uint sequence_length,
  uint k_head_stride,
  uint k_seq_stride,
  uint v_head_stride,
  uint v_seq_stride,
  [[Optional("is_kv_cache_ring")]] RingParams ring_params,
  float scale,
  uint num_heads,
  uint suffix_length,
  [[Optional("is_trie")]] Ptr<TrieNode, Access.Read> trie,
  [[Optional("is_sliding_window")]] uint sliding_window_size,
  [[Optional("has_sinks")]] Ptr<T, Access.Read> sinks,
  [[Groups("num_heads.div_ceil(8)")]] uint head_group,
  [[Groups("suffix_length")]] uint query,
  [[Groups("crate::backends::vulkan::gpu_types::attention_two_pass::ATTENTION_TWO_PASS_BLOCKS")]] uint block,
  [[Threads("256")]] uint tid,
  [[Specialize]] uint HEAD_DIM,
  [[Specialize]] bool has_sinks,
  [[Specialize]] bool is_kv_cache_ring,
  [[Specialize]] bool is_causal,
  [[Specialize]] bool is_trie,
  [[Specialize]] bool is_sliding_window,
) {
  let invocation = subgroup_invocation();
  let simdgroup = invocation / SIMD_SIZE;
  let lane = invocation % SIMD_SIZE;
  let elements = HEAD_DIM / SIMD_SIZE;
  let head = head_group * TWO_PASS_SIMDGROUPS + simdgroup;
  // Simdgroups past the last head take every barrier the reductions need.
  let active = head < num_heads;
  let kv_head = (active ? head : 0) / gqa_factor;
  let q_offset = head * suffix_length + query;
  let mask = AttentionMask(
    sequence_length, suffix_length, ring_params, trie, sliding_window_size, is_kv_cache_ring, is_causal, is_trie,
    is_sliding_window
  );
  let query_position = mask.query_position(query);
  let first = lane * elements;

  float q[ATTENTION_DIMS];
  float o[ATTENTION_DIMS];
  for (uint i = 0; i < elements; i++) {
    q[i] = active ? scale * queries[q_offset * HEAD_DIM + first + i].toFloat() : 0.0;
    o[i] = 0.0;
  }
  var max_score = -1e9;
  var sum_exp_score = 0.0;
  if (active && has_sinks && block == 0) {
    max_score = sinks[head].toFloat();
    sum_exp_score = 1.0;
  }

  for (uint key = block; key < sequence_length; key += ATTENTION_TWO_PASS_BLOCKS) {
    // The mask depends only on the query and the key, so it is uniform across the workgroup.
    if (mask.admits(key, query, query_position)) {
      let key_row = kv_head * k_head_stride + key * k_seq_stride + first;
      var score = 0.0;
      for (uint j = 0; j < elements; j++) {
        score += q[j] * keys[key_row + j].toFloat();
      }
      score = segment_reduce<Sum<float>, TWO_PASS_SIMDGROUPS * SIMD_SIZE>(score, SIMD_SIZE);

      let new_max = max(max_score, score);
      let factor = exp(max_score - new_max);
      let exp_score = exp(score - new_max);
      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      let value_row = kv_head * v_head_stride + key * v_seq_stride + first;
      for (uint j = 0; j < elements; j++) {
        o[j] = o[j] * factor + exp_score * values[value_row + j].toFloat();
      }
    }
  }

  if (active) {
    let partial = q_offset * ATTENTION_TWO_PASS_BLOCKS + block;
    if (lane == 0) {
      sums[partial] = sum_exp_score;
      maxs[partial] = max_score;
    }
    for (uint i = 0; i < elements; i++) {
      out[partial * HEAD_DIM + first + i] = o[i];
    }
  }
}

[[Kernel]]
[[Precondition("new", "HEAD_DIM == 64 || HEAD_DIM == 128 || HEAD_DIM == 256 || HEAD_DIM == 512")]]
[[Precondition("encode", "blocks > 0 && blocks <= crate::backends::vulkan::gpu_types::attention_two_pass::ATTENTION_TWO_PASS_BLOCKS")]]
func AttentionTwoPass2<T: IModelFloat>(
  Ptr<float, Access.Read> partials,
  Ptr<float, Access.Read> sums,
  Ptr<float, Access.Read> maxs,
  Ptr<T, Access.ReadWrite> out,
  [[Optional("has_gate")]] Ptr<T, Access.Read> gate,
  uint num_heads,
  uint suffix_length,
  [[Optional("has_gate")]] uint gate_row_stride,
  uint blocks,
  [[Groups("num_heads")]] uint head,
  [[Groups("suffix_length")]] uint query,
  [[Threads("HEAD_DIM")]] uint channel,
  [[Specialize]] uint HEAD_DIM,
  [[Specialize]] bool has_gate,
) {
  let base = (head * suffix_length + query) * blocks;
  var new_max = maxs[base];
  for (uint block = 1; block < blocks; block++) {
    new_max = max(new_max, maxs[base + block]);
  }
  var sum_exp_score = 0.0;
  var o = 0.0;
  for (uint block = 0; block < blocks; block++) {
    let factor = exp(maxs[base + block] - new_max);
    sum_exp_score = fma(sums[base + block], factor, sum_exp_score);
    o = fma(partials[(base + block) * HEAD_DIM + channel], factor, o);
  }

  var result = T.fromFloat(o / sum_exp_score);
  if (has_gate) {
    // The gate applies to the T-rounded result.
    result = T.fromFloat(result.toFloat() * sigmoid(gate[query * gate_row_stride + head * HEAD_DIM + channel].toFloat()));
  }
  out[(query * num_heads + head) * HEAD_DIM + channel] = result;
}
