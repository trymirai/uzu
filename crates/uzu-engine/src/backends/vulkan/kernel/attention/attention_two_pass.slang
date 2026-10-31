import definitions;
import common.activation;
import attention.attention;
import attention.mask;
import common.workgroup_reduction;
import generated.attention_two_pass;
import generated.ring;
import generated.trie;

// Pass 1: block b of ATTENTION_TWO_PASS_BLOCKS takes keys b, b + ATTENTION_TWO_PASS_BLOCKS, ...; each simdgroup holds one
// query head, lane l owning HEAD_DIM / 32 consecutive dimensions of the INT8 codes, and writes its block's FP32 partial
// output, sum and maximum, -inf without an admitted key, to partial (head * suffix_length + query) * blocks + block.
// Pass 2: per head and query, the merge of the blocks' partials with any sink, then the inverse rotation and the gate.

[[Kernel]]
[[Precondition("new", "HEAD_DIM == 64 || HEAD_DIM == 128 || HEAD_DIM == 256 || HEAD_DIM == 512")]]
[[Precondition("encode", "gqa_factor > 0")]]
[[Precondition("encode", "suffix_length <= sequence_length")]]
func AttentionTwoPass1(
  Ptr<int8_t, Access.Read> q_codes,
  Ptr<bf16, Access.Read> q_scales,
  Ptr<int8_t, Access.Read> k_codes,
  Ptr<bf16, Access.Read> k_scales,
  Ptr<uint8_t, Access.Read> v_codes,
  Ptr<bf16, Access.Read> v_scales,
  Ptr<float, Access.ReadWrite> out,
  Ptr<float, Access.ReadWrite> sums,
  Ptr<float, Access.ReadWrite> maxs,
  uint gqa_factor,
  uint num_kv_heads,
  uint sequence_length,
  [[Optional("is_kv_cache_ring")]] RingParams ring_params,
  float scale,
  [[Optional("is_trie")]] Ptr<TrieNode, Access.Read> trie,
  [[Optional("is_sliding_window")]] uint sliding_window_size,
  uint num_heads,
  uint suffix_length,
  [[Groups("num_heads.div_ceil(crate::backends::vulkan::gpu_types::attention_two_pass::ATTENTION_TWO_PASS_SIMDGROUPS)")]] uint head_group,
  [[Groups("suffix_length")]] uint query,
  [[Groups("crate::backends::vulkan::gpu_types::attention_two_pass::ATTENTION_TWO_PASS_BLOCKS")]] uint block,
  [[Threads("256")]] uint tid,
  [[Specialize]] uint HEAD_DIM,
  [[Specialize]] bool is_kv_cache_ring,
  [[Specialize]] bool is_causal,
  [[Specialize]] bool is_trie,
  [[Specialize]] bool is_sliding_window,
) {
  static_assert(ATTENTION_TWO_PASS_SIMDGROUPS * SIMD_SIZE == 256, "a simdgroup per head in Threads(256)");
  let invocation = subgroup_invocation();
  let simdgroup = invocation / SIMD_SIZE;
  let lane = invocation % SIMD_SIZE;
  let elements = HEAD_DIM / SIMD_SIZE;
  let head = head_group * ATTENTION_TWO_PASS_SIMDGROUPS + simdgroup;
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
  var query_scale = 0.0;
  var query_words = uint4(0);
  if (active) {
    query_words = attention_code_words(q_codes, q_offset * HEAD_DIM + first, elements);
    query_scale = scale * q_scales[q_offset].toFloat();
  }
  for (uint i = 0; i < elements; i++) {
    q[i] = attention_key_code(query_words, i);
    o[i] = 0.0;
  }
  var max_score = NEGATIVE_INFINITY;
  var sum_exp_score = 0.0;

  for (uint key = block; key < sequence_length; key += ATTENTION_TWO_PASS_BLOCKS) {
    // The mask depends only on the query and the key, so it is uniform across the workgroup.
    if (mask.admits(key, query, query_position)) {
      let row = key * num_kv_heads + kv_head;
      let key_words = attention_code_words(k_codes, row * HEAD_DIM + first, elements);
      var dot = 0.0;
      for (uint j = 0; j < elements; j++) {
        dot += q[j] * attention_key_code(key_words, j);
      }
      dot = segment_reduce<Sum<float>, ATTENTION_TWO_PASS_SIMDGROUPS * SIMD_SIZE>(dot, SIMD_SIZE);
      let score = dot * query_scale * k_scales[row].toFloat();

      let new_max = max(max_score, score);
      let factor = exp(max_score - new_max);
      let exp_score = exp(score - new_max);
      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      let value_weight = exp_score * v_scales[row].toFloat();
      let value_words = attention_code_words(v_codes, row * HEAD_DIM + first, elements);
      for (uint j = 0; j < elements; j++) {
        o[j] = o[j] * factor + value_weight * attention_value_code(value_words, j);
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

// Merges a head and query's `blocks` partials with its sink, a key of value zero, into the attention output, 0 without
// any admitted key or sink, rotates it back, and multiplies the T-rounded result by the sigmoid of its gate. One group
// of 32 invocations, lane l holding dimensions [l HEAD_DIM / 32, (l + 1) HEAD_DIM / 32).
[[Variants("T", "float, bf16")]]
[[Kernel]]
[[Precondition("new", "HEAD_DIM == 64 || HEAD_DIM == 128 || HEAD_DIM == 256 || HEAD_DIM == 512")]]
[[Precondition("encode", "blocks > 0 && blocks <= crate::backends::vulkan::gpu_types::attention_two_pass::ATTENTION_TWO_PASS_BLOCKS")]]
func AttentionTwoPass2<T: IStorageFloat>(
  Ptr<float, Access.Read> partials,
  Ptr<float, Access.Read> sums,
  Ptr<float, Access.Read> maxs,
  Ptr<T, Access.ReadWrite> out,
  [[Optional("has_sinks")]] Ptr<T, Access.Read> sinks,
  [[Optional("has_gate")]] Ptr<T, Access.Read> gate,
  uint num_heads,
  uint suffix_length,
  [[Optional("has_gate")]] uint gate_row_stride,
  uint blocks,
  [[Groups("num_heads")]] uint head,
  [[Groups("suffix_length")]] uint query,
  [[Threads("32")]] uint tid,
  [[Specialize]] uint HEAD_DIM,
  [[Specialize]] bool has_sinks,
  [[Specialize]] bool has_gate,
) {
  let lane = subgroup_invocation();
  let dims = HEAD_DIM / SIMD_SIZE;
  let first = lane * dims;
  let base = (head * suffix_length + query) * blocks;
  var sink = NEGATIVE_INFINITY;
  if (has_sinks) {
    sink = sinks[head].toFloat();
  }
  var new_max = sink;
  for (uint block = 0; block < blocks; block++) {
    new_max = max(new_max, maxs[base + block]);
  }
  var sum_exp_score = has_sinks ? exp(sink - new_max) : 0.0;
  float values[ATTENTION_DIMS];
  for (uint a = 0; a < dims; a++) {
    values[a] = 0.0;
  }
  for (uint block = 0; block < blocks; block++) {
    let block_max = maxs[base + block];
    // A block without an admitted key adds nothing, even when no block or sink has one.
    let factor = block_max == NEGATIVE_INFINITY ? 0.0 : exp(block_max - new_max);
    sum_exp_score = fma(sums[base + block], factor, sum_exp_score);
    for (uint a = 0; a < dims; a++) {
      values[a] = fma(partials[(base + block) * HEAD_DIM + first + a], factor, values[a]);
    }
  }
  for (uint a = 0; a < dims; a++) {
    values[a] = sum_exp_score > 0.0 ? values[a] / sum_exp_score : 0.0;
  }
  attention_unrotate<32>(values, dims, lane);

  for (uint a = 0; a < dims; a++) {
    let channel = first + a;
    var result = T.fromFloat(values[a]);
    if (has_gate) {
      // The gate applies to the T-rounded result.
      let gate_value = gate[query * gate_row_stride + head * HEAD_DIM + channel].toFloat();
      result = T.fromFloat(result.toFloat() * sigmoid(gate_value));
    }
    out[(query * num_heads + head) * HEAD_DIM + channel] = result;
  }
}
