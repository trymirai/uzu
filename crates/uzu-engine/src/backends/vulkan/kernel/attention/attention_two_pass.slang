import definitions;
import common.activation;
import attention.mask;
import generated.ring;
import generated.trie;

// Counterpart of Metal attention/attention_two_pass.metal. Pass 1: TWO_PASS_BLOCKS blocks take keys block, block + 32,
// ... of every query; a workgroup of 8 simdgroups holds one query head per simdgroup, lane l owning HEAD_DIM / 32
// consecutive dimensions, an online softmax per head and block, written as FP32 partial outputs, sums and maxima.
// Pass 2: one workgroup of 32 simdgroups per query head and query, lane l taking block l, simdgroup s dimensions
// s HEAD_DIM / 32 on, merging the blocks and, with a gate, applying it as Metal's attention_apply_gate.
// Simdgroups are subgroups where the device runs full 32-lane subgroups, otherwise logical groups of 32 consecutive
// invocations whose reductions pass through shared memory; every invocation then reaches every barrier.

static const uint TWO_PASS_BLOCKS = 32;
static const uint TWO_PASS_SIMD = 32;
static const uint TWO_PASS_SIMDGROUPS = 8;
static const uint TWO_PASS_MERGE_THREADS = 1024;
static const uint TWO_PASS_MAX_ELEMENTS = 16;

groupshared float two_pass_lanes[TWO_PASS_MERGE_THREADS];

// The sum (or maximum) over the 32 lanes of a simdgroup, in every lane.
float two_pass_reduce(float value, bool maximum, bool native, uint simdgroup, uint lane) {
  if (native) {
    return maximum ? WaveActiveMax(value) : WaveActiveSum(value);
  }
  let base = simdgroup * TWO_PASS_SIMD;
  two_pass_lanes[base + lane] = value;
  GroupMemoryBarrierWithGroupSync();
  var result = two_pass_lanes[base];
  for (uint other = 1; other < TWO_PASS_SIMD; other++) {
    let entry = two_pass_lanes[base + other];
    result = maximum ? max(result, entry) : result + entry;
  }
  GroupMemoryBarrierWithGroupSync();
  return result;
}

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
  [[Groups("32")]] uint block,
  [[Threads("256")]] uint tid,
  [[Specialize]] uint HEAD_DIM,
  [[Specialize]] bool has_sinks,
  [[Specialize]] bool is_kv_cache_ring,
  [[Specialize]] bool is_causal,
  [[Specialize]] bool is_trie,
  [[Specialize]] bool is_sliding_window,
) {
  let native = subgroups_full32(TWO_PASS_SIMDGROUPS);
  let simdgroup = native ? subgroup_id() : tid / TWO_PASS_SIMD;
  let lane = native ? WaveGetLaneIndex() : tid % TWO_PASS_SIMD;
  let elements = HEAD_DIM / TWO_PASS_SIMD;
  let head = head_group * TWO_PASS_SIMDGROUPS + simdgroup;
  // Logical simdgroups past the last head still take every barrier.
  let active = head < num_heads;
  if (native && !active) {
    return;
  }
  let kv_head = (active ? head : 0) / gqa_factor;
  let o_offset = query * num_heads + head;
  let q_offset = head * suffix_length + query;
  let prefix_length = sequence_length - suffix_length;
  let suffix_position = is_kv_cache_ring ? ring_params.ring_length : prefix_length;
  let query_position = is_trie ? suffix_position + trie[query].height : suffix_position + query;
  let first = lane * elements;

  float q[TWO_PASS_MAX_ELEMENTS];
  float o[TWO_PASS_MAX_ELEMENTS];
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

  for (uint key = block; key < sequence_length; key += TWO_PASS_BLOCKS) {
    // The mask depends only on the query and the key, so it is uniform across the workgroup.
    if (attention_use_key(
          key, query, prefix_length, suffix_position, query_position, ring_params, trie, sliding_window_size,
          is_kv_cache_ring, is_causal, is_trie, is_sliding_window
        )) {
      let key_row = kv_head * k_head_stride + key * k_seq_stride + first;
      var score = 0.0;
      for (uint j = 0; j < elements; j++) {
        score += q[j] * keys[key_row + j].toFloat();
      }
      score = two_pass_reduce(score, false, native, simdgroup, lane);

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
    let partial = o_offset * TWO_PASS_BLOCKS + block;
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
func AttentionTwoPass2<T: IModelFloat>(
  Ptr<float, Access.Read> partials,
  Ptr<float, Access.Read> sums,
  Ptr<float, Access.Read> maxs,
  Ptr<T, Access.ReadWrite> out,
  [[Optional("has_gate")]] Ptr<T, Access.Read> gate,
  uint num_heads,
  uint suffix_length,
  [[Optional("has_gate")]] uint gate_row_stride,
  [[Groups("num_heads")]] uint head,
  [[Groups("suffix_length")]] uint query,
  [[Threads("1024")]] uint tid,
  [[Specialize]] uint HEAD_DIM,
  [[Specialize]] bool has_gate,
) {
  let native = subgroups_full32(TWO_PASS_MERGE_THREADS / TWO_PASS_SIMD);
  let simdgroup = native ? subgroup_id() : tid / TWO_PASS_SIMD;
  let lane = native ? WaveGetLaneIndex() : tid % TWO_PASS_SIMD;
  let elements = HEAD_DIM / TWO_PASS_SIMD;
  let o_offset = query * num_heads + head;
  let blocks = o_offset * TWO_PASS_BLOCKS;

  // TWO_PASS_BLOCKS == TWO_PASS_SIMD: lane l takes block l.
  let block_max = maxs[blocks + lane];
  let new_max = two_pass_reduce(block_max, true, native, simdgroup, lane);
  let factor = exp(block_max - new_max);
  let sum_exp_score = two_pass_reduce(sums[blocks + lane] * factor, false, native, simdgroup, lane);

  let first = simdgroup * elements;
  float o[TWO_PASS_MAX_ELEMENTS];
  for (uint i = 0; i < elements; i++) {
    o[i] = two_pass_reduce(partials[(blocks + lane) * HEAD_DIM + first + i] * factor, false, native, simdgroup, lane);
  }

  if (lane == 0) {
    for (uint i = 0; i < elements; i++) {
      var result = T.fromFloat(o[i] / sum_exp_score);
      if (has_gate) {
        let channel = first + i;
        // Metal's attention_apply_gate, as SigmoidGate computes it: the T-rounded result times the FP32 sigmoid of
        // the gate, rounded to T.
        let sigmoid = activate_silu_alpha<true>(1.0, gate[query * gate_row_stride + head * HEAD_DIM + channel].toFloat());
        result = T.fromFloat(exact_product(result.toFloat(), sigmoid));
      }
      out[o_offset * HEAD_DIM + first + i] = result;
    }
  }
}
