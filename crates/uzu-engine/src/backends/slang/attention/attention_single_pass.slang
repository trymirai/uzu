import definitions;
import storage;
import generated.ring;
import generated.trie;

groupshared float reduction[512];

[[Variants("T", "float, half, BFloat16")]]
[[Variants("HEAD_DIM", "64, 128, 256, 512")]]
[[Public]] [[Kernel]] func AttentionSinglePass<T : ITensorElement, let HEAD_DIM : uint>(
    StructuredBuffer<uint> queries,
    StructuredBuffer<uint> keys,
    StructuredBuffer<uint> values,
    RWStructuredBuffer<Atomic<uint>> out,
    uint gqa_factor,
    uint sequence_length,
    uint k_head_stride,
    uint k_seq_stride,
    uint v_head_stride,
    uint v_seq_stride,
    [[Optional("is_kv_cache_ring")]] RingParams ring_params,
    float scale,
    [[Optional("is_trie")]] StructuredBuffer<TrieNode> trie,
    [[Optional("is_sliding_window")]] uint sliding_window_size,
    [[Optional("has_sinks")]] StructuredBuffer<uint> sinks,
    uint num_heads,
    uint suffix_length,
    [[Specialize]] bool has_sinks,
    [[Specialize]] bool is_kv_cache_ring,
    [[Specialize]] bool is_causal,
    [[Specialize]] bool is_trie,
    [[Specialize]] bool is_sliding_window,
    ThreadContext thread_context,
    [[Groups("num_heads")]] uint head_idx,
    [[Groups("suffix_length")]] uint q_seq_idx,
    [[Threads("HEAD_DIM")]] uint tid,
) {
  uint3 local = uint3(tid, 0, 0);
  uint3 group = uint3(head_idx, q_seq_idx, 0);

  let lane = local.x;
  let head = group.x;
  let query = group.y;
  let kvHead = head / gqa_factor;
  let prefix = sequence_length - suffix_length;
  var suffixPosition = prefix;
  if (is_kv_cache_ring) {
    suffixPosition = ring_params.ring_length;
  }
  let queryPosition = suffixPosition + query;
  var q = 0.0;
  if (lane < HEAD_DIM) {
    q = tensorLoad<T>(queries, (head * suffix_length + query) * HEAD_DIM + lane) * scale;
  }
  var accumulator = 0.0;
  var maxScore = -3.402823466e+38;
  var sum = 0.0;
  if (has_sinks) {
    maxScore = tensorLoad<T>(sinks, head);
    sum = 1.0;
  }
  for (var key = 0u; key < sequence_length; key++) {
    var useKey = true;
    var keyPosition = key;
    if (key >= prefix) {
      let suffixKey = key - prefix;
      keyPosition = suffixPosition + suffixKey;
      if (is_causal) {
        useKey = suffixKey <= query;
      }
    } else if (is_kv_cache_ring) {
      keyPosition = (prefix + key - ring_params.ring_offset) % max(prefix, 1u);
      useKey = keyPosition < ring_params.ring_length;
    }
    if (is_sliding_window) {
      if (is_causal) {
        useKey = useKey && keyPosition <= queryPosition && queryPosition - keyPosition < sliding_window_size;
      } else {
        let distance = max(keyPosition, queryPosition) - min(keyPosition, queryPosition);
        useKey = useKey && distance <= sliding_window_size / 2u;
      }
    }
    if (!useKey) {
      continue;
    }
    var product = 0.0;
    if (lane < HEAD_DIM) {
      product = q * tensorLoad<T>(keys, kvHead * k_head_stride + key * k_seq_stride + lane);
    }
    reduction[lane] = product;
    GroupMemoryBarrierWithGroupSync();
    for (var stride = HEAD_DIM / 2u; stride > 0u; stride /= 2u) {
      if (lane < stride) {
        reduction[lane] += reduction[lane + stride];
      }
      GroupMemoryBarrierWithGroupSync();
    }
    let score = reduction[0];
    let nextMax = max(maxScore, score);
    let factor = exp(maxScore - nextMax);
    let probability = exp(score - nextMax);
    sum = sum * factor + probability;
    maxScore = nextMax;
    if (lane < HEAD_DIM) {
      accumulator = accumulator * factor +
                    probability * tensorLoad<T>(values, kvHead * v_head_stride + key * v_seq_stride + lane);
    }
    // All lanes must consume the reduced score before the next key reuses it.
    GroupMemoryBarrierWithGroupSync();
  }
  if (lane < HEAD_DIM) {
    tensorStore<T>(out, (query * num_heads + head) * HEAD_DIM + lane, accumulator / sum);
  }
}
