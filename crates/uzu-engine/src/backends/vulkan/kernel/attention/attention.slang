import definitions;
import common.workgroup_reduction;

public static const uint ATTENTION_THREADS = 1024;
// SIMD_SIZE groups of SIMD_SIZE lanes; a lane owns HEAD_DIM / 32 consecutive dimensions, at most 16.
static const uint ATTENTION_MAX_HEAD_DIM = 512;
public static const uint ATTENTION_DIMS = ATTENTION_MAX_HEAD_DIM / SIMD_SIZE;

// The output transpose of the merge.
groupshared float attention_outputs[ATTENTION_THREADS];
// Per group: the maximum and normalizer at the merge.
groupshared float attention_max[SIMD_SIZE];
groupshared float attention_normalizer[SIMD_SIZE];

// The query and the keys one workgroup attends: element j of the query as stored, whether key `key` is used, and
// element j of its key and value. Unused keys are never read.
public interface IAttentionKeys {
  float query(uint j);
  bool used(uint key);
  float key(uint key, uint j);
  float value(uint key, uint j);
}

// A rotary element: the input times the cosine plus the signed paired element times the sine.
public float rope_rotated(float value, float paired, bool first, float cosine, float sine) {
  return value * cosine + (first ? -paired : paired) * sine;
}

// The sum of x over the lanes of a group, the same in every lane.
float attention_sum(float x) {
  return segment_reduce<Sum<float>, ATTENTION_THREADS>(x, SIMD_SIZE);
}

// The score of `key` in every lane of its group, 0 for an unused key, which is never read: each lane's products of its
// dimensions, summed over the group.
float attention_score<Keys: IAttentionKeys>(
  Keys source,
  bool used,
  uint key,
  float query[ATTENTION_DIMS],
  uint first,
  uint dims
) {
  var partial = 0.0;
  if (used) {
    for (uint a = 0; a < dims; a++) {
      partial += query[a] * source.key(key, first + a);
    }
  }
  return attention_sum(partial);
}

// Attention of one query over `sequence_length` keys by one workgroup, written to `out[out_base + j]`: group g takes keys
// g + 32 n with its own online softmax, then the groups merge through a transpose. A sink is a zero-valued key of group
// 0. Groups reduce in uniform rounds of 32 keys, so unused keys are never read and every invocation reaches every barrier.
public void attention_single_query<T: IModelFloat, Keys: IAttentionKeys>(
  Keys source,
  uint head_dim,
  uint sequence_length,
  float scale,
  bool has_sinks,
  float sink,
  Ptr<T, Access.ReadWrite> out,
  uint out_base
) {
  let invocation = subgroup_invocation();
  let group = invocation / SIMD_SIZE;
  let lane = invocation % SIMD_SIZE;
  let dims = head_dim / SIMD_SIZE;
  let first = lane * dims;
  float query[ATTENTION_DIMS];
  float accumulators[ATTENTION_DIMS];
  for (uint a = 0; a < dims; a++) {
    query[a] = scale * source.query(first + a);
    accumulators[a] = 0.0;
  }
  var max_score = has_sinks && group == 0 ? sink : NEGATIVE_INFINITY;
  var normalizer = has_sinks && group == 0 ? 1.0 : 0.0;

  // Round r holds keys 32 r + g; the last may be partial, its key valid only below the remaining length.
  let rounds = sequence_length / SIMD_SIZE + (sequence_length % SIMD_SIZE != 0 ? 1u : 0u);
  for (uint round_index = 0; round_index < rounds; round_index++) {
    let base = round_index * SIMD_SIZE;
    var key = base;
    var used = false;
    if (group < sequence_length - base) {
      key = base + group;
      used = source.used(key);
    }
    let score = attention_score(source, used, key, query, first, dims);
    if (used) {
      let next = max(max_score, score);
      let factor = exp(max_score - next);
      let weight = exp(score - next);
      max_score = next;
      normalizer = normalizer * factor + weight;
      for (uint a = 0; a < dims; a++) {
        accumulators[a] = accumulators[a] * factor + weight * source.value(key, first + a);
      }
    }
  }
  if (lane == 0) {
    attention_max[group] = max_score;
    attention_normalizer[group] = normalizer;
  }
  GroupMemoryBarrierWithGroupSync();

  // Lane l takes group l's maximum and factor.
  let entry = attention_max[lane];
  let largest = segment_reduce<Max<float>, ATTENTION_THREADS>(entry, SIMD_SIZE);
  let factor = exp(entry - largest);
  let denominator = attention_sum(attention_normalizer[lane] * factor);
  // Transposed, lane l of group g holds group l's output at dimension g dims + a.
  for (uint a = 0; a < dims; a++) {
    attention_outputs[lane * SIMD_SIZE + group] = accumulators[a];
    GroupMemoryBarrierWithGroupSync();
    let numerator = attention_sum(attention_outputs[group * SIMD_SIZE + lane] * factor);
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0) {
      out[out_base + group * dims + a] = T.fromFloat(numerator / denominator);
    }
  }
  GroupMemoryBarrierWithGroupSync();
}
