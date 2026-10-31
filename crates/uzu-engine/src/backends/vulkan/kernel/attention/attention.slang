import definitions;
import common.hadamard;
import common.workgroup_reduction;
import generated.kv_layout;

public static const uint ATTENTION_THREADS = 1024;
// SIMD_SIZE groups of SIMD_SIZE lanes; a lane owns HEAD_DIM / 32 consecutive dimensions, at most 16.
static const uint ATTENTION_MAX_HEAD_DIM = 512;
public static const uint ATTENTION_DIMS = ATTENTION_MAX_HEAD_DIM / SIMD_SIZE;

// The output transpose of the merge.
groupshared float attention_outputs[ATTENTION_THREADS];
// Per group: the maximum and normalizer at the merge.
groupshared float attention_max[SIMD_SIZE];
groupshared float attention_normalizer[SIMD_SIZE];
// The merged numerators of attention_single_query's row, by dimension.
groupshared float attention_numerators[ATTENTION_MAX_HEAD_DIM];

// The INT8 query and keys one workgroup attends: code j of the query and its scale, whether key `key` is used, code j of
// its key and of its value, centered, and their scales. Unused keys are never read.
public interface IAttentionKeys {
  float query(uint j);
  float query_scale();
  bool used(uint key);
  float key(uint key, uint j);
  float key_scale(uint key);
  float value(uint key, uint j);
  float value_scale(uint key);
}

// A rotary element: the input times the cosine plus the signed paired element times the sine.
public float rope_rotated(float value, float paired, bool first, float cosine, float sine) {
  return value * cosine + (first ? -paired : paired) * sine;
}

// The sign the K/V rotation gives dimension `channel` before its Hadamard transform.
float rotation_sign(uint channel) {
  return ((ROTATION_SIGN_BITS[(channel / WORD_BITS) % 8] >> (channel % WORD_BITS)) & 1u) != 0 ? -1.0 : 1.0;
}

// The normalized Hadamard transform of a row held by a group of SIMD_SIZE invocations in subgroup_invocation order, lane
// l holding dimensions [l dims, (l + 1) dims): butterflies in registers, then across the group's lanes.
void attention_hadamard<let THREADS : uint>(inout float values[ATTENTION_DIMS], uint dims) {
  for (uint stride = 1; stride < dims; stride <<= 1) {
    for (uint a = 0; a < dims; a++) {
      if ((a & stride) == 0) {
        let low = values[a];
        let high = values[a | stride];
        values[a] = low + high;
        values[a | stride] = low - high;
      }
    }
  }
  let normalization = 1.0 / sqrt(float(dims));
  for (uint a = 0; a < dims; a++) {
    values[a] = hadamard_transform<THREADS>(values[a]) * normalization;
  }
}

// The rotation K, V and query codes are quantized in: the rotation signs, then the Hadamard transform, of a row laid out
// as attention_hadamard's. attention_unrotate inverts it. Every invocation of the workgroup calls both.
public void attention_rotate<let THREADS : uint>(inout float values[ATTENTION_DIMS], uint dims, uint lane) {
  for (uint a = 0; a < dims; a++) {
    values[a] *= rotation_sign(lane * dims + a);
  }
  attention_hadamard<THREADS>(values, dims);
}

public void attention_unrotate<let THREADS : uint>(inout float values[ATTENTION_DIMS], uint dims, uint lane) {
  attention_hadamard<THREADS>(values, dims);
  for (uint a = 0; a < dims; a++) {
    values[a] *= rotation_sign(lane * dims + a);
  }
}

// The BF16 scale of a rotated row laid out as attention_hadamard's, its largest magnitude over 127 or 1 for a zero row,
// and its INT8 codes. Every invocation of the workgroup calls it.
public bf16 attention_quantize<let THREADS : uint>(float values[ATTENTION_DIMS], uint dims, out int codes[ATTENTION_DIMS]) {
  var largest = 0.0;
  for (uint a = 0; a < dims; a++) {
    largest = max(largest, abs(values[a]));
  }
  largest = segment_reduce<Max<float>, THREADS>(largest, SIMD_SIZE);
  let scale = bf16.fromFloat(largest > 0.0 ? largest / 127.0 : 1.0);
  for (uint a = 0; a < dims; a++) {
    codes[a] = int(clamp(round(values[a] / scale.toFloat()), -127.0, 127.0));
  }
  return scale;
}

// The `count` codes of `codes` from `element`, which their size aligns: 2, 4, 8 or 16 bytes, low bytes first.
public uint4 attention_code_words<T>(Ptr<T, Access.Read> codes, uint element, uint count) {
  if (count == 16) {
    return load_aligned<uint4, 16>(codes, element);
  }
  if (count == 8) {
    return uint4(load_aligned<uint2, 8>(codes, element), 0, 0);
  }
  if (count == 4) {
    return uint4(load_aligned<uint, 4>(codes, element), 0, 0, 0);
  }
  return uint4(uint(load_aligned<uint16_t, 2>(codes, element)), 0, 0, 0);
}

// Code `index` of attention_code_words' words: a signed key or query code.
public float attention_key_code(uint4 words, uint index) {
  return float(asint(words[index / 4] << (24 - 8 * (index % 4))) >> 24);
}

// Code `index` of attention_code_words' words: a value code, centered.
public float attention_value_code(uint4 words, uint index) {
  return float(int((words[index / 4] >> (8 * (index % 4))) & 0xffu) - int(V_CODE_BIAS));
}

// The sum of x over the lanes of a group, the same in every lane.
float attention_sum(float x) {
  return segment_reduce<Sum<float>, ATTENTION_THREADS>(x, SIMD_SIZE);
}

// The code dot of the query with `key` in every lane of its group, 0 for an unused key, which is never read: each
// lane's products of its dimensions, summed over the group.
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

// Attention of one query over `sequence_length` keys by one workgroup: group g takes keys g + 32 n with its own online
// softmax, then the groups merge through a transpose into the numerators attention_numerator reads. The result is the
// largest score and the denominator, 0 without a used key. Groups reduce in uniform rounds of 32 keys, so unused keys are
// never read and every invocation reaches every barrier.
public float2 attention_single_query<Keys: IAttentionKeys>(
  Keys source,
  uint head_dim,
  uint sequence_length,
  float scale
) {
  let invocation = subgroup_invocation();
  let group = invocation / SIMD_SIZE;
  let lane = invocation % SIMD_SIZE;
  let dims = head_dim / SIMD_SIZE;
  let first = lane * dims;
  float query[ATTENTION_DIMS];
  float accumulators[ATTENTION_DIMS];
  for (uint a = 0; a < dims; a++) {
    query[a] = source.query(first + a);
    accumulators[a] = 0.0;
  }
  let query_scale = scale * source.query_scale();
  var max_score = NEGATIVE_INFINITY;
  var normalizer = 0.0;

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
    let dot = attention_score(source, used, key, query, first, dims);
    if (used) {
      let score = dot * query_scale * source.key_scale(key);
      let next = max(max_score, score);
      let factor = exp(max_score - next);
      let weight = exp(score - next);
      max_score = next;
      normalizer = normalizer * factor + weight;
      let value_weight = weight * source.value_scale(key);
      for (uint a = 0; a < dims; a++) {
        accumulators[a] = accumulators[a] * factor + value_weight * source.value(key, first + a);
      }
    }
  }
  if (lane == 0) {
    attention_max[group] = max_score;
    attention_normalizer[group] = normalizer;
  }
  GroupMemoryBarrierWithGroupSync();

  // Lane l takes group l's maximum and factor; a group without a used key adds nothing.
  let entry = attention_max[lane];
  let largest = segment_reduce<Max<float>, ATTENTION_THREADS>(entry, SIMD_SIZE);
  let factor = entry == NEGATIVE_INFINITY ? 0.0 : exp(entry - largest);
  let denominator = attention_sum(attention_normalizer[lane] * factor);
  // Transposed, lane l of group g holds group l's numerator at dimension g dims + a.
  for (uint a = 0; a < dims; a++) {
    attention_outputs[lane * SIMD_SIZE + group] = accumulators[a];
    GroupMemoryBarrierWithGroupSync();
    let numerator = attention_sum(attention_outputs[group * SIMD_SIZE + lane] * factor);
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0) {
      attention_numerators[group * dims + a] = numerator;
    }
  }
  GroupMemoryBarrierWithGroupSync();
  return float2(largest, denominator);
}

// Dimension `dimension`'s numerator of the row attention_single_query last merged.
public float attention_numerator(uint dimension) {
  return attention_numerators[dimension];
}
