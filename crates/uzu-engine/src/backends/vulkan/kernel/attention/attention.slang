import definitions;
import common.hadamard;
import common.workgroup_reduction;
import generated.kv_layout;

public static const uint ATTENTION_THREADS = 1024;
static const uint ATTENTION_MAX_HEAD_DIM = 512;
public static const uint ATTENTION_DIMS = ATTENTION_MAX_HEAD_DIM / SIMD_SIZE;

groupshared float attention_outputs[ATTENTION_THREADS];
groupshared float attention_max[SIMD_SIZE];
groupshared float attention_normalizer[SIMD_SIZE];
groupshared float attention_numerators[ATTENTION_MAX_HEAD_DIM];

public interface IAttentionKeys {
  float query(uint j);
  float query_scale();
  bool used(uint key);
  float key(uint key, uint j);
  float key_scale(uint key);
  float value(uint key, uint j);
  float value_scale(uint key);
}

public float rope_rotated(float value, float paired, bool first, float cosine, float sine) {
  return value * cosine + (first ? -paired : paired) * sine;
}

float rotation_sign(uint channel) {
  return ((ROTATION_SIGN_BITS[(channel / WORD_BITS) % 8] >> (channel % WORD_BITS)) & 1u) != 0 ? -1.0 : 1.0;
}

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

  let rounds = sequence_length / SIMD_SIZE + (sequence_length % SIMD_SIZE != 0 ? 1u : 0u);
  for (uint round_index = 0; round_index < rounds; round_index++) {
    let base = round_index * SIMD_SIZE;
    var key = base;
    var used = false;
    if (group < sequence_length - base) {
      key = base + group;
      used = source.used(key);
    }
    var partial = 0.0;
    if (used) {
      for (uint a = 0; a < dims; a++) {
        partial += query[a] * source.key(key, first + a);
      }
    }
    let dot = segment_reduce<Sum<float>, ATTENTION_THREADS>(partial, SIMD_SIZE);
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

  let entry = attention_max[lane];
  let largest = segment_reduce<Max<float>, ATTENTION_THREADS>(entry, SIMD_SIZE);
  let factor = entry == NEGATIVE_INFINITY ? 0.0 : exp(entry - largest);
  let denominator = segment_reduce<Sum<float>, ATTENTION_THREADS>(attention_normalizer[lane] * factor, SIMD_SIZE);
  for (uint a = 0; a < dims; a++) {
    attention_outputs[lane * SIMD_SIZE + group] = accumulators[a];
    GroupMemoryBarrierWithGroupSync();
    let numerator = segment_reduce<Sum<float>, ATTENTION_THREADS>(
      attention_outputs[group * SIMD_SIZE + lane] * factor, SIMD_SIZE
    );
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0) {
      attention_numerators[group * dims + a] = numerator;
    }
  }
  GroupMemoryBarrierWithGroupSync();
  return float2(largest, denominator);
}

public float attention_numerator(uint dimension) {
  return attention_numerators[dimension];
}
