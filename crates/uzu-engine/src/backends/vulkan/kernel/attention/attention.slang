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
  float key(uint key, uint j);
  float key_scale(uint key);
  float value(uint key, uint j);
  float value_scale(uint key);
}

// 2π split into an f32 head and the remainder, for Cody-Waite range reduction.
static const float ROPE_TWO_PI_HI = 6.28318548202514648;
static const float ROPE_TWO_PI_LO = -1.7484555314695172e-7;

// Rotates `value` against its RoPE pair `paired` at `position`; `channel` indexes the value within the rotated
// `rope_dim` channels, whose inverse frequencies repeat across both halves.
public float rope_rotate(
  float value,
  float paired,
  Ptr<float, Access.Read> inverse_frequencies,
  float scale,
  uint position,
  uint channel,
  uint rope_dim
) {
  let half = rope_dim / 2;
  let angle = float(position) * inverse_frequencies[channel % half];
  // SPIR-V `sin`/`cos` only promise accuracy within [-π, π], and angles here reach the context length in radians.
  let turns = round(angle / ROPE_TWO_PI_HI);
  let reduced = fma(-turns, ROPE_TWO_PI_LO, fma(-turns, ROPE_TWO_PI_HI, angle));
  let signed_paired = channel < half ? -paired : paired;
  return value * (cos(reduced) * scale) + signed_paired * (sin(reduced) * scale);
}

// Interleaved M-RoPE: pairs 1, 4, 7, ... add the grid row, pairs 2, 5, 8, ... the grid column, while the axis has pairs
// left. The rest stay at the grid's position.
public uint mrope_axis_offset(uint pair, uint grid_row, uint grid_column, uint height_pairs, uint width_pairs) {
  if (pair % 3 == 1 && pair < 3 * height_pairs) {
    return grid_row;
  }
  if (pair % 3 == 2 && pair < 3 * width_pairs) {
    return grid_column;
  }
  return 0;
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
  for (uint a = 0; a < dims; a++) {
    values[a] = hadamard_transform<THREADS>(values[a], SIMD_SIZE, dims * SIMD_SIZE);
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
  let step = scale.toFloat();
  for (uint a = 0; a < dims; a++) {
    // Vulkan division may err by 2.5 ULP and move a tie; code * step is exact, so the remainder corrects the code.
    var code = rint(values[a] / step);
    let remainder = values[a] - code * step;
    if (abs(remainder) > 0.5 * step || (abs(remainder) == 0.5 * step && fmod(code, 2.0) != 0.0)) {
      code += sign(remainder);
    }
    codes[a] = int(clamp(code, -127.0, 127.0));
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
    let key = base + group;
    let used = key < sequence_length;
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
