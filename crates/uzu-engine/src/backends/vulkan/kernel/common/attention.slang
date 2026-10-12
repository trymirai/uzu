import definitions;
import common.activation;

// Single-query attention shared by AttentionSinglePass and AncestorAttention, and the rotary element shared by
// AttentionPrepare and AncestorAttention.

public static const uint ATTENTION_THREADS = 1024;
// As the Metal kernel: 32 groups of 32 lanes; a lane owns HEAD_DIM / 32 consecutive dimensions, at most 16.
static const uint ATTENTION_GROUPS = 32;
static const uint ATTENTION_MAX_HEAD_DIM = 512;
static const uint ATTENTION_DIMS = ATTENTION_MAX_HEAD_DIM / ATTENTION_GROUPS;
static const float ATTENTION_NEGATIVE_INFINITY = asfloat(0xff800000u);
// e^x from here down is below the normal range, which this device class flushes: it is taken as 2^(x log2(e) + 24) and
// rounded onto the FP32 subnormal grid exactly, staged as the CPU rounds its e^x to FP32.
static const float ATTENTION_SUBNORMAL_EXPONENT = -87.0;
static const float ATTENTION_LOG2E = 1.44269504;
// 2^122: while each lane's values sum below it in magnitude, the 32 groups stay below 2^127, so no weighted sum
// overflows, on the CPU either, as weights and rescales are at most 1.
static const float ATTENTION_MAGNITUDE_LIMIT = asfloat(0x7c800000u);

// All as bits, so subnormals and signed zeros pass through shared memory unchanged. The output transpose of the merge,
// also the logical groups' scratch for their sums.
groupshared uint attention_outputs[ATTENTION_THREADS];
// Per group: the maximum and normalizer at the merge; during a replay round the key's score and whether it is used.
groupshared uint attention_max[ATTENTION_GROUPS];
groupshared uint attention_normalizer[ATTENTION_GROUPS];
// Set when the parallel pass meets a class only the CPU's per-key order decides; read after the merge barrier.
groupshared uint attention_replay;

// The query and the keys one workgroup attends: element j of the query as stored, whether key `key` is used, and
// element j of its key and value. Unused keys are never read.
public interface IAttentionKeys {
  float query(uint j);
  bool used(uint key);
  float key(uint key, uint j);
  float value(uint key, uint j);
}

// The CPU apply_rope element: the input times the cosine plus the signed paired element times the sine, each product
// and the sum an FP32 rounding of its own with subnormals kept. The paired element of the first half is negated through
// its sign bit, which the device may flush otherwise.
public float rope_rotated(float value, float paired, bool first, float cosine, float sine) {
  let signed_paired = asfloat(asuint(paired) ^ (first ? 0x80000000u : 0u));
  return exact_sum(exact_product(value, cosine), exact_product(signed_paired, sine), false);
}

// Whether x is a nonzero subnormal, which this device class flushes as an operand.
bool attention_subnormal(float x) {
  return (asuint(x) & 0x7fffffffu) - 1u < 0x007fffffu;
}

// e^x for x <= 0 or NaN in FP32 with subnormals kept, staged as the CPU's rounded e^x: exactly 1 at 0, the builtin over
// the normal range, below it 2^(x log2(e) + 24) rounded onto the subnormal grid (0 where the CPU's underflows). The
// builtins approximate e^x within Vulkan's 3 + 2|x| ULPs, so the result need not equal the CPU's bit for bit.
float attention_exp(float x) {
  if (x == 0.0) {
    return 1.0;
  }
  if (x < ATTENTION_SUBNORMAL_EXPONENT) {
    return exact_scale_down(exp2(x * ATTENTION_LOG2E + 24.0), 24);
  }
  return exp(x);
}

// The sum of x over the 32 lanes of a group, the same in every lane, with subnormals and signed zeros kept: natively a
// butterfly of exact sums, which commute, so both lanes of each pair sum alike; logically an ascending scan of the
// group's row of attention_outputs between two barriers, so every invocation must call it.
float attention_group_sum(float x, bool native, uint group, uint lane) {
  if (native) {
    for (uint offset = ATTENTION_GROUPS / 2; offset > 0; offset /= 2) {
      x = exact_sum(x, asfloat(WaveReadLaneAt(asuint(x), int(lane ^ offset))), false);
    }
    return x;
  }
  let row = group * ATTENTION_GROUPS;
  attention_outputs[row + lane] = asuint(x);
  GroupMemoryBarrierWithGroupSync();
  var sum = asfloat(attention_outputs[row]);
  for (uint other = 1; other < ATTENTION_GROUPS; other++) {
    sum = exact_sum(sum, asfloat(attention_outputs[row + other]), false);
  }
  GroupMemoryBarrierWithGroupSync();
  return sum;
}

// The score of `key` in every lane of its group, 0 for an unused key, which is never read: each lane's products of its
// dimensions, natively unless one of its query or key operands is subnormal, which the device flushes while it may weigh
// in a normal score (then exactly), summed by WaveActiveSum natively, as Metal's simd_sum, or by attention_group_sum.
float attention_score<Keys: IAttentionKeys>(
  Keys source,
  bool used,
  uint key,
  float query[ATTENTION_DIMS],
  bool query_tiny,
  uint first,
  uint dims,
  bool native,
  uint group,
  uint lane
) {
  var partial = 0.0;
  if (used) {
    for (uint a = 0; a < dims; a++) {
      let k = source.key(key, first + a);
      if (query_tiny || attention_subnormal(k)) {
        partial += exact_product(query[a], k);
      } else {
        partial += query[a] * k;
      }
    }
  }
  if (native) {
    return WaveActiveSum(partial);
  }
  return attention_group_sum(partial, false, group, lane);
}

// numerator / denominator as the CPU divides, NaN with neither a used key nor a sink. Where the quotient may be
// subnormal, the numerator is scaled up by 2^24 exactly and the quotient rounded back down.
float attention_quotient(float numerator, float denominator) {
  if (denominator == 0.0) {
    return asfloat(0x7fc00000u);
  }
  if (((asuint(numerator) >> 23) & 0xffu) + 126 <= ((asuint(denominator) >> 23) & 0xffu)) {
    return exact_scale_down(exact_scale_up_24(numerator) / denominator, 24);
  }
  return numerator / denominator;
}

// Attention of one query over the `sequence_length` keys of `source` by one workgroup of ATTENTION_THREADS, written
// to `out[out_base + j]`, in the Metal kernel's ownership: group g of 32 lanes takes keys g + 32 n with its own online
// softmax, each lane holding the scaled query, output and its dimensions in registers; the groups then merge their
// maxima and normalizers and, through a transpose, their outputs. Groups are subgroups where the device runs 32 full
// subgroups of 32 lanes, otherwise logical groups of 32 consecutive invocations whose sums pass through shared memory in
// uniform rounds of 32 keys. As the CPU: the query is scaled exactly in FP32 first; a sink is a key of value zero in
// group 0; a rescale by e^0 is skipped as exact. Weights and rescales are rounded to FP32 before they multiply, and every
// weighted value, sum, merge and the quotient keep subnormals through the exact helpers, which this device class would
// flush. A nonfinite used score or value, a lane's values reaching the magnitude limit, or a nonfinite sink replay the
// whole sequence in the CPU's per-key order, by group 0 in rounds of 32 keys: it alone decides those classes, including
// the NaN of a first used key scoring -inf against no sink. Otherwise the result agrees with the CPU within rounding,
// exp and division approximation and reordering; no used key and no sink give NaN. Unused keys are never read. Every
// invocation reaches every barrier; the last one follows the output.
public void attention_single_query<T: IModelFloat, Keys: IAttentionKeys>(
  Keys source,
  uint head_dim,
  uint sequence_length,
  float scale,
  bool has_sinks,
  float sink,
  Ptr<T, Access.ReadWrite> out,
  uint out_base,
  uint tid
) {
  // Uniform: logical groups take every invocation's barriers in the same rounds.
  let native = subgroups_full32(ATTENTION_GROUPS);
  let group = native ? subgroup_id() : tid / ATTENTION_GROUPS;
  let lane = native ? WaveGetLaneIndex() : tid % ATTENTION_GROUPS;
  let dims = head_dim / ATTENTION_GROUPS;
  let first = lane * dims;
  float query[ATTENTION_DIMS];
  var query_tiny = false;
  for (uint a = 0; a < dims; a++) {
    query[a] = exact_product(scale, source.query(first + a));
    query_tiny = query_tiny || attention_subnormal(query[a]);
  }
  if (tid == 0) {
    attention_replay = 0;
  }
  GroupMemoryBarrierWithGroupSync();

  // Round r holds keys 32 r + g; the last may be partial, its key valid only below the remaining length.
  let rounds = sequence_length / ATTENTION_GROUPS + (sequence_length % ATTENTION_GROUPS != 0 ? 1u : 0u);
  float accumulators[ATTENTION_DIMS];
  var max_score = ATTENTION_NEGATIVE_INFINITY;
  var normalizer = 0.0;
  // A nonfinite sink, known at entry, skips the parallel pass.
  var replay = has_sinks && !isfinite(sink);
  if (!replay) {
    for (uint a = 0; a < dims; a++) {
      accumulators[a] = 0.0;
    }
    max_score = has_sinks && group == 0 ? sink : ATTENTION_NEGATIVE_INFINITY;
    normalizer = has_sinks && group == 0 ? 1.0 : 0.0;
    var exceptional = false;
    var magnitude = 0.0;
    for (uint round_index = 0; round_index < rounds; round_index++) {
      let base = round_index * ATTENTION_GROUPS;
      var key = base;
      var used = false;
      if (group < sequence_length - base) {
        key = base + group;
        used = source.used(key);
      }
      let score = attention_score(source, used, key, query, query_tiny, first, dims, native, group, lane);
      if (used) {
        // The CPU's staging, subnormals kept: the rescale and the weight rounded to FP32, then each product and sum.
        exceptional = exceptional || !isfinite(score);
        let next = max(max_score, score);
        if (!(next == max_score && isfinite(max_score))) {
          let factor = attention_exp(max_score - next);
          for (uint a = 0; a < dims; a++) {
            accumulators[a] = exact_product(accumulators[a], factor);
          }
          normalizer = exact_product(normalizer, factor);
        }
        max_score = next;
        let weight = attention_exp(score - next);
        for (uint a = 0; a < dims; a++) {
          let value = source.value(key, first + a);
          magnitude += abs(value);
          accumulators[a] = exact_sum(accumulators[a], exact_product(weight, value), false);
        }
        normalizer = exact_sum(normalizer, weight, false);
      }
    }
    if (lane == 0) {
      attention_max[group] = asuint(max_score);
      attention_normalizer[group] = asuint(normalizer);
    }
    if (exceptional || !(magnitude < ATTENTION_MAGNITUDE_LIMIT)) {
      InterlockedOr(attention_replay, 1u);
    }
    GroupMemoryBarrierWithGroupSync();
    replay = attention_replay != 0;
  }

  if (replay) {
    for (uint a = 0; a < dims; a++) {
      accumulators[a] = 0.0;
    }
    max_score = has_sinks && group == 0 ? sink : ATTENTION_NEGATIVE_INFINITY;
    normalizer = has_sinks && group == 0 ? 1.0 : 0.0;
    for (uint round_index = 0; round_index < rounds; round_index++) {
      let base = round_index * ATTENTION_GROUPS;
      var key = base;
      var used = false;
      if (group < sequence_length - base) {
        key = base + group;
        used = source.used(key);
      }
      let score = attention_score(source, used, key, query, query_tiny, first, dims, native, group, lane);
      // The round's scores, then group 0 takes the CPU's steps over its used keys in key order: the running maximum
      // ignoring NaN as f32::max does, then the rescale and the weight rounded to FP32 and every product and sum
      // exact, in its order.
      if (lane == 0) {
        attention_max[group] = asuint(score);
        attention_normalizer[group] = used ? 1u : 0u;
      }
      GroupMemoryBarrierWithGroupSync();
      if (group == 0) {
        for (uint entry = 0; entry < ATTENTION_GROUPS; entry++) {
          if (attention_normalizer[entry] != 0) {
            let key_score = asfloat(attention_max[entry]);
            let next = isnan(key_score) ? max_score : (isnan(max_score) ? key_score : max(max_score, key_score));
            let factor = attention_exp(max_score - next);
            let weight = attention_exp(key_score - next);
            for (uint a = 0; a < dims; a++) {
              let product = exact_product(weight, source.value(base + entry, first + a));
              accumulators[a] = ordered_sum(exact_product(accumulators[a], factor), product);
            }
            normalizer = ordered_sum(exact_product(normalizer, factor), weight);
            max_score = next;
          }
        }
      }
      GroupMemoryBarrierWithGroupSync();
    }
    if (group == 0) {
      for (uint a = 0; a < dims; a++) {
        out[out_base + first + a] = T.fromFloat(attention_quotient(accumulators[a], normalizer));
      }
    }
  } else {
    // Lane l takes group l's maximum and factor, as Metal; an empty group weighs nothing, also when every group is
    // empty and no sink gives a maximum.
    let entry = asfloat(attention_max[lane]);
    var largest = entry;
    if (native) {
      largest = WaveActiveMax(entry);
    } else {
      for (uint other = 0; other < ATTENTION_GROUPS; other++) {
        largest = max(largest, asfloat(attention_max[other]));
      }
    }
    let factor = entry == ATTENTION_NEGATIVE_INFINITY ? 0.0 : attention_exp(entry - largest);
    let normalized = exact_product(asfloat(attention_normalizer[lane]), factor);
    let denominator = attention_group_sum(normalized, native, group, lane);
    // Transposed, lane l of group g holds group l's output at dimension g dims + a.
    for (uint a = 0; a < dims; a++) {
      attention_outputs[lane * ATTENTION_GROUPS + group] = asuint(accumulators[a]);
      GroupMemoryBarrierWithGroupSync();
      let rescaled = exact_product(asfloat(attention_outputs[group * ATTENTION_GROUPS + lane]), factor);
      let numerator = attention_group_sum(rescaled, native, group, lane);
      GroupMemoryBarrierWithGroupSync();
      if (lane == 0) {
        out[out_base + group * dims + a] = T.fromFloat(attention_quotient(numerator, denominator));
      }
    }
  }
  GroupMemoryBarrierWithGroupSync();
}
