import definitions;
import common.ranking;
import common.sampling;
import common.workgroup_reduction;

static const uint UNIFIED_SAMPLING_THREADS = 1024;
static const uint UNIFIED_SAMPLING_ITERATIONS = 64;

// The numeric order of FP32 values as integer keys, subnormals kept: signed zeros equal, NaN 0, below -infinity's key.
uint unified_sampling_key(float value) {
  let bits = asuint(value);
  let magnitude = bits & 0x7fffffffu;
  return magnitude > 0x7f800000u ? 0u : float_order_key(magnitude == 0 ? 0u : bits);
}

// A token's logit masked to -infinity, then scaled by the inverse temperature.
float unified_sampling_value<T : IModelFloat>(
  Ptr<T, Access.Read> logits,
  uint logits_offset,
  Ptr<uint, Access.Read> mask,
  uint mask_offset,
  uint token,
  float inverse_temperature,
  bool has_bitmask,
  bool has_temperature,
) {
  var value = logits[logits_offset + token].toFloat();
  if (has_bitmask) {
    let word = mask[mask_offset + token / 32];
    if ((word & (1u << (token & 31u))) == 0) {
      value = NEGATIVE_INFINITY;
    }
  }
  return has_temperature ? value * inverse_temperature : value;
}

// The (key, index) candidate with the largest key and then the lowest index.
struct UnifiedSamplingArgmax : IReduction {
  typealias Value = uint2;
  static uint2 combine(uint2 a, uint2 b) {
    return b.x > a.x || (b.x == a.x && b.y < a.y) ? b : a;
  }
  static uint2 subgroup(uint2 value) {
    let key = WaveActiveMax(value.x);
    return uint2(key, WaveActiveMin(value.x == key ? value.y : 0xffffffffu));
  }
}

// One sweep over the row by this invocation: tokens outranking `rank` (by value, then lower index; (0, 0) admits every
// number) compete by Gumbel-perturbed score for `best`. The first sweep gathers the running (maximum, normalizer), later
// ones count competitors and sum exp(value - maximum). Tokens column + 1024 w share one Philox counter (sampling_index).
void unified_sampling_sweep<T : IModelFloat>(
  Ptr<T, Access.Read> logits,
  uint logits_offset,
  Ptr<uint, Access.Read> mask,
  uint mask_offset,
  uint vocab_size,
  uint tid,
  uint2 seed,
  uint2 rank,
  float maximum,
  float inverse_temperature,
  bool first,
  bool is_stochastic,
  bool has_bitmask,
  bool has_temperature,
  bool has_top_p,
  out uint2 best,
  out float2 statistics,
  out uint count,
) {
  best = uint2(0, 0xffffffffu);
  statistics = float2(NEGATIVE_INFINITY, 0.0);
  count = 0;
  let negative_infinity_key = unified_sampling_key(NEGATIVE_INFINITY);
  for (uint tile = 0; tile < vocab_size; tile += 4096) {
    let column = tile + tid;
    if (column >= vocab_size) {
      break;
    }
    float values[4];
    var competing = 0u;
    for (uint word = 0; word < 4; word++) {
      let token = column + word * 1024;
      values[word] = 0.0;
      if (token >= vocab_size) {
        continue;
      }
      let value = unified_sampling_value(logits, logits_offset, mask, mask_offset, token, inverse_temperature,
                                         has_bitmask, has_temperature);
      values[word] = value;
      let key = unified_sampling_key(value);
      if (first) {
        if (key != negative_infinity_key) {
          let previous = statistics.x;
          if (key > unified_sampling_key(previous)) {
            statistics.x = value;
          }
          if (has_top_p) {
            statistics.y = statistics.y * exp(previous - statistics.x) + exp(value - statistics.x);
          }
        }
      }
      if (!(key > rank.x || (key == rank.x && token < rank.y))) {
        continue;
      }
      competing |= 1u << word;
      if (!first) {
        count += 1;
        if (has_top_p) {
          statistics.y += exp(value - maximum);
        }
      }
    }
    if (competing == 0) {
      continue;
    }
    var words = uint4(0);
    if (is_stochastic) {
      words = sampling_words(seed, sampling_index(column, vocab_size).x);
    }
    for (uint word = 0; word < 4; word++) {
      if (((competing >> word) & 1u) == 0) {
        continue;
      }
      let token = column + word * 1024;
      let score = is_stochastic ? values[word] + sampling_noise(sampling_uniform(words[word])) : values[word];
      let key = unified_sampling_key(score);
      if (key != 0 && (key > best.x || (key == best.x && token < best.y))) {
        best = uint2(key, token);
      }
    }
  }
}

// Candidate refinement under independent top-k, top-p and min-p filters: while a filter rejects the Gumbel argmax, the
// next is the argmax among the tokens outranking it, up to 64 candidates, the last returned unfiltered. Token 0 where no
// candidate is defined or a filter retains nothing.
[[Kernel]] [[Public]] [[Precondition("encode", "vocab_size > 0")]]
func UnifiedSampling<T : IModelFloat>(
  Ptr<T, Access.Read> logits,
  Ptr<uint, Access.ReadWrite> output,
  [[Optional("is_stochastic")]] Ptr<uint, Access.Read> seeds,
  [[Optional("has_bitmask")]] Ptr<uint, Access.Read> bitmask,
  [[Optional("has_temperature")]] float temperature,
  [[Optional("has_top_k")]] uint top_k,
  [[Optional("has_top_p")]] float top_p,
  [[Optional("has_min_p")]] float min_p,
  uint vocab_size,
  uint batch_size,
  [[Specialize]] bool is_stochastic,
  [[Specialize]] bool has_bitmask,
  [[Specialize]] bool has_temperature,
  [[Specialize]] bool has_top_k,
  [[Specialize]] bool has_top_p,
  [[Specialize]] bool has_min_p,
  [[Groups("batch_size")]] uint group,
  [[Threads("1024")]] uint tid,
) {
  static_assert(UNIFIED_SAMPLING_THREADS == 1024, "1024 invocations sweep the 1024 columns of 4096-token tiles");
  let inverse_temperature = has_temperature ? 1.0 / temperature : 1.0;
  let log_min_p = has_min_p ? log(min_p) : 0.0;
  let logits_offset = group * vocab_size;
  let mask_offset = (vocab_size + 31) / 32 * group;
  var seed = uint2(0, 0);
  if (is_stochastic) {
    seed = uint2(seeds[group * 2], seeds[group * 2 + 1]);
  }
  uint2 best;
  float2 statistics;
  uint count;
  unified_sampling_sweep(logits, logits_offset, bitmask, mask_offset, vocab_size, tid, seed, uint2(0, 0), 0.0,
                         inverse_temperature, true, is_stochastic, has_bitmask, has_temperature, has_top_p, best,
                         statistics, count);
  var candidate = workgroup_reduce<UnifiedSamplingArgmax, UNIFIED_SAMPLING_THREADS>(best);
  var maximum = NEGATIVE_INFINITY;
  if (has_top_p || has_min_p) {
    maximum = workgroup_reduce<Max<float>, UNIFIED_SAMPLING_THREADS>(statistics.x);
  }
  var norm = 0.0;
  if (has_top_p) {
    if (statistics.y != 0.0) {
      statistics.y *= exp(statistics.x - maximum);
    }
    norm = workgroup_reduce<Sum<float>, UNIFIED_SAMPLING_THREADS>(statistics.y);
  }
  let threshold_key = unified_sampling_key(maximum + log_min_p);
  let empty = (has_top_k && top_k == 0) || (has_top_p && top_p <= 0.0) ||
              (has_min_p && threshold_key > unified_sampling_key(maximum));
  var chosen = 0u;
  if (!(has_top_k || has_top_p || has_min_p)) {
    chosen = candidate.x != 0 ? candidate.y : 0u;
  } else if (!empty) {
    for (uint iteration = 0; candidate.x != 0; iteration++) {
      if (iteration == UNIFIED_SAMPLING_ITERATIONS - 1) {
        chosen = candidate.y;
        break;
      }
      let value = unified_sampling_value(logits, logits_offset, bitmask, mask_offset, candidate.y, inverse_temperature,
                                         false, has_temperature);
      let rank = uint2(unified_sampling_key(value), candidate.y);
      unified_sampling_sweep(logits, logits_offset, bitmask, mask_offset, vocab_size, tid, seed, rank, maximum,
                             inverse_temperature, false, is_stochastic, has_bitmask, has_temperature, has_top_p, best,
                             statistics, count);
      let next = workgroup_reduce<UnifiedSamplingArgmax, UNIFIED_SAMPLING_THREADS>(best);
      var passed = !(has_min_p && rank.x < threshold_key);
      if (has_top_k) {
        let above = workgroup_reduce<Sum<uint>, UNIFIED_SAMPLING_THREADS>(count);
        passed = passed && above < top_k;
      }
      if (has_top_p) {
        let mass = workgroup_reduce<Sum<float>, UNIFIED_SAMPLING_THREADS>(statistics.y) / norm;
        passed = passed && !(mass >= top_p);
      }
      if (passed) {
        chosen = candidate.y;
        break;
      }
      candidate = next;
    }
  }
  if (tid == 0) {
    output[group] = chosen;
  }
}
