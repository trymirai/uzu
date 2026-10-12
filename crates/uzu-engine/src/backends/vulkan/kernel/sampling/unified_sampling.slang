import definitions;
import common.activation;
import common.ranking;
import common.sampling;
import common.storage_address;
import common.workgroup_reduction;

static const uint UNIFIED_SAMPLING_THREADS = 256;
static const uint UNIFIED_SAMPLING_ITERATIONS = 64;
static const float UNIFIED_SAMPLING_NEGATIVE_INFINITY = asfloat(0xff800000u);
// One entry per subgroup; 256 covers the smallest possible subgroup width of the workgroup.
groupshared uint2 unified_sampling_scratch[256];

// The numeric order of FP32 values as integer keys, subnormals kept: signed zeros equal, NaN 0, below -infinity's key.
uint unified_sampling_key(float value) {
  let bits = asuint(value);
  let magnitude = bits & 0x7fffffffu;
  return magnitude > 0x7f800000u ? 0u : float_order_key(magnitude == 0 ? 0u : bits);
}

// A token's logit masked to -infinity, then scaled by the inverse temperature, in the Metal kernel's order.
float unified_sampling_value<T : IModelFloat>(
  Ptr<T, Access.Read> logits,
  Ptr<uint, Access.Read> mask,
  uint token,
  float inverse_temperature,
  bool has_bitmask,
  bool has_temperature,
) {
  var value = element_pointer(logits, uint2(token, 0))[0].toFloat();
  if (has_bitmask) {
    let word = element_pointer(mask, uint2(token / 32, 0))[0];
    if ((word & (1u << (token & 31u))) == 0) {
      value = UNIFIED_SAMPLING_NEGATIVE_INFINITY;
    }
  }
  return has_temperature ? exact_product(value, inverse_temperature) : value;
}

// Subgroup sum of x (count), or the (key, index) candidate with the largest key and then the lowest index.
uint2 unified_sampling_wave(uint2 value, bool count) {
  if (count) {
    return uint2(WaveActiveSum(value.x), 0);
  }
  let key = WaveActiveMax(value.x);
  return uint2(key, WaveActiveMin(value.x == key ? value.y : 0xffffffffu));
}

// unified_sampling_wave over the workgroup through the shared scratch. Every invocation must call it, and all read the
// same result before the scratch is reused.
uint2 unified_sampling_reduce(uint2 value, bool count) {
  let partial = unified_sampling_wave(value, count);
  let lane = WavePrefixSum(1u);
  let lanes = WaveActiveSum(1u);
  if (lane == 0) {
    unified_sampling_scratch[subgroup_id()] = partial;
  }
  GroupMemoryBarrierWithGroupSync();
  if (subgroup_id() == 0) {
    var total = uint2(0, count ? 0u : 0xffffffffu);
    for (uint subgroup = lane; subgroup < subgroup_count(); subgroup += lanes) {
      let entry = unified_sampling_scratch[subgroup];
      if (count) {
        total.x += entry.x;
      } else if (entry.x > total.x || (entry.x == total.x && entry.y < total.y)) {
        total = entry;
      }
    }
    total = unified_sampling_wave(total, count);
    if (lane == 0) {
      unified_sampling_scratch[0] = total;
    }
  }
  GroupMemoryBarrierWithGroupSync();
  let result = unified_sampling_scratch[0];
  GroupMemoryBarrierWithGroupSync();
  return result;
}

// log(min_p) through staged_log, a positive subnormal scaled up by 2^24 exactly first; NaN for negative or NaN min_p.
float unified_sampling_log(float min_p) {
  let bits = asuint(min_p);
  if ((bits & 0x7fffffffu) == 0) {
    return UNIFIED_SAMPLING_NEGATIVE_INFINITY;
  }
  if (bits > 0x7f800000u) {
    return asfloat(0x7fc00000u);
  }
  if (bits == 0x7f800000u) {
    return min_p;
  }
  if (bits < 0x00800000u) {
    return staged_log(exact_scale_up_24(min_p)) - 16.635532333438687;
  }
  return staged_log(min_p);
}

// One sweep over the row by this invocation: tokens whose value outranks rank (numerically, then by lower index; rank
// (0, 0) admits every number) compete by their Gumbel-perturbed score for best. The first sweep gathers the running
// (maximum, normalizer) of finite and +infinite values; later ones count the competitors and sum exp(value - maximum).
// Tokens tile + column + 1024 w for w < 4 share one Philox counter, the CPU's sampling_index, and each read of a w and
// slot is contiguous across the workgroup.
void unified_sampling_sweep<T : IModelFloat>(
  Ptr<T, Access.Read> logits,
  Ptr<uint, Access.Read> mask,
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
  statistics = float2(UNIFIED_SAMPLING_NEGATIVE_INFINITY, 0.0);
  count = 0;
  let negative_infinity_key = unified_sampling_key(UNIFIED_SAMPLING_NEGATIVE_INFINITY);
  var tile = 0u;
  while (true) {
    let rest = vocab_size - tile;
    for (uint slot = 0; slot < 4; slot++) {
      let column = tid + slot * UNIFIED_SAMPLING_THREADS;
      if (column >= rest) {
        break;
      }
      float values[4];
      var competing = 0u;
      for (uint word = 0; word < 4; word++) {
        let offset = column + word * 1024;
        values[word] = 0.0;
        if (offset >= rest) {
          continue;
        }
        let token = tile + offset;
        let value = unified_sampling_value(logits, mask, token, inverse_temperature, has_bitmask, has_temperature);
        values[word] = value;
        let key = unified_sampling_key(value);
        if (first) {
          if (key != negative_infinity_key) {
            let previous = statistics.x;
            if (key > unified_sampling_key(previous)) {
              statistics.x = value;
            }
            if (has_top_p) {
              statistics.y = statistics.y * exp(exact_sum(previous, statistics.x, true)) +
                             exp(exact_sum(value, statistics.x, true));
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
            statistics.y += exp(exact_sum(value, maximum, true));
          }
        }
      }
      if (competing == 0) {
        continue;
      }
      var words = uint4(0);
      if (is_stochastic) {
        words = sampling_words(seed, sampling_index(tile + column, vocab_size).x);
      }
      for (uint word = 0; word < 4; word++) {
        if (((competing >> word) & 1u) == 0) {
          continue;
        }
        let token = tile + column + word * 1024;
        let score = is_stochastic ? ordered_sum(values[word], sampling_noise(sampling_uniform(words[word]))) : values[word];
        let key = unified_sampling_key(score);
        if (key != 0 && (key > best.x || (key == best.x && token < best.y))) {
          best = uint2(key, token);
        }
      }
    }
    if (rest <= 4096) {
      break;
    }
    tile += 4096;
  }
}

// The Metal kernel's candidate refinement with independent top-k, top-p and min-p filters: the Gumbel argmax is the
// first candidate; while a filter rejects it, the next is the Gumbel argmax among the tokens outranking it, up to 64
// candidates, the last returned unfiltered. Token 0 where no candidate is defined or a filter retains nothing (top_k 0,
// top_p at most 0, a min-p threshold above the maximum). Rows past the physical group limit loop.
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
  [[Groups("if batch_size == 0 { 0 } else { batch_size.min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[0]) }")]] uint group,
  [[Threads("256")]] uint tid,
) {
  static_assert(UNIFIED_SAMPLING_THREADS == 256, "256 invocations sweep 1024-token columns in four slots");
  let groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
  let inverse_temperature = has_temperature ? exact_quotient(1.0, temperature) : 1.0;
  let log_min_p = has_min_p ? unified_sampling_log(min_p) : 0.0;
  let mask_words = vocab_size / 32 + (vocab_size % 32 != 0 ? 1u : 0u);
  var row = group;
  while (row < batch_size) {
    let row_logits = element_pointer(logits, element_index(row, vocab_size, 0));
    let row_mask = element_pointer(bitmask, element_index(row, mask_words, 0));
    var seed = uint2(0, 0);
    if (is_stochastic) {
      let pair = element_pointer(seeds, element_index(row, 2, 0));
      seed = uint2(pair[0], pair[1]);
    }
    uint2 best;
    float2 statistics;
    uint count;
    unified_sampling_sweep(row_logits, row_mask, vocab_size, tid, seed, uint2(0, 0), 0.0, inverse_temperature, true,
                           is_stochastic, has_bitmask, has_temperature, has_top_p, best, statistics, count);
    var candidate = unified_sampling_reduce(best, false);
    var maximum = UNIFIED_SAMPLING_NEGATIVE_INFINITY;
    if (has_top_p || has_min_p) {
      maximum = workgroup_max<UNIFIED_SAMPLING_THREADS>(statistics.xx);
    }
    var norm = 0.0;
    if (has_top_p) {
      if (statistics.y != 0.0) {
        statistics.y *= exp(statistics.x - maximum);
      }
      norm = workgroup_sum<UNIFIED_SAMPLING_THREADS>(statistics.y);
    }
    let threshold_key = unified_sampling_key(ordered_sum(maximum, log_min_p));
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
        let value =
            unified_sampling_value(row_logits, row_mask, candidate.y, inverse_temperature, false, has_temperature);
        let rank = uint2(unified_sampling_key(value), candidate.y);
        unified_sampling_sweep(row_logits, row_mask, vocab_size, tid, seed, rank, maximum, inverse_temperature, false,
                               is_stochastic, has_bitmask, has_temperature, has_top_p, best, statistics, count);
        let next = unified_sampling_reduce(best, false);
        var passed = !(has_min_p && rank.x < threshold_key);
        if (has_top_k) {
          let above = unified_sampling_reduce(uint2(count, 0), true).x;
          passed = passed && above < top_k;
        }
        if (has_top_p) {
          let mass = workgroup_sum<UNIFIED_SAMPLING_THREADS>(statistics.y) / norm;
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
      element_pointer(output, uint2(row, 0))[0] = chosen;
    }
    if (batch_size - row <= groups.x) {
      break;
    }
    row += groups.x;
  }
}
