import definitions;
import common.activation;
import generated.ring;
import generated.trie;

static const uint ATTENTION_THREADS = 1024;
static const uint ATTENTION_TILE = 64;
static const uint ATTENTION_MAX_HEAD_DIM = 512;
// Head dimensions past the workgroup's threads take more than one accumulator per thread.
static const uint ATTENTION_DIMS_PER_THREAD = (ATTENTION_MAX_HEAD_DIM + ATTENTION_THREADS - 1) / ATTENTION_THREADS;
static const uint ATTENTION_NONE = 0xffffffffu;
static const float ATTENTION_NEGATIVE_INFINITY = asfloat(0xff800000u);
// e^x from here down is below the normal range, which this device class flushes: it is taken as 2^(x log2(e) + 24) and
// rounded onto the FP32 subnormal grid exactly, staged as the CPU rounds its e^x to FP32.
static const float ATTENTION_SUBNORMAL_EXPONENT = -87.0;
static const float ATTENTION_LOG2E = 1.44269504;
// 2^122: while each slice's values sum below it in magnitude, the 16 slices of the largest workgroup stay below 2^126,
// so no weighted sum overflows, on the CPU either, as weights and rescales are at most 1.
static const float ATTENTION_MAGNITUDE_LIMIT = asfloat(0x7c800000u);

groupshared float attention_query[ATTENTION_MAX_HEAD_DIM];
groupshared uint attention_query_tiny;
// Per tile, double-buffered by tile parity: a tile's scores, flags and per-subgroup maxima and first used keys are
// written before its first barrier and its weights before its second; the next tile writes the other buffer, and the
// one after reuses this one only after every thread has passed the next tile's first barrier.
groupshared float attention_scores[2][ATTENTION_TILE];
groupshared bool attention_used[2][ATTENTION_TILE];
// Weight bits, so subnormal weights pass through shared memory unchanged.
groupshared uint attention_weights[2][ATTENTION_TILE];
// One entry per subgroup; the workgroup size covers the smallest possible subgroup width.
groupshared float attention_subgroup_max[2][ATTENTION_THREADS];
groupshared uint attention_subgroup_first[2][ATTENTION_THREADS];
groupshared uint attention_partial[ATTENTION_THREADS];
groupshared uint attention_normalizer[ATTENTION_THREADS];
// Per tile parity: set when a slice's values reach ATTENTION_MAGNITUDE_LIMIT or are not finite; cleared before the
// tile's values are read and read after the next tile's first barrier or after the last tile.
groupshared uint attention_unsafe[2];

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

// a + b as exact_sum, its native sum decorated NoContraction so it is never fused with the product feeding it: the
// per-key path rounds each product and each sum as the CPU does.
float attention_ordered_sum(float a, float b) {
  [branch]
  if (max(asuint(a) & 0x7fffffffu, asuint(b) & 0x7fffffffu) < 0x0d000000u) {
    return exact_sum(a, b, false);
  }
  return spirv_asm {
    %sum:$$float = OpFAdd $a $b;
    OpDecorate %sum NoContraction;
    result:$$float = OpCopyObject %sum;
  };
}

// The CPU's per-key steps over one tile's used keys, by slice 0, which holds the whole state: the running maximum
// ignoring NaN as f32::max does, then the rescale and the weight rounded to FP32 and every product and sum exact, in
// its order.
void attention_ordered_tile<T: IModelFloat>(
  Ptr<T, Access.Read> values,
  uint buffer,
  uint row,
  uint row_stride,
  uint slice_dims,
  uint dims_per_thread,
  inout float max_score,
  inout float normalizer,
  inout float accumulators[ATTENTION_DIMS_PER_THREAD]
) {
  for (uint index = 0; index < ATTENTION_TILE; index++) {
    if (attention_used[buffer][index]) {
      let score = attention_scores[buffer][index];
      let next = isnan(score) ? max_score : (isnan(max_score) ? score : max(max_score, score));
      let factor = attention_exp(max_score - next);
      let weight = attention_exp(score - next);
      for (uint a = 0; a < dims_per_thread; a++) {
        let product = exact_product(weight, values[row + index * row_stride + a * slice_dims].toFloat());
        accumulators[a] = attention_ordered_sum(exact_product(accumulators[a], factor), product);
      }
      normalizer = attention_ordered_sum(exact_product(normalizer, factor), weight);
      max_score = next;
    }
  }
}

// The CPU should_use_key, in its unsigned arithmetic: suffix keys at their trie height or suffix index, ancestry or
// index order under causality; ring prefix keys at their ring position, filled entries only; then the causal or
// centered sliding window around the query position.
bool attention_use_key(
  uint key,
  uint query,
  uint prefix_length,
  uint suffix_position,
  uint query_position,
  RingParams ring_params,
  Ptr<TrieNode, Access.Read> trie,
  uint sliding_window_size,
  bool is_kv_cache_ring,
  bool is_causal,
  bool is_trie,
  bool is_sliding_window
) {
  var use_key = true;
  uint key_position;
  if (key >= prefix_length) {
    let in_suffix = key - prefix_length;
    if (is_trie) {
      let node = trie[in_suffix];
      key_position = suffix_position + node.height;
      if (is_causal) {
        use_key = query >= node.trie_start && query <= node.trie_end;
      }
    } else {
      key_position = suffix_position + in_suffix;
      if (is_causal) {
        use_key = in_suffix <= query;
      }
    }
  } else if (is_kv_cache_ring) {
    key_position = (prefix_length + key - ring_params.ring_offset) % prefix_length;
    use_key = key_position < ring_params.ring_length;
  } else {
    key_position = key;
  }
  if (is_sliding_window) {
    if (is_causal) {
      use_key = use_key && key_position <= query_position && query_position - key_position < sliding_window_size;
    } else if (key_position <= query_position) {
      use_key = use_key && query_position - key_position <= sliding_window_size / 2;
    } else {
      use_key = use_key && key_position - query_position <= sliding_window_size / 2;
    }
  }
  return use_key;
}

// Attention of query `query` of head `head` over the keys the CPU mask admits, one workgroup each, in tiles of
// ATTENTION_TILE keys with the online softmax of the Metal kernel at tile granularity, for any subgroup width. Each
// subgroup scores keys of the tile, its lanes striding the head dimension (masked keys are never read); the workgroup
// takes the tile maximum, rescales once, and each thread accumulates its dimensions over one slice of the tile's keys
// (masked keys are never read). The running maximum is uniform, so slices sum without rescaling. As the CPU: the query is
// scaled exactly in FP32 first; a sink is a key of value zero; a rescale by e^0 is skipped as exact; NaN and infinite
// scores and maxima propagate through the rescale as in the CPU's per-key order, including the NaN of a first used key
// scoring -inf against no sink; no used key and no sink give NaN. Weights and rescales are rounded to FP32 before they
// multiply, and every weighted value, sum and the quotient keep subnormals through the exact helpers, which this device
// class would flush; score products with a subnormal operand are exact too, so no flushed value weighs in a normal one.
// Values that are not finite or may overflow replay the whole sequence in the CPU's per-key order, whose classes the
// tile order does not keep; finite values below the magnitude limit agree with the CPU within rounding, exp
// approximation and reordering.
[[Kernel]] [[Public]]
[[Precondition("new", "HEAD_DIM == 64 || HEAD_DIM == 128 || HEAD_DIM == 256 || HEAD_DIM == 512")]]
[[Precondition("encode", "gqa_factor > 0")]]
[[Precondition("encode", "suffix_length <= sequence_length")]]
func AttentionSinglePass<T: IModelFloat>(
  Ptr<T, Access.Read> queries,
  Ptr<T, Access.Read> keys,
  Ptr<T, Access.Read> values,
  Ptr<T, Access.ReadWrite> out,
  uint gqa_factor,
  uint sequence_length,
  uint k_head_stride,
  uint k_seq_stride,
  uint v_head_stride,
  uint v_seq_stride,
  [[Optional("is_kv_cache_ring")]] RingParams ring_params,
  float scale,
  [[Optional("is_trie")]] Ptr<TrieNode, Access.Read> trie,
  [[Optional("is_sliding_window")]] uint sliding_window_size,
  [[Optional("has_sinks")]] Ptr<T, Access.Read> sinks,
  uint num_heads,
  uint suffix_length,
  [[Groups("num_heads")]] uint head,
  [[Groups("suffix_length")]] uint query,
  [[Threads("1024")]] uint tid,
  [[Specialize]] uint HEAD_DIM,
  [[Specialize]] bool has_sinks,
  [[Specialize]] bool is_kv_cache_ring,
  [[Specialize]] bool is_causal,
  [[Specialize]] bool is_trie,
  [[Specialize]] bool is_sliding_window,
) {
  static_assert(ATTENTION_THREADS == 1024, "one workgroup of Threads(1024) per head and query");
  let lane = WavePrefixSum(1u);
  let lanes = WaveActiveSum(1u);
  let subgroup = subgroup_id();
  let subgroups = subgroup_count();

  let prefix_length = sequence_length - suffix_length;
  let suffix_position = is_kv_cache_ring ? ring_params.ring_length : prefix_length;
  var query_position = suffix_position + query;
  if (is_trie) {
    query_position = suffix_position + trie[query].height;
  }
  let key_base = head / gqa_factor * k_head_stride;
  let value_base = head / gqa_factor * v_head_stride;

  if (tid == 0) {
    attention_query_tiny = 0;
    attention_unsafe[0] = 0;
    attention_unsafe[1] = 0;
  }
  GroupMemoryBarrierWithGroupSync();
  for (uint j = tid; j < HEAD_DIM; j += ATTENTION_THREADS) {
    let scaled = exact_product(scale, queries[(head * suffix_length + query) * HEAD_DIM + j].toFloat());
    attention_query[j] = scaled;
    if (attention_subnormal(scaled)) {
      InterlockedOr(attention_query_tiny, 1u);
    }
  }
  GroupMemoryBarrierWithGroupSync();
  let query_tiny = attention_query_tiny != 0;

  // Each thread owns dimensions dim + a slice_dims of the keys of its slice.
  let slice_dims = min(HEAD_DIM, ATTENTION_THREADS);
  let slices = ATTENTION_THREADS / slice_dims;
  let slice = tid / slice_dims;
  let dim = tid % slice_dims;
  let dims_per_thread = HEAD_DIM / slice_dims;
  float accumulators[ATTENTION_DIMS_PER_THREAD];
  var max_score = ATTENTION_NEGATIVE_INFINITY;
  var normalizer = 0.0;
  var poisoned = false;
  // The tile-wide rescale may give an infinity or an overflowing sum another class than the CPU's per-key rescales, and
  // slice sums may round differently near overflow. So once a slice's values reach the magnitude limit or are not
  // finite, the whole sequence is replayed from the initial state in the CPU's per-key order by slice 0.
  var ordered = false;
  while (true) {
    for (uint a = 0; a < ATTENTION_DIMS_PER_THREAD; a++) {
      accumulators[a] = 0.0;
    }
    max_score = ATTENTION_NEGATIVE_INFINITY;
    normalizer = 0.0;
    if (has_sinks) {
      max_score = sinks[head].toFloat();
      normalizer = slice == 0 ? 1.0 : 0.0;
    }
    poisoned = false;
    var any_used = false;
    var magnitude = 0.0;

    uint buffer = 0;
    for (uint tile = 0; tile < sequence_length; tile += ATTENTION_TILE) {
      var subgroup_max = ATTENTION_NEGATIVE_INFINITY;
      var subgroup_first = ATTENTION_NONE;
      for (uint index = subgroup; index < ATTENTION_TILE; index += subgroups) {
        let key = tile + index;
        var used = false;
        if (key < sequence_length) {
          used = attention_use_key(
            key, query, prefix_length, suffix_position, query_position, ring_params, trie, sliding_window_size,
            is_kv_cache_ring, is_causal, is_trie, is_sliding_window
          );
        }
        var score = 0.0;
        if (used) {
          let row = key_base + key * k_seq_stride;
          var partial = 0.0;
          var tiny = query_tiny;
          for (uint j = lane; j < HEAD_DIM; j += lanes) {
            let k = keys[row + j].toFloat();
            partial += attention_query[j] * k;
            tiny = tiny || attention_subnormal(k);
          }
          // A subnormal operand, flushed natively, may weigh in a normal score: the products again, exactly.
          if (WaveActiveAnyTrue(tiny)) {
            partial = 0.0;
            for (uint j = lane; j < HEAD_DIM; j += lanes) {
              partial += exact_product(attention_query[j], keys[row + j].toFloat());
            }
          }
          score = WaveActiveSum(partial);
          subgroup_max = max(subgroup_max, score);
          subgroup_first = min(subgroup_first, index);
        }
        if (lane == 0) {
          attention_scores[buffer][index] = score;
          attention_used[buffer][index] = used;
        }
      }
      if (lane == 0) {
        attention_subgroup_max[buffer][subgroup] = subgroup_max;
        attention_subgroup_first[buffer][subgroup] = subgroup_first;
      }
      GroupMemoryBarrierWithGroupSync();
      // The previous tile's flag, final once every invocation has passed this barrier: replay from the start.
      if (tile > 0 && attention_unsafe[buffer ^ 1] != 0) {
        break;
      }

      // Each subgroup reduces every entry, its lanes striding them: the same tile maximum and first used key
      // everywhere, maxima being exact (a NaN maximum, which the device may or may not return, leaves the output NaN
      // either way).
      var tile_max = ATTENTION_NEGATIVE_INFINITY;
      var first = ATTENTION_NONE;
      for (uint entry = lane; entry < subgroups; entry += lanes) {
        tile_max = max(tile_max, attention_subgroup_max[buffer][entry]);
        first = min(first, attention_subgroup_first[buffer][entry]);
      }
      tile_max = WaveActiveMax(tile_max);
      first = WaveActiveMin(first);
      let tile_used = first != ATTENTION_NONE;
      let previous_max = max_score;
      if (tile_used) {
        if (!any_used && max_score == ATTENTION_NEGATIVE_INFINITY && attention_scores[buffer][first] == max_score) {
          poisoned = true;
        }
        any_used = true;
        if (!ordered) {
          max_score = max(max_score, tile_max);
          if (tid < ATTENTION_TILE && attention_used[buffer][tid]) {
            attention_weights[buffer][tid] = asuint(attention_exp(attention_scores[buffer][tid] - max_score));
          }
        }
      }
      if (tid == 0) {
        attention_unsafe[buffer] = 0;
      }
      GroupMemoryBarrierWithGroupSync();

      if (tile_used && ordered) {
        if (slice == 0) {
          let row = value_base + tile * v_seq_stride + dim;
          attention_ordered_tile(
            values, buffer, row, v_seq_stride, slice_dims, dims_per_thread, max_score, normalizer, accumulators
          );
        }
      } else if (tile_used) {
        // The CPU's staging, subnormals kept: the rescale and each weight rounded to FP32, then each product and sum.
        if (!(max_score == previous_max && isfinite(previous_max))) {
          let factor = attention_exp(previous_max - max_score);
          for (uint a = 0; a < dims_per_thread; a++) {
            accumulators[a] = exact_product(accumulators[a], factor);
          }
          normalizer = exact_product(normalizer, factor);
        }
        for (uint index = slice; index < ATTENTION_TILE; index += slices) {
          if (attention_used[buffer][index]) {
            let weight = asfloat(attention_weights[buffer][index]);
            let row = value_base + (tile + index) * v_seq_stride + dim;
            for (uint a = 0; a < dims_per_thread; a++) {
              let value = values[row + a * slice_dims].toFloat();
              magnitude += abs(value);
              accumulators[a] = exact_sum(accumulators[a], exact_product(weight, value), false);
            }
            normalizer = exact_sum(normalizer, weight, false);
          }
        }
        if (!(magnitude < ATTENTION_MAGNITUDE_LIMIT)) {
          InterlockedOr(attention_unsafe[buffer], 1u);
        }
      }
      buffer ^= 1;
    }
    // After a break `buffer ^ 1` is the flagged tile, otherwise the last one; an ordered pass sets no flag.
    GroupMemoryBarrierWithGroupSync();
    if (ordered || attention_unsafe[buffer ^ 1] == 0) {
      break;
    }
    ordered = true;
  }

  // Slices sum in order, as bits through shared memory; the normalizer holds 1 from the maximum's key or the sink unless
  // no key was used. Where the quotient may be subnormal, the numerator is scaled up by 2^24 exactly and the quotient
  // rounded back down.
  if (dim == 0) {
    attention_normalizer[slice] = asuint(normalizer);
  }
  for (uint a = 0; a < dims_per_thread; a++) {
    attention_partial[tid] = asuint(accumulators[a]);
    GroupMemoryBarrierWithGroupSync();
    if (slice == 0) {
      var numerator = asfloat(attention_partial[dim]);
      var denominator = asfloat(attention_normalizer[0]);
      for (uint other = 1; other < slices; other++) {
        numerator = exact_sum(numerator, asfloat(attention_partial[other * slice_dims + dim]), false);
        denominator = exact_sum(denominator, asfloat(attention_normalizer[other]), false);
      }
      var result = numerator / denominator;
      if (poisoned || denominator == 0.0) {
        result = asfloat(0x7fc00000u);
      } else if (((asuint(numerator) >> 23) & 0xffu) + 126 <= ((asuint(denominator) >> 23) & 0xffu)) {
        result = exact_scale_down(exact_scale_up_24(numerator) / denominator, 24);
      }
      out[(query * num_heads + head) * HEAD_DIM + a * slice_dims + dim] = T.fromFloat(result);
    }
    GroupMemoryBarrierWithGroupSync();
  }
}
