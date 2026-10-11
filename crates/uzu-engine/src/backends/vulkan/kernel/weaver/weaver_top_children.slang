import definitions;
import common.activation;
import common.sampling;
import common.storage_address;
import common.ranking;
import common.workgroup_reduction;
import generated.weaver;

groupshared uint4 weaver_top_reduce[512];
groupshared uint4 weaver_top_winner;

// One owner for the candidate's FP32 stages; the test entry observes the words before selection consumes them.
float4 weaver_top_values(float candidate, bf16 residual, uint2 key, uint token, uint vocab_size) {
  let logit = ordered_sum(candidate, residual.toFloat());
  let index = sampling_index(token, vocab_size);
  let word = sampling_words(key, index.x)[index.y];
  let noise = sampling_noise(float(word) * (1.0 / 4294967296.0));
  let target_noise = sampling_noise(sampling_uniform(word));
  return float4(logit, ordered_sum(logit, noise), noise, target_noise);
}

// Full, distinct CPU-shaped spans. Each U64 seed is its low/high U32 words, so no shader Int64 is required.
// RNG bits are exact; logarithms and parallel normalizers obey numerical GPU parity, including close score ties.
[[Kernel]] [[Public]]
func WeaverTopChildren(
  Ptr<bf16, Access.Read> residual_logits,
  Ptr<float, Access.Read> candidate_logits,
  Ptr<uint, Access.Read> candidate_ids,
  Ptr<uint, Access.Read> depth_seeds,
  Ptr<uint, Access.Read> node_metadata,
  Ptr<uint, Access.ReadWrite> output_token_ids,
  Ptr<float, Access.ReadWrite> output_model_logprobs,
  [[Optional("has_prune_noise")]] Ptr<float, Access.ReadWrite> output_prune_logprobs,
  uint rows,
  uint candidates,
  uint expand_width,
  uint vocab_size,
  [[Optional("has_prune_noise")]] float prune_noise_scale,
  [[Groups("if candidates == 0 || candidates > crate::backends::common::gpu_types::weaver::CANDIDATES_MAX || expand_width == 0 || expand_width > candidates { 0 } else { rows.min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[0]) }")]] uint group,
  [[Threads("256")]] uint tid,
  [[Specialize]] bool has_prune_noise,
) {
  static_assert(CANDIDATES_MAX == 512 && TOP_CHILDREN_THREADS == 256, "two cached candidates must cover each row");
  let groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
  let neutral = uint4(0, 0xffffffffu, 0xffffffffu, 0xffffffffu);
  var row = group;
  while (row < rows) {
    let depth = element_pointer(node_metadata, element_index(uint(MetadataIdx.Depth), rows, row))[0];
    let seed = element_pointer(depth_seeds, element_index(depth, 2, 0));
    let key = uint2(seed[0], seed[1]);
    float logits[2];
    float prune_logits[2];
    uint4 entries[2];
    for (uint cache = 0; cache < 2; cache++) {
      let column = tid + cache * 256;
      logits[cache] = asfloat(0xff800000u);
      prune_logits[cache] = asfloat(0xff800000u);
      entries[cache] = neutral;
      if (column >= candidates) { continue; }
      let source = element_index(row, candidates, column);
      let token = element_pointer(candidate_ids, source)[0];
      let values = weaver_top_values(element_pointer(candidate_logits, source)[0],
                                    element_pointer(residual_logits, source)[0], key, token, vocab_size);
      logits[cache] = values.x;
      entries[cache] = uint4(float_order_key(asuint(values.y)), token, column, column);
      if (has_prune_noise) {
        prune_logits[cache] = ordered_sum(values.x, exact_product(prune_noise_scale, values.w));
      }
    }
    let maximum = workgroup_max<256>(float2(logits[0], logits[1]));
    float local_sum = 0.0;
    for (uint cache = 0; cache < 2; cache++) {
      if (tid + cache * 256 < candidates) {
        local_sum = ordered_sum(local_sum, exp(exact_sum(logits[cache], maximum, true)));
      }
    }
    let log_sum = ordered_sum(staged_log(workgroup_sum<256>(local_sum)), maximum);
    float prune_log_sum = 0.0;
    if (has_prune_noise) {
      let prune_maximum = workgroup_max<256>(float2(prune_logits[0], prune_logits[1]));
      float prune_local_sum = 0.0;
      for (uint cache = 0; cache < 2; cache++) {
        if (tid + cache * 256 < candidates) {
          prune_local_sum = ordered_sum(prune_local_sum, exp(exact_sum(prune_logits[cache], prune_maximum, true)));
        }
      }
      prune_log_sum = ordered_sum(staged_log(workgroup_sum<256>(prune_local_sum)), prune_maximum);
    }
    if (expand_width < 32) {
      for (uint child = 0; child < expand_width; child++) {
        let partial = weaver_rank_wave(weaver_rank_better(entries[0], entries[1]));
        let lane = WavePrefixSum(1u);
        let lanes = WaveActiveSum(1u);
        if (lane == 0) { weaver_top_reduce[subgroup_id()] = partial; }
        GroupMemoryBarrierWithGroupSync();
        if (subgroup_id() == 0) {
          var total = neutral;
          for (uint subgroup = lane; subgroup < subgroup_count(); subgroup += lanes) {
            total = weaver_rank_better(total, weaver_top_reduce[subgroup]);
          }
          total = weaver_rank_wave(total);
          if (lane == 0) { weaver_top_winner = total; }
        }
        GroupMemoryBarrierWithGroupSync();
        let winner = weaver_top_winner;
        for (uint cache = 0; cache < 2; cache++) {
          if (tid + cache * 256 != winner.z) { continue; }
          let destination = element_index(row, expand_width, child);
          element_pointer(output_token_ids, destination)[0] = winner.y;
          element_pointer(output_model_logprobs, destination)[0] = exact_sum(logits[cache], log_sum, true);
          if (has_prune_noise) {
            element_pointer(output_prune_logprobs, destination)[0] = exact_sum(prune_logits[cache], prune_log_sum, true);
          }
          entries[cache] = neutral;
        }
      }
    } else {
      weaver_top_reduce[tid] = entries[0];
      weaver_top_reduce[tid + 256] = entries[1];
      GroupMemoryBarrierWithGroupSync();
      uint span = 1;
      while (span < candidates) { span <<= 1; }
      for (uint size = 2; size <= span; size <<= 1) {
        for (uint distance = size >> 1; distance > 0; distance >>= 1) {
          let lower = (tid & (distance - 1)) | ((tid & ~(distance - 1)) << 1);
          let higher = lower | distance;
          if (higher < span) {
            let a = weaver_top_reduce[lower];
            let b = weaver_top_reduce[higher];
            let best = weaver_rank_better(a, b);
            let worst = all(best == a) ? b : a;
            let forward = (lower & size) == 0;
            weaver_top_reduce[lower] = forward ? best : worst;
            weaver_top_reduce[higher] = forward ? worst : best;
          }
          GroupMemoryBarrierWithGroupSync();
        }
      }
      let sorted = uint2(weaver_top_reduce[tid].w, weaver_top_reduce[tid + 256].w);
      GroupMemoryBarrierWithGroupSync();
      for (uint cache = 0; cache < 2; cache++) {
        if (sorted[cache] < candidates) { weaver_top_reduce[sorted[cache]].w = tid + cache * 256; }
      }
      GroupMemoryBarrierWithGroupSync();
      for (uint cache = 0; cache < 2; cache++) {
        let column = tid + cache * 256;
        if (column >= candidates) { continue; }
        let rank = weaver_top_reduce[column].w;
        if (rank >= expand_width) { continue; }
        let destination = element_index(row, expand_width, rank);
        element_pointer(output_token_ids, destination)[0] = entries[cache].y;
        element_pointer(output_model_logprobs, destination)[0] = exact_sum(logits[cache], log_sum, true);
        if (has_prune_noise) {
          element_pointer(output_prune_logprobs, destination)[0] = exact_sum(prune_logits[cache], prune_log_sum, true);
        }
      }
    }
    if (groups.x > rows - 1 - row) { break; }
    row += groups.x;
  }
}

// Candidate bits, BF16 bits, seed low/high, token and vocabulary -> logit, score, selection and pruning noise words.
// NaN words are observed from the actual stage, since a different arithmetic operation need not produce the same NaN.
[[Kernel]] [[Test]]
func TestWeaverTopValues(
  Ptr<uint, Access.Read> inputs,
  Ptr<uint, Access.ReadWrite> outputs,
  uint count,
  [[Axis("count", "64")]] uint index,
) {
  if (index >= count) { return; }
  let record = element_pointer(inputs, element_index(index, 6, 0));
  bf16 residual;
  residual.bits = uint16_t(record[1]);
  let values = weaver_top_values(asfloat(record[0]), residual, uint2(record[2], record[3]), record[4], record[5]);
  let output = element_pointer(outputs, element_index(index, 4, 0));
  output[0] = asuint(values.x);
  output[1] = asuint(values.y);
  output[2] = asuint(values.z);
  output[3] = asuint(values.w);
}
