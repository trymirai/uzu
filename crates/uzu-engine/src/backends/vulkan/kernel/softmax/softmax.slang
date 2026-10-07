import definitions;

// The pinned Slang 2026.2 SDK has no wave-index/wave-count intrinsics; load the SPIR-V SubgroupId and NumSubgroups
// builtins directly.
uint subgroup_id() {
  return spirv_asm { result:$$uint = OpLoad builtin(SubgroupId:uint); };
}

uint subgroup_count() {
  return spirv_asm { result:$$uint = OpLoad builtin(NumSubgroups:uint); };
}

// Online row-wise softmax ported from the Metal kernel: each thread accumulates a running (max, normalizer) over
// strided chunks of the row, subgroups combine with wave reductions, and subgroups combine through shared memory.
// `has_sinks` seeds thread 0 with (sinks[outer_index], 1) as a virtual element. Every invocation reaches every
// barrier: there is no early exit, and masked tail elements contribute nothing.

static const uint SOFTMAX_THREADS = 256;
static const uint SOFTMAX_ELEMENTS_PER_THREAD = 4;
static const float SOFTMAX_LOWEST = asfloat(0xff7fffffu);  // -FLT_MAX

// One entry per subgroup; 256 covers the smallest possible subgroup width.
groupshared float softmax_subgroup_max[SOFTMAX_THREADS];
groupshared float softmax_subgroup_norm[SOFTMAX_THREADS];

// Merges two (max, normalizer) pairs.
float2 softmax_combine(float2 a, float2 b) {
  float max_value = max(a.x, b.x);
  return float2(max_value, a.y * exp(a.x - max_value) + b.y * exp(b.x - max_value));
}

// Reduces one (max, normalizer) pair per active lane across the subgroup.
float2 softmax_wave_combine(float2 value) {
  float max_value = WaveActiveMax(value.x);
  return float2(max_value, WaveActiveSum(value.y * exp(value.x - max_value)));
}

[[Kernel]] [[Public]] func Softmax<T: IStorageFloat>(
  Ptr<T, Access.ReadWrite> values,
  [[Optional("has_sinks")]] Ptr<T, Access.Read> sinks,
  uint row_length,
  uint outer_dim,
  uint batch_dim,
  [[Groups("if row_length == 0 { 0 } else { outer_dim }")]] uint outer_index,
  [[Groups("batch_dim")]] uint batch_index,
  [[Threads("256")]] uint tid,
  [[Specialize]] bool has_sinks,
) {
  let row = (outer_index * batch_dim + batch_index) * row_length;
  let stride = SOFTMAX_THREADS * SOFTMAX_ELEMENTS_PER_THREAD;

  float2 state = float2(SOFTMAX_LOWEST, 0.0);
  if (has_sinks && tid == 0) {
    state = float2(sinks[outer_index].toFloat(), 1.0);
  }

  // Pass 1: online (max, normalizer) over this thread's chunks.
  for (uint base = tid * SOFTMAX_ELEMENTS_PER_THREAD; base < row_length; base += stride) {
    float chunk[SOFTMAX_ELEMENTS_PER_THREAD];
    float chunk_max = state.x;
    for (uint i = 0; i < SOFTMAX_ELEMENTS_PER_THREAD; i++) {
      chunk[i] = SOFTMAX_LOWEST;
      if (base + i < row_length) {
        chunk[i] = values[row + base + i].toFloat();
      }
      chunk_max = max(chunk_max, chunk[i]);
    }
    float norm = state.y * exp(state.x - chunk_max);
    for (uint i = 0; i < SOFTMAX_ELEMENTS_PER_THREAD; i++) {
      if (base + i < row_length) {
        norm += exp(chunk[i] - chunk_max);
      }
    }
    state = float2(chunk_max, norm);
  }

  // Combine within each subgroup, then across all subgroups through shared memory.
  state = softmax_wave_combine(state);
  if (WavePrefixSum(1u) == 0) {
    softmax_subgroup_max[subgroup_id()] = state.x;
    softmax_subgroup_norm[subgroup_id()] = state.y;
  }
  GroupMemoryBarrierWithGroupSync();
  if (subgroup_id() == 0) {
    // Active lanes stride over every subgroup entry, even when there are more subgroups than lanes.
    let lane = WavePrefixSum(1u);
    let lanes = WaveActiveSum(1u);
    float2 total = float2(SOFTMAX_LOWEST, 0.0);
    for (uint subgroup = lane; subgroup < subgroup_count(); subgroup += lanes) {
      total = softmax_combine(total, float2(softmax_subgroup_max[subgroup], softmax_subgroup_norm[subgroup]));
    }
    total = softmax_wave_combine(total);
    if (WavePrefixSum(1u) == 0) {
      softmax_subgroup_max[0] = total.x;
      softmax_subgroup_norm[0] = total.y;
    }
  }
  GroupMemoryBarrierWithGroupSync();
  let final_max = softmax_subgroup_max[0];
  let inverse_norm = 1.0 / softmax_subgroup_norm[0];

  // Pass 2: normalize.
  for (uint base = tid * SOFTMAX_ELEMENTS_PER_THREAD; base < row_length; base += stride) {
    for (uint i = 0; i < SOFTMAX_ELEMENTS_PER_THREAD; i++) {
      if (base + i < row_length) {
        let index = row + base + i;
        values[index] = T.fromFloat(exp(values[index].toFloat() - final_max) * inverse_norm);
      }
    }
  }
}
