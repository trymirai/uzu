import definitions;
import generated.hadamard_order;

// Row normalization ported from the Metal kernel, one workgroup per row. Every value is converted exactly where the
// Metal kernel converts it: the residual sum, its scaling and the shortcut store in InputT; normalization in FP32;
// the only-normalization scale product, bias sum, Hadamard input and output scaling rounded to OutputT. With
// subtract_mean the variance is a shifted two-pass one around the row's first stored element. Every invocation reaches
// every barrier: loops that contain barriers run the same trip count in all invocations.

static const uint NORMALIZATION_THREADS = 256;

// One entry per subgroup; 256 covers the smallest possible subgroup width.
groupshared float normalization_subgroup_sums[NORMALIZATION_THREADS];
groupshared float normalization_pivot;
// Ping-pong Hadamard butterfly stages, one element per invocation.
groupshared float normalization_hadamard_stages[2][NORMALIZATION_THREADS];

// Sums over the workgroup as Softmax combines its pairs: subgroup sums, one shared entry per subgroup, then subgroup 0
// sums the entries.
float normalization_workgroup_sum(float value) {
  value = WaveActiveSum(value);
  if (WavePrefixSum(1u) == 0) {
    normalization_subgroup_sums[subgroup_id()] = value;
  }
  GroupMemoryBarrierWithGroupSync();
  if (subgroup_id() == 0) {
    // Active lanes stride over every subgroup entry, even when there are more subgroups than lanes.
    let lane = WavePrefixSum(1u);
    let lanes = WaveActiveSum(1u);
    float total = 0.0;
    for (uint subgroup = lane; subgroup < subgroup_count(); subgroup += lanes) {
      total += normalization_subgroup_sums[subgroup];
    }
    total = WaveActiveSum(total);
    if (WavePrefixSum(1u) == 0) {
      normalization_subgroup_sums[0] = total;
    }
  }
  GroupMemoryBarrierWithGroupSync();
  let total = normalization_subgroup_sums[0];
  // Every invocation has read the total before another sum reuses the scratch.
  GroupMemoryBarrierWithGroupSync();
  return total;
}

// The stored value of a row element after step 1: the shortcut after a residual add, otherwise the input.
float normalization_stored<InputT: IStorageFloat, OutputT: IStorageFloat>(
  Ptr<InputT, Access.Read> input,
  Ptr<OutputT, Access.ReadWrite> output,
  Ptr<InputT, Access.ReadWrite> shortcut,
  uint index,
  bool in_place,
  bool residual_add
) {
  if (residual_add) {
    return shortcut[index].toFloat();
  }
  if (in_place) {
    return output[index].toFloat();
  }
  return input[index].toFloat();
}

// a - b evaluated as written (NoContraction): reassociating (x - pivot) - mean_delta into x - (pivot + mean_delta)
// would round the shift at the row's magnitude, and Slang 2026.2 emits no decoration for `precise`.
float normalization_difference(float a, float b) {
  return spirv_asm {
    %difference:$$float = OpFSub $a $b;
    OpDecorate %difference NoContraction;
    result:$$float = OpCopyObject %difference;
  };
}

// Input random Hadamard transform of logical blocks of HADAMARD_TRANSFORM_BLOCK_SIZE consecutive invocations, as the
// Metal simdgroup transform: the butterfly of the factored values, divided by sqrt(block size). Blocks live in
// workgroup shared memory, so they do not depend on the device subgroup width.
float normalization_hadamard(float value, uint tid) {
  static_assert(
    HADAMARD_TRANSFORM_BLOCK_SIZE > 0 && (HADAMARD_TRANSFORM_BLOCK_SIZE & (HADAMARD_TRANSFORM_BLOCK_SIZE - 1)) == 0,
    "Hadamard transform block size must be a power of two"
  );
  static_assert(
    NORMALIZATION_THREADS % HADAMARD_TRANSFORM_BLOCK_SIZE == 0, "Hadamard transform blocks must tile the workgroup"
  );
  normalization_hadamard_stages[0][tid] = value;
  GroupMemoryBarrierWithGroupSync();
  uint stage = 0;
  for (uint stride = 1; stride < HADAMARD_TRANSFORM_BLOCK_SIZE; stride <<= 1) {
    let own = normalization_hadamard_stages[stage][tid];
    let partner = normalization_hadamard_stages[stage][tid ^ stride];
    stage ^= 1;
    normalization_hadamard_stages[stage][tid] = (tid & stride) != 0 ? partner - own : partner + own;
    GroupMemoryBarrierWithGroupSync();
  }
  let transformed = normalization_hadamard_stages[stage][tid];
  // An even stage count ends in stage 0, which the next call overwrites: every invocation reads first.
  GroupMemoryBarrierWithGroupSync();
  return transformed / sqrt(float(HADAMARD_TRANSFORM_BLOCK_SIZE));
}

[[Kernel]] [[Public]] func Normalization<
  InputT: IStorageFloat,
  AffineT: IStorageFloat,
  OutputT: IStorageFloat,
  AccumT: IFloat32
>(
  [[Optional("!in_place")]] Ptr<InputT, Access.Read> input,
  [[Optional("has_scales")]] Ptr<AffineT, Access.Read> scales,
  [[Optional("has_biases")]] Ptr<AffineT, Access.Read> biases,
  Ptr<OutputT, Access.ReadWrite> output,
  [[Optional("copy_to_shortcut")]] Ptr<InputT, Access.ReadWrite> shortcut,
  [[Optional("use_hadamard")]] Ptr<int, Access.Read> hadamard_factors,
  uint batch_size,
  uint element_count,
  float epsilon,
  float scale_offset,
  float post_layer_scalar,
  [[Groups("if element_count == 0 { 0 } else { batch_size }")]] uint batch_idx,
  [[Threads("256")]] uint tid,
  [[Specialize]] bool in_place,
  [[Specialize]] bool subtract_mean,
  [[Specialize]] bool full_layer,
  [[Specialize]] bool copy_to_shortcut,
  [[Specialize]] bool residual_add,
  [[Specialize]] bool use_hadamard,
  [[Specialize]] bool scale_residual_sum,
  [[Specialize]] bool scale_output,
  [[Specialize]] bool has_biases,
  [[Specialize]] bool has_scales,
) {
  let row = batch_idx * element_count;

  // Step 1: residual add / shortcut copy, and the partial sum of squares of the RMS path.
  float partial = 0.0;
  for (uint i = tid; i < element_count && (copy_to_shortcut || !subtract_mean); i += NORMALIZATION_THREADS) {
    InputT value;
    if (in_place) {
      value = InputT.fromFloat(output[row + i].toFloat());
    } else {
      value = input[row + i];
    }
    if (copy_to_shortcut) {
      if (residual_add) {
        value = InputT.fromFloat(value.toFloat() + shortcut[row + i].toFloat());
        if (scale_residual_sum) {
          value = InputT.fromFloat(value.toFloat() * post_layer_scalar);
        }
      }
      shortcut[row + i] = value;
    }
    let x = value.toFloat();
    partial += x * x;
  }

  // Step 2: inverse RMS over the row. Each invocation reads back only the elements it stored in step 1, so no memory
  // barrier is needed. With subtract_mean, deltas from the first stored element are exact for near-constant rows, and
  // squared deviations from their mean avoid the cancellation of E[x²] - mean².
  let count = float(element_count);
  float pivot = 0.0;
  float mean_delta = 0.0;
  float sum_of_squares;
  if (subtract_mean) {
    if (tid == 0) {
      normalization_pivot = normalization_stored(input, output, shortcut, row, in_place, residual_add);
    }
    GroupMemoryBarrierWithGroupSync();
    pivot = normalization_pivot;
    float delta_sum = 0.0;
    for (uint i = tid; i < element_count; i += NORMALIZATION_THREADS) {
      let x = normalization_stored(input, output, shortcut, row + i, in_place, residual_add);
      delta_sum += normalization_difference(x, pivot);
    }
    mean_delta = normalization_workgroup_sum(delta_sum) / count;
    float squares = 0.0;
    for (uint i = tid; i < element_count; i += NORMALIZATION_THREADS) {
      let x = normalization_stored(input, output, shortcut, row + i, in_place, residual_add);
      let deviation = normalization_difference(normalization_difference(x, pivot), mean_delta);
      squares += deviation * deviation;
    }
    sum_of_squares = normalization_workgroup_sum(squares);
  } else {
    sum_of_squares = normalization_workgroup_sum(partial);
  }
  let rms_inv = rsqrt(sum_of_squares / count + epsilon);

  // Step 3: elementwise normalization.
  for (uint base = 0; base < element_count; base += NORMALIZATION_THREADS) {
    let i = base + tid;
    let active = i < element_count;
    OutputT value = OutputT.fromFloat(0.0);
    if (active) {
      let x = normalization_stored(input, output, shortcut, row + i, in_place, residual_add);
      let normalized = normalization_difference(normalization_difference(x, pivot), mean_delta) * rms_inv;
      if (has_scales) {
        let scale = scales[i].toFloat() + scale_offset;
        if (full_layer) {
          value = OutputT.fromFloat(normalized * scale);
        } else {
          value = OutputT.fromFloat(OutputT.fromFloat(normalized).toFloat() * OutputT.fromFloat(scale).toFloat());
        }
      } else {
        value = OutputT.fromFloat(normalized);
      }
      if (has_biases) {
        value = OutputT.fromFloat(value.toFloat() + biases[i].toFloat());
      }
    }
    // element_count is a multiple of the block size, so every block is entirely active or entirely inactive.
    if (use_hadamard) {
      float factored = 0.0;
      if (active) {
        factored = value.toFloat() * float(hadamard_factors[i]);
      }
      value = OutputT.fromFloat(normalization_hadamard(factored, tid));
    }
    if (active) {
      if (scale_output) {
        value = OutputT.fromFloat(value.toFloat() * post_layer_scalar);
      }
      output[row + i] = value;
    }
  }
}
