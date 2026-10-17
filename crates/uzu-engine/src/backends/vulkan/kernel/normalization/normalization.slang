import definitions;
import common.activation;
import common.hadamard;
import common.workgroup_reduction;

// Row normalization ported from the Metal kernel, one workgroup per row, its elementwise sums and products exact FP32
// roundings with subnormals kept. Every value is converted exactly where the Metal kernel converts it: the residual
// sum, its scaling and the shortcut store in InputT; normalization in FP32; the only-normalization scale product, bias
// sum, Hadamard input and output scaling rounded to OutputT. As in Metal, one pass accumulates each invocation's sum
// and sum of squares, the mean is sum / n and the variance sum of squares / n - mean². Every invocation reaches every
// barrier: loops that contain barriers run the same trip count in all invocations.

static const uint NORMALIZATION_THREADS = 1024;

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

// a - b, one rounding (NoContraction, as Slang 2026.2 emits no decoration for `precise`). Where both finite magnitudes
// are below 2^-101 the device would flush a subnormal operand or difference, so exact_sum forms it instead.
float normalization_difference(float a, float b) {
  [branch]
  if (max(asuint(a) & 0x7fffffffu, asuint(b) & 0x7fffffffu) < 0x0d000000u) {
    return exact_sum(a, b, true);
  }
  return spirv_asm {
    %difference:$$float = OpFSub $a $b;
    OpDecorate %difference NoContraction;
    result:$$float = OpCopyObject %difference;
  };
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
  [[Threads("1024")]] uint tid,
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

  // Step 1: residual add / shortcut copy, and this invocation's sum and sum of squares.
  float sum = 0.0;
  float squares = 0.0;
  for (uint i = tid; i < element_count; i += NORMALIZATION_THREADS) {
    InputT value;
    if (in_place) {
      value = InputT.fromFloat(output[row + i].toFloat());
    } else {
      value = input[row + i];
    }
    if (copy_to_shortcut) {
      if (residual_add) {
        value = InputT.fromFloat(exact_sum(value.toFloat(), shortcut[row + i].toFloat(), false));
        if (scale_residual_sum) {
          value = InputT.fromFloat(exact_product(value.toFloat(), post_layer_scalar));
        }
      }
      shortcut[row + i] = value;
    }
    let x = value.toFloat();
    if (subtract_mean) {
      sum = exact_sum(sum, x, false);
    }
    squares = accumulate_squares(squares, x);
  }

  // Step 2: mean, variance and inverse RMS over the row. Each invocation reads back only the elements it stored in step
  // 1, so no memory barrier is needed.
  float mean = 0.0;
  if (subtract_mean) {
    mean = staged_mean(workgroup_sum<NORMALIZATION_THREADS>(sum), element_count);
  }
  let mean_square = staged_mean(workgroup_sum<NORMALIZATION_THREADS>(squares), element_count);
  let variance = exact_sum(mean_square, -exact_product(mean, mean), false);
  let rms_inv = reciprocal_root(exact_sum(variance, epsilon, false));

  // Step 3: elementwise normalization.
  for (uint base = 0; base < element_count; base += NORMALIZATION_THREADS) {
    let i = base + tid;
    let active = i < element_count;
    OutputT value = OutputT.fromFloat(0.0);
    if (active) {
      // Without subtract_mean x - 0 is x itself, signed zeros and subnormals included; NaN stays NaN, its payload's
      // quieting unpromised either way.
      var numerator = normalization_stored(input, output, shortcut, row + i, in_place, residual_add);
      if (subtract_mean) {
        numerator = normalization_difference(numerator, mean);
      }
      let normalized = exact_product(numerator, rms_inv);
      if (has_scales) {
        let scale = exact_sum(scales[i].toFloat(), scale_offset, false);
        if (full_layer) {
          value = OutputT.fromFloat(exact_product(normalized, scale));
        } else {
          value = OutputT.fromFloat(exact_product(OutputT.fromFloat(normalized).toFloat(), OutputT.fromFloat(scale).toFloat()));
        }
      } else {
        value = OutputT.fromFloat(normalized);
      }
      if (has_biases) {
        value = OutputT.fromFloat(exact_sum(value.toFloat(), biases[i].toFloat(), false));
      }
    }
    // element_count is a multiple of the block size, so every block is entirely active or entirely inactive.
    if (use_hadamard) {
      float factored = 0.0;
      if (active) {
        factored = hadamard_factor(value.toFloat(), hadamard_factors[i]);
      }
      // The input random Hadamard transform: the factors, then the transform.
      value = OutputT.fromFloat(hadamard_transform<NORMALIZATION_THREADS>(factored, tid));
    }
    if (active) {
      if (scale_output) {
        value = OutputT.fromFloat(exact_product(value.toFloat(), post_layer_scalar));
      }
      output[row + i] = value;
    }
  }
}
