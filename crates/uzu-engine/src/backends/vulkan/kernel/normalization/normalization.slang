import definitions;
import common.hadamard;
import common.workgroup_reduction;

// One workgroup per row. The residual sum, its scaling and the shortcut store round to InputT; normalization is FP32
// from one pass of sums and sums of squares (variance E[x²] - mean²); scale products, bias sums and the Hadamard scalings
// round to OutputT. Loops containing barriers run the same trip count in every invocation.

static const uint NORMALIZATION_THREADS = 1024;

[[Variants("InputT", "float, half, bf16")]]
[[Variants("AffineT", "float, half, bf16")]]
[[Variants("OutputT", "float, half, bf16")]]
[[Variants("AccumT", "float")]]
[[Kernel]] [[Public]]
[[Precondition("new", "copy_to_shortcut || !residual_add")]]
func Normalization<
  InputT: IStorageFloat,
  AffineT: IStorageFloat,
  OutputT: IStorageFloat,
  AccumT: IStorageFloat
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
  let invocation = subgroup_invocation();

  // Step 1: residual add / shortcut copy, and this invocation's sum and sum of squares.
  float sum = 0.0;
  float squares = 0.0;
  for (uint i = invocation; i < element_count; i += NORMALIZATION_THREADS) {
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
    if (subtract_mean) {
      sum += x;
    }
    squares += x * x;
  }

  // Step 2: mean, variance and inverse RMS over the row. Each invocation reads back only the elements it stored in step
  // 1, so no memory barrier is needed.
  float mean = 0.0;
  if (subtract_mean) {
    mean = workgroup_reduce<Sum<float>, NORMALIZATION_THREADS>(sum) / float(element_count);
  }
  let variance = workgroup_reduce<Sum<float>, NORMALIZATION_THREADS>(squares) / float(element_count) - mean * mean;
  let rms_inv = rsqrt(variance + epsilon);

  // Step 3: elementwise normalization.
  for (uint base = 0; base < element_count; base += NORMALIZATION_THREADS) {
    let i = base + invocation;
    let active = i < element_count;
    OutputT value = OutputT.fromFloat(0.0);
    if (active) {
      let x = residual_add ? shortcut[row + i].toFloat() : in_place_load(input, output, row + i, in_place);
      let normalized = (x - mean) * rms_inv;
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
        factored = hadamard_factor(value.toFloat(), hadamard_factors[i]);
      }
      // The input random Hadamard transform: the factors, then the transform.
      value = OutputT.fromFloat(hadamard_transform<NORMALIZATION_THREADS>(factored));
    }
    if (active) {
      if (scale_output) {
        value = OutputT.fromFloat(value.toFloat() * post_layer_scalar);
      }
      output[row + i] = value;
    }
  }
}
