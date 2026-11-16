import definitions;
import common.hadamard;
import common.workgroup_reduction;

static const uint NORMALIZATION_THREADS = 1024;

[[Variants("InputT", "float, half, bf16")]]
[[Variants("AffineT", "float, half, bf16")]]
[[Variants("OutputT", "float, half, bf16")]]
[[Variants("AccumT", "float")]]
[[Kernel]] [[Public]]
[[Precondition("new", "hadamard_size.is_power_of_two() && hadamard_size <= 128")]]
[[Precondition("new", "!in_place || InputT == OutputT")]]
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
  uint mixture_size,
  [[Groups("batch_size")]] uint batch_idx,
  [[Threads("1024")]] uint tid,
  [[Specialize]] bool in_place,
  [[Specialize]] bool subtract_mean,
  [[Specialize]] bool full_layer,
  [[Specialize]] bool copy_to_shortcut,
  [[Specialize]] bool residual_add,
  [[Specialize]] bool use_hadamard,
  [[Specialize]] uint hadamard_size,
  [[Specialize]] bool has_biases,
  [[Specialize]] bool has_scales,
) {
  static_assert(NORMALIZATION_THREADS == 1024, "Threads(1024)");
  let row = batch_idx * element_count;
  let affine_row = (batch_idx % mixture_size) * element_count;
  let invocation = subgroup_invocation();

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
      }
      shortcut[row + i] = value;
    }
    let x = value.toFloat();
    if (subtract_mean) {
      sum += x;
    }
    squares += x * x;
  }

  float mean = 0.0;
  if (subtract_mean) {
    mean = workgroup_reduce<Sum<float>, NORMALIZATION_THREADS>(sum) / float(element_count);
  }
  let variance = workgroup_reduce<Sum<float>, NORMALIZATION_THREADS>(squares) / float(element_count) - mean * mean;
  let rms_inv = rsqrt(variance + epsilon);

  for (uint base = 0; base < element_count; base += NORMALIZATION_THREADS) {
    let i = base + invocation;
    let active = i < element_count;
    var value = 0.0;
    if (active) {
      let x = residual_add ? shortcut[row + i].toFloat() : in_place_load(input, output, row + i, in_place);
      value = (x - mean) * rms_inv;
      if (has_scales) {
        let scale = scales[affine_row + i].toFloat() + scale_offset;
        value = full_layer ? value * scale : round_to<OutputT>(round_to<OutputT>(value) * round_to<OutputT>(scale));
      }
      // The input Hadamard transform rotates the unrounded values.
      if (!use_hadamard) {
        value = round_to<OutputT>(value);
      }
      if (has_biases) {
        value += biases[affine_row + i].toFloat();
      }
    }
    if (use_hadamard) {
      let factored = active ? value * float(hadamard_factors[i]) : 0.0;
      value = hadamard_transform<NORMALIZATION_THREADS>(factored, hadamard_size, hadamard_size);
    }
    if (active) {
      output[row + i] = OutputT.fromFloat(value);
    }
  }
}
