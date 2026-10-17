import definitions;
import common.activation_quantization;
import common.hadamard;
import generated.activation_transform;

// Random Hadamard transform of each row, one workgroup per tile of a row: the input transform multiplies by the factors
// before the FP32 transform, the output transform after it. The full-precision result is rounded to T, then a bias is
// added in FP32 and the sum rounded to T. Every invocation reaches every barrier: tiles past the row hold neutral zeros.
// Quantization writes no full-precision output to transform in place, and its groups are powers of two up to the
// tile, which the shared reducer covers; without sums the CPU still divides rows by the sum group.
[[Kernel]] [[Public]]
[[Precondition("new", "!in_place || ops == ActivationTransformOp::InputRht || ops == ActivationTransformOp::OutputRht")]]
[[Precondition("new", "ops == ActivationTransformOp::InputRht || ops == ActivationTransformOp::OutputRht || activation_scale_group_size.is_power_of_two() && activation_scale_group_size <= 128")]]
[[Precondition("new", "ops != ActivationTransformOp::QuantizeWithGroupSums || sum_group_size.is_power_of_two() && sum_group_size <= 128")]]
[[Precondition("new", "ops != ActivationTransformOp::Quantize || sum_group_size > 0")]]
[[Precondition("encode", "element_count.is_multiple_of(32)")]]
[[Precondition("encode", "ops == ActivationTransformOp::InputRht || ops == ActivationTransformOp::OutputRht || element_count.is_multiple_of(activation_scale_group_size)")]]
[[Precondition("encode", "ops != ActivationTransformOp::QuantizeWithGroupSums || element_count.is_multiple_of(sum_group_size)")]]
func ActivationTransform<T: IModelFloat, BiasT: IModelFloat>(
  [[Optional("!in_place")]] Ptr<T, Access.Read> input,
  [[Optional("ops == ActivationTransformOp::InputRht || ops == ActivationTransformOp::OutputRht")]]
  Ptr<T, Access.ReadWrite> fp_out,
  [[Optional("has_bias")]] Ptr<BiasT, Access.Read> bias,
  [[Optional("ops == ActivationTransformOp::Quantize || ops == ActivationTransformOp::QuantizeWithGroupSums")]]
  Ptr<int8_t, Access.ReadWrite> q_out,
  [[Optional("ops == ActivationTransformOp::Quantize || ops == ActivationTransformOp::QuantizeWithGroupSums")]]
  Ptr<float, Access.ReadWrite> scales_out,
  [[Optional("ops == ActivationTransformOp::QuantizeWithGroupSums")]] Ptr<int, Access.ReadWrite> group_sums_out,
  Ptr<int, Access.Read> rht_factors,
  uint batch_size,
  uint element_count,
  [[Groups("element_count.div_ceil(128)")]] uint tile,
  [[Groups("batch_size")]] uint batch,
  [[Threads("128")]] uint tid,
  [[Specialize]] ActivationTransformOp ops,
  [[Specialize]] bool grouped_by_weight_nibble,
  [[Specialize]] bool in_place,
  [[Specialize]] uint activation_scale_group_size,
  [[Specialize]] uint sum_group_size,
  [[Specialize]] bool has_bias,
) {
  static_assert(ACTIVATION_QUANTIZATION_TILE == 128, "one tile per workgroup");
  let column = tile * ACTIVATION_QUANTIZATION_TILE + tid;
  let active = column < element_count;
  let index = batch * element_count + column;
  let input_rht = ops != ActivationTransformOp.OutputRht;
  float value = 0.0;
  int factor = 0;
  if (active) {
    // In place, each invocation reads only the element it writes.
    if (in_place) {
      value = fp_out[index].toFloat();
    } else {
      value = input[index].toFloat();
    }
    factor = rht_factors[column];
    if (input_rht) {
      value = hadamard_factor(value, factor);
    }
  }
  value = hadamard_transform<ACTIVATION_QUANTIZATION_TILE>(value, tid);
  if (active && !input_rht) {
    value = hadamard_factor(value, factor);
  }
  if (ops == ActivationTransformOp.InputRht || ops == ActivationTransformOp.OutputRht) {
    if (active) {
      if (has_bias) {
        // Evaluated as written (NoContraction): the device otherwise fuses the transform's final product into the
        // addition, skipping the rounding to T before the bias.
        let rounded = T.fromFloat(value).toFloat();
        let addend = bias[column].toFloat();
        value = spirv_asm {
          %sum:$$float = OpFAdd $rounded $addend;
          OpDecorate %sum NoContraction;
          result:$$float = OpCopyObject %sum;
        };
      }
      fp_out[index] = T.fromFloat(value);
    }
  } else {
    quantize_activation(
      value,
      tid,
      batch,
      column,
      element_count,
      q_out,
      scales_out,
      group_sums_out,
      activation_scale_group_size,
      sum_group_size,
      ops == ActivationTransformOp.QuantizeWithGroupSums,
      grouped_by_weight_nibble
    );
  }
}
