import definitions;
import common.activation_quantization;
import common.hadamard;
import generated.activation_transform;

[[Variants("T", "float, bf16")]]
[[Variants("BiasT", "float, bf16")]]
[[Kernel]] [[Public]]
[[Precondition("new", "ops != ActivationTransformOp::PackHalf")]]
[[Precondition("new", "ops == ActivationTransformOp::InputRht || ops == ActivationTransformOp::OutputRht || activation_scale_group_size.is_power_of_two() && activation_scale_group_size <= 128")]]
[[Precondition("new", "ops != ActivationTransformOp::QuantizeWithGroupSums || sum_group_size.is_power_of_two() && sum_group_size <= 128")]]
func ActivationTransform<T: IStorageFloat, BiasT: IStorageFloat>(
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
  [[Optional("ops == ActivationTransformOp::PackHalf")]] Ptr<uint, Access.ReadWrite> packed_half_out,
  [[Groups("element_count.div_ceil(128)")]] uint tile,
  [[Groups("batch_size")]] uint batch,
  [[Threads("128")]] uint tid,
  [[Specialize]] ActivationTransformOp ops,
  [[Specialize]] bool grouped_by_weight_nibble,
  [[Specialize]] bool in_place,
  [[Specialize]] uint activation_scale_group_size,
  [[Specialize]] uint sum_group_size,
  [[Specialize]] uint weight_bits,
  [[Specialize]] bool has_bias,
) {
  static_assert(ACTIVATION_QUANTIZATION_TILE == 128, "one tile per workgroup");
  let column = tile * ACTIVATION_QUANTIZATION_TILE + subgroup_invocation();
  let active = column < element_count;
  let index = batch * element_count + column;
  let input_rht = ops != ActivationTransformOp.OutputRht;
  float value = 0.0;
  int factor = 0;
  if (active) {
    value = in_place_load(input, fp_out, index, in_place);
    factor = rht_factors[column];
    if (input_rht) {
      value = value * float(factor);
    }
  }
  value = hadamard_transform<ACTIVATION_QUANTIZATION_TILE>(value);
  if (active && !input_rht) {
    value = value * float(factor);
  }
  if (ops == ActivationTransformOp.InputRht || ops == ActivationTransformOp.OutputRht) {
    if (active) {
      if (has_bias) {
        value = T.fromFloat(value).toFloat() + bias[column].toFloat();
      }
      fp_out[index] = T.fromFloat(value);
    }
  } else {
    quantize_activation(
      value,
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

[[Variants("T", "float, bf16")]]
[[Kernel]]
[[Precondition("new", "activation_scale_group_size.is_power_of_two() && activation_scale_group_size <= 128")]]
[[Precondition("new", "sum_group_size == 0 || sum_group_size.is_power_of_two() && sum_group_size <= 128")]]
[[Precondition("encode", "element_count.is_multiple_of(activation_scale_group_size) && element_count.is_multiple_of(sum_group_size.max(1))")]]
func ActivationQuantize<T: IStorageFloat>(
  Ptr<T, Access.Read> input,
  Ptr<int8_t, Access.ReadWrite> q_out,
  Ptr<float, Access.ReadWrite> scales_out,
  [[Optional("sum_group_size > 0")]] Ptr<int, Access.ReadWrite> group_sums_out,
  uint batch_size,
  uint element_count,
  [[Groups("element_count.div_ceil(128)")]] uint tile,
  [[Groups("batch_size")]] uint batch,
  [[Threads("128")]] uint tid,
  [[Specialize]] bool grouped_by_weight_nibble,
  [[Specialize]] uint activation_scale_group_size,
  [[Specialize]] uint sum_group_size,
) {
  let column = tile * ACTIVATION_QUANTIZATION_TILE + subgroup_invocation();
  var value = 0.0;
  if (column < element_count) {
    value = input[batch * element_count + column].toFloat();
  }
  quantize_activation(
    value,
    batch,
    column,
    element_count,
    q_out,
    scales_out,
    group_sums_out,
    activation_scale_group_size,
    sum_group_size,
    sum_group_size > 0,
    grouped_by_weight_nibble
  );
}
