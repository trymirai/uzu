import definitions;
import common.activation_quantization;

// The production quantizer on given FP32 rows, so tests reach quotients no transform produces, like nextdown(1/2).
[[Kernel]] [[Test]] func TestActivationQuantization(
  Ptr<float, Access.Read> values,
  Ptr<int8_t, Access.ReadWrite> q_out,
  Ptr<float, Access.ReadWrite> scales_out,
  [[Optional("emit_group_sums")]] Ptr<int, Access.ReadWrite> group_sums_out,
  uint batch_size,
  uint element_count,
  [[Groups("element_count.div_ceil(128)")]] uint tile,
  [[Groups("batch_size")]] uint batch,
  [[Threads("128")]] uint tid,
  [[Specialize]] bool grouped_by_weight_nibble,
  [[Specialize]] uint activation_scale_group_size,
  [[Specialize]] uint sum_group_size,
  [[Specialize]] bool emit_group_sums,
) {
  let column = tile * ACTIVATION_QUANTIZATION_TILE + tid;
  float value = 0.0;
  if (column < element_count) {
    value = values[batch * element_count + column];
  }
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
    emit_group_sums,
    grouped_by_weight_nibble
  );
}
