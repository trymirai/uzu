import definitions;
import common.activation;
import common.activation_quantization;
import common.hadamard;
import generated.activation_type;
import generated.gated_act_mul;

// The gate activation times the value, one workgroup per tile of a row, in the CPU's staging: the clipped gate, its
// activation, the clipped value and their product are each rounded to T, then the input random Hadamard transform and
// quantization run in FP32. Every invocation reaches every barrier: tiles past the row hold neutral zeros. Quantization
// needs the transform, and its groups powers of two up to the tile, which the shared reducer covers.
[[Kernel]] [[Public]]
[[Precondition("new", "ops == GatedActMulOp::FullPrecision || use_hadamard")]]
[[Precondition("new", "ops == GatedActMulOp::FullPrecision || activation_scale_group_size.is_power_of_two() && activation_scale_group_size <= 128")]]
[[Precondition("new", "ops != GatedActMulOp::QuantizeWithGroupSums || sum_group_size.is_power_of_two() && sum_group_size <= 128")]]
[[Precondition("encode", "!use_hadamard || gated_dim.is_multiple_of(32)")]]
[[Precondition("encode", "ops == GatedActMulOp::FullPrecision || gated_dim.is_multiple_of(activation_scale_group_size)")]]
[[Precondition("encode", "ops != GatedActMulOp::QuantizeWithGroupSums || gated_dim.is_multiple_of(sum_group_size)")]]
[[Precondition("encode", "!clip_gate || gate_clip_min <= gate_clip_max")]]
[[Precondition("encode", "!clip_value || value_clip_min <= value_clip_max")]]
func GatedActMul<T: IModelFloat>(
  Ptr<T, Access.Read> act_operand,
  [[Optional("!interleaved")]] Ptr<T, Access.Read> value_operand,
  [[Optional("ops == GatedActMulOp::FullPrecision")]] Ptr<T, Access.ReadWrite> fp_out,
  [[Optional("ops == GatedActMulOp::Quantize || ops == GatedActMulOp::QuantizeWithGroupSums")]]
  Ptr<int8_t, Access.ReadWrite> q_out,
  [[Optional("ops == GatedActMulOp::Quantize || ops == GatedActMulOp::QuantizeWithGroupSums")]]
  Ptr<float, Access.ReadWrite> scales_out,
  [[Optional("ops == GatedActMulOp::QuantizeWithGroupSums")]] Ptr<int, Access.ReadWrite> group_sums_out,
  [[Optional("use_hadamard")]] Ptr<int, Access.Read> hadamard_factors,
  uint gated_dim,
  uint batch_dim,
  uint value_offset,
  uint value_row_stride,
  [[PipelineVariants]] ActivationType act_type,
  [[Optional("custom_activation_alpha")]] float activation_alpha,
  [[Optional("clip_gate")]] float gate_clip_min,
  [[Optional("clip_gate")]] float gate_clip_max,
  [[Optional("clip_value")]] float value_clip_min,
  [[Optional("clip_value")]] float value_clip_max,
  [[Groups("gated_dim.div_ceil(128)")]] uint tile,
  [[Groups("batch_dim")]] uint batch,
  [[Threads("128")]] uint tid,
  [[Specialize]] GatedActMulOp ops,
  [[Specialize]] bool grouped_by_weight_nibble,
  [[Specialize]] bool interleaved,
  [[Specialize]] bool use_hadamard,
  [[Specialize]] uint activation_scale_group_size,
  [[Specialize]] uint sum_group_size,
  [[Specialize]] bool custom_activation_alpha,
  [[Specialize]] bool clip_gate,
  [[Specialize]] bool clip_value,
) {
  static_assert(ACTIVATION_QUANTIZATION_TILE == 128, "one tile per workgroup");
  let column = tile * ACTIVATION_QUANTIZATION_TILE + tid;
  let active = column < gated_dim;
  float result = 0.0;
  if (active) {
    T value;
    T gate;
    if (interleaved) {
      let base = batch * 2 * gated_dim;
      value = act_operand[base + column];
      gate = act_operand[base + gated_dim + column];
    } else {
      value = value_operand[batch * value_row_stride + value_offset + column];
      gate = act_operand[batch * gated_dim + column];
    }
    // Clamping as the CPU's f32::clamp, which keeps NaN.
    if (clip_gate) {
      let x = gate.toFloat();
      gate = T.fromFloat(x < gate_clip_min ? gate_clip_min : x > gate_clip_max ? gate_clip_max : x);
    }
    // A custom alpha applies only to SiLU.
    T activated;
    if (custom_activation_alpha && act_type == ActivationType.SILU) {
      activated = T.fromFloat(activate_silu_alpha<true>(gate.toFloat(), activation_alpha));
    } else {
      activated = T.fromFloat(activate(gate.toFloat(), act_type));
    }
    if (clip_value) {
      let x = value.toFloat();
      value = T.fromFloat(x < value_clip_min ? value_clip_min : x > value_clip_max ? value_clip_max : x);
    }
    result = T.fromFloat(exact_product(value.toFloat(), activated.toFloat())).toFloat();
    if (use_hadamard) {
      result = hadamard_factor(result, hadamard_factors[column]);
    }
  }
  if (use_hadamard) {
    result = hadamard_transform<ACTIVATION_QUANTIZATION_TILE>(result, tid);
  }
  if (ops == GatedActMulOp.FullPrecision) {
    if (active) {
      fp_out[batch * gated_dim + column] = T.fromFloat(result);
    }
  } else {
    quantize_activation(
      result,
      tid,
      batch,
      column,
      gated_dim,
      q_out,
      scales_out,
      group_sums_out,
      activation_scale_group_size,
      sum_group_size,
      ops == GatedActMulOp.QuantizeWithGroupSums,
      grouped_by_weight_nibble
    );
  }
}
