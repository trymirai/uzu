import definitions;
import storage;
import generated.activation_type;
import generated.gated_act_mul;

groupshared float h[256];

[[Variants("T", "float, half, BFloat16")]]
[[Public]] [[Kernel]] func GatedActMul<T : ITensorElement>(
    StructuredBuffer<uint> act_operand,
    [[Optional("!interleaved")]] StructuredBuffer<uint> value_operand,
    [[Optional(
        "!(ops == GatedActMulOp::Quantize || ops == GatedActMulOp::QuantizeWithGroupSums)"
    )]] RWStructuredBuffer<Atomic<uint>> fp_out,
    [[Optional(
        "(ops == GatedActMulOp::Quantize || ops == GatedActMulOp::QuantizeWithGroupSums)"
    )]] RWStructuredBuffer<int8_t> q_out,
    [[Optional(
        "(ops == GatedActMulOp::Quantize || ops == GatedActMulOp::QuantizeWithGroupSums)"
    )]] RWStructuredBuffer<float> scales_out,
    [[Optional("(ops == GatedActMulOp::QuantizeWithGroupSums)")]] RWStructuredBuffer<int> group_sums_out,
    [[Optional("use_hadamard")]] StructuredBuffer<int> hadamard_factors,
    uint gated_dim,
    uint batch_dim,
    uint value_offset,
    uint value_row_stride,
    ActivationType act_type,
    [[Optional("custom_activation_alpha")]] float activation_alpha,
    [[Optional("clip_gate")]] float gate_clip_min,
    [[Optional("clip_gate")]] float gate_clip_max,
    [[Optional("clip_value")]] float value_clip_min,
    [[Optional("clip_value")]] float value_clip_max,
    [[Specialize]] GatedActMulOp ops,
    [[Specialize]] bool grouped_by_weight_nibble,
    [[Specialize]] bool interleaved,
    [[Specialize]] bool use_hadamard,
    [[Specialize]] uint activation_scale_group_size,
    [[Specialize]] uint sum_group_size,
    [[Specialize]] bool custom_activation_alpha,
    [[Specialize]] bool clip_gate,
    [[Specialize]] bool clip_value,
    [[Groups("gated_dim.div_ceil(256)")]] uint activation_tile_index,
    [[Groups("batch_dim")]] uint batch_idx,
    [[Threads("256")]] uint thread_index,
    ThreadContext thread_context,
) {
  uint lane = thread_index;
  uint3 gid = uint3(activation_tile_index, batch_idx, 0);

  let col = gid.x * 256u + lane;
  let row = gid.y;
  var v = 0.0;
  if (col < gated_dim) {
    float gate;
    float value;
    if (interleaved) {
      gate = tensorLoad<T>(act_operand, row * 2u * gated_dim + gated_dim + col);
      value = tensorLoad<T>(act_operand, row * 2u * gated_dim + col);
    } else {
      gate = tensorLoad<T>(act_operand, row * gated_dim + col);
      value = tensorLoad<T>(value_operand, row * value_row_stride + value_offset + col);
    }
    if (clip_gate) {
      gate = tensorRound<T>(clamp(gate, gate_clip_min, gate_clip_max));
    }
    if (clip_value) {
      value = tensorRound<T>(clamp(value, value_clip_min, value_clip_max));
    }
    var alpha = 1.0;
    if (custom_activation_alpha) {
      alpha = activation_alpha;
    }
    v = tensorRound<T>(value * tensorRound<T>(activate(gate, uint(act_type), alpha)));
    if (use_hadamard) {
      v *= hadamard_factors[col];
    }
  }
  if (use_hadamard) {
    h[lane] = v;
    GroupMemoryBarrierWithGroupSync();
    for (var stride = 1u; stride < 32u; stride *= 2u) {
      let own = h[lane];
      let paired = h[lane ^ stride];
      GroupMemoryBarrierWithGroupSync();
      h[lane] = choose(own + paired, paired - own, (lane & stride) != 0u);
      GroupMemoryBarrierWithGroupSync();
    }
    v = h[lane] * 0.1767766952966369;
  }
  if (col < gated_dim) {
    tensorStore<T>(fp_out, row * gated_dim + col, v);
  }
}
