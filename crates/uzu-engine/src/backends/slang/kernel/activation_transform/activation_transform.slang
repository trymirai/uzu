import definitions;
import storage;
import generated.activation_transform;

groupshared float h[256];

[[Variants("T", "float, half, BFloat16")]]
[[Variants("BiasT", "float, half, BFloat16")]]
[[Variants("ops", "ActivationTransformOp::InputRht, ActivationTransformOp::OutputRht")]]
[[Public]] [[Kernel]] func ActivationTransform<T : ITensorElement, BiasT : ITensorElement, let ops : ActivationTransformOp>(
    [[Optional("!in_place")]] StructuredBuffer<uint32_t> input,
    [[Optional(
        "!(ops == ActivationTransformOp::Quantize || ops == ActivationTransformOp::QuantizeWithGroupSums) && !(ops == ActivationTransformOp::PackHalf)"
    )]] RWStructuredBuffer<Atomic<uint32_t>> fp_out,
    [[Optional("has_bias")]] StructuredBuffer<uint32_t> bias,
    [[Optional(
        "(ops == ActivationTransformOp::Quantize || ops == ActivationTransformOp::QuantizeWithGroupSums)"
    )]] RWStructuredBuffer<int8_t> q_out,
    [[Optional(
        "(ops == ActivationTransformOp::Quantize || ops == ActivationTransformOp::QuantizeWithGroupSums)"
    )]] RWStructuredBuffer<float> scales_out,
    [[Optional("(ops == ActivationTransformOp::QuantizeWithGroupSums)")]] RWStructuredBuffer<int32_t> group_sums_out,
    StructuredBuffer<int32_t> rht_factors,
    uint32_t batch_size,
    uint32_t element_count,
    [[Optional("(ops == ActivationTransformOp::PackHalf)")]] RWStructuredBuffer<uint32_t> packed_half_out,
    [[Specialize]] bool grouped_by_weight_nibble,
    [[Specialize]] bool in_place,
    [[Specialize]] uint32_t activation_scale_group_size,
    [[Specialize]] uint32_t sum_group_size,
    [[Specialize]] uint32_t weight_bits,
    [[Specialize]] bool has_bias,
    [[Groups("element_count.div_ceil(256)")]] uint32_t activation_tile_index,
    [[Groups("batch_size")]] uint32_t batch_index,
    [[Threads("256")]] uint32_t thread_index,
    ThreadContext thread_context,
) {
  uint lane = thread_index;
  uint3 gid = uint3(activation_tile_index, batch_index, 0);

  let col = gid.x * 256u + lane;
  let row = gid.y;
  var v = 0.0;
  if (col < element_count) {
    v = (in_place ? tensorLoad<T>(fp_out, row * element_count + col) : tensorLoad<T>(input, row * element_count + col));
    if (ops != ActivationTransformOp::OutputRht) {
      v *= rht_factors[col];
    }
  }

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
  if (ops == ActivationTransformOp::OutputRht && col < element_count) {
    v *= rht_factors[col];
  }
  if (has_bias && col < element_count) {
    v = tensorRound<T>(v) + tensorLoad<BiasT>(bias, col);
  }
  if (col < element_count) {
    tensorStore<T>(fp_out, row * element_count + col, v);
  }
}
