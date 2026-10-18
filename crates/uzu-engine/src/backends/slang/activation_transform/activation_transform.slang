import definitions;
import storage;
import generated.activation_transform;

groupshared float h[256];

[[Variants("T", "float, half, BFloat16")]]
[[Variants("BiasT", "float, half, BFloat16")]]
[[Public]] [[Kernel]] func ActivationTransform<T : ITensorElement, BiasT : ITensorElement>(
    [[Optional("!in_place")]] StructuredBuffer<uint> input,
    [[Optional(
        "!(ops == ActivationTransformOp::Quantize || ops == ActivationTransformOp::QuantizeWithGroupSums)"
    )]] RWStructuredBuffer<Atomic<uint>> fp_out,
    [[Optional("has_bias")]] StructuredBuffer<uint> bias,
    [[Optional(
        "(ops == ActivationTransformOp::Quantize || ops == ActivationTransformOp::QuantizeWithGroupSums)"
    )]] RWStructuredBuffer<int8_t> q_out,
    [[Optional(
        "(ops == ActivationTransformOp::Quantize || ops == ActivationTransformOp::QuantizeWithGroupSums)"
    )]] RWStructuredBuffer<float> scales_out,
    [[Optional("(ops == ActivationTransformOp::QuantizeWithGroupSums)")]] RWStructuredBuffer<int> group_sums_out,
    StructuredBuffer<int> rht_factors,
    uint batch_size,
    uint element_count,
    [[Specialize]] ActivationTransformOp ops,
    [[Specialize]] bool grouped_by_weight_nibble,
    [[Specialize]] bool in_place,
    [[Specialize]] uint activation_scale_group_size,
    [[Specialize]] uint sum_group_size,
    [[Specialize]] bool has_bias,
    [[Groups("element_count.div_ceil(256)")]] uint activation_tile_index,
    [[Groups("batch_size")]] uint batch_index,
    [[Threads("256")]] uint thread_index,
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
