import definitions;
import storage;

groupshared float sum_values[256];
groupshared float sum_squares[256];

[[Variants("InputT", "float, half, BFloat16")]]
[[Variants("AffineT", "float, half")]]
[[Variants("OutputT", "float, half, BFloat16")]]
[[Variants("AccumT", "float")]]
[[Public]] [[Kernel]] func Normalization<InputT : ITensorElement, AffineT : ITensorElement, OutputT : ITensorElement, AccumT : ITensorElement>(
    [[Optional("!in_place")]] StructuredBuffer<uint> input,
    [[Optional("has_scales")]] StructuredBuffer<uint> scales,
    [[Optional("has_biases")]] StructuredBuffer<uint> biases,
    RWStructuredBuffer<Atomic<uint>> output,
    [[Optional("copy_to_shortcut")]] RWStructuredBuffer<Atomic<uint>> shortcut,
    [[Optional("use_hadamard")]] StructuredBuffer<int> hadamard_factors,
    uint batch_size,
    uint element_count,
    float epsilon,
    float scale_offset,
    float post_layer_scalar,
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
    ThreadContext thread_context,
    [[Groups("batch_size")]] uint batch_idx,
    [[Threads("256")]] uint thread_in_row,
) {
  uint lane = thread_in_row;
  uint3 gid = uint3(batch_idx, 0, 0);

  let base = gid.x * element_count;
  var sum = 0.0;
  var sum_sq = 0.0;
  for (var i = lane; i < element_count; i += 256u) {
    var v = (in_place ? tensorLoad<OutputT>(output, base + i) : tensorLoad<InputT>(input, base + i));
    if (copy_to_shortcut) {
      if (residual_add) {
        v = tensorRound<InputT>(v + tensorLoad<InputT>(shortcut, base + i));
        if (scale_residual_sum) {
          v = tensorRound<InputT>(v * post_layer_scalar);
        }
      }
      tensorStore<InputT>(shortcut, base + i, v);
    }
    sum += v;
    sum_sq += v * v;
  }
  sum_values[lane] = sum;
  sum_squares[lane] = sum_sq;
  GroupMemoryBarrierWithGroupSync();
  DeviceMemoryBarrierWithGroupSync();
  for (var stride = 128u; stride > 0u; stride /= 2u) {
    if (lane < stride) {
      sum_values[lane] += sum_values[lane + stride];
      sum_squares[lane] += sum_squares[lane + stride];
    }
    GroupMemoryBarrierWithGroupSync();
  }
  var mean = 0.0;
  if (subtract_mean) {
    mean = sum_values[0] / float(element_count);
  }
  let rms = rsqrt(sum_squares[0] / float(element_count) - mean * mean + epsilon);
  GroupMemoryBarrierWithGroupSync();
  for (var tile = 0u; tile < element_count; tile += 256u) {
    let i = tile + lane;
    var result = 0.0;
    if (i < element_count) {
      var v = (in_place ? tensorLoad<OutputT>(output, base + i) : tensorLoad<InputT>(input, base + i));
      if (residual_add) {
        v = tensorLoad<InputT>(shortcut, base + i);
      }
      result = (v - mean) * rms;
      if (has_scales) {
        let scale = tensorLoad<AffineT>(scales, i) + scale_offset;
        if (full_layer) {
          result *= scale;
        } else {
          result = tensorRound<OutputT>(result) * tensorRound<OutputT>(scale);
        }
      }
      result = tensorRound<OutputT>(result);
      if (has_biases) {
        result = tensorRound<OutputT>(result + tensorLoad<AffineT>(biases, i));
      }
      if (use_hadamard) {
        result *= hadamard_factors[i];
      }
    }
    if (use_hadamard) {
      sum_values[lane] = result;
      GroupMemoryBarrierWithGroupSync();
      for (var stride = 1u; stride < 32u; stride *= 2u) {
        let paired = sum_values[lane ^ stride];
        let own = sum_values[lane];
        GroupMemoryBarrierWithGroupSync();
        sum_values[lane] = choose(own + paired, paired - own, (lane & stride) != 0u);
        GroupMemoryBarrierWithGroupSync();
      }
      result = tensorRound<OutputT>(sum_values[lane] * 0.1767766952966369);
    }
    if (scale_output) {
      result *= tensorRound<OutputT>(post_layer_scalar);
    }
    if (i < element_count) {
      tensorStore<OutputT>(output, base + i, result);
    }
    GroupMemoryBarrierWithGroupSync();
  }
}
