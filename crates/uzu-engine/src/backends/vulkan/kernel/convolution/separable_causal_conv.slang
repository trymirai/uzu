import definitions;
import common.activation;
import common.storage_address;

// The CPU SeparableCausalConv, BF16 only like it. input and output are [Q, C], weights [C, K] and bias [C]. Output (t, c)
// accumulates in FP32 from the bias (or +0) over the taps j from 0 to min(K, t + 1) in order, each adding
// input[t - j, c] (weights[c, K - 1 - j] + coefficient_deltas[t R + j (C / G) + c / G]) with C / G and c / G floored as
// the CPU does, every sum and product an IEEE FP32 operation never fused (ordered_sum, exact_product), and is rounded to
// BF16 once. Elements are addressed only by taps and outputs that exist; offsets are 64-bit words formed one term at a
// time. Workgroups stride over channel blocks and tokens by the actual NumWorkgroups, which the host clamps to the device
// limits, advancing only while more than one step remains, so no index wraps for any u32 C or Q.

static const uint SEPARABLE_CAUSAL_CONV_THREADS = 64;

[[Kernel]] [[Public]]
[[Precondition("encode", "group_size > 0")]]
func SeparableCausalConv<T: IBFloat16>(
  Ptr<T, Access.Read> input,
  Ptr<T, Access.Read> coefficient_deltas,
  Ptr<T, Access.Read> weights,
  [[Optional("has_bias")]] Ptr<T, Access.Read> bias,
  Ptr<T, Access.ReadWrite> output,
  uint sequence_length,
  uint coefficient_row_stride,
  [[Groups("if sequence_length == 0 { 0 } else { model_dim.div_ceil(64).min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[0]) }")]] uint channel_group,
  [[Groups("sequence_length.min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[1])")]] uint token_group,
  [[Threads("64")]] uint thread_idx,
  [[Specialize]] uint model_dim,
  [[Specialize]] uint kernel_size,
  [[Specialize]] uint group_size,
  [[Specialize]] bool has_bias,
) {
  let groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
  let channel_groups = model_dim / SEPARABLE_CAUSAL_CONV_THREADS + (model_dim % SEPARABLE_CAUSAL_CONV_THREADS != 0 ? 1u : 0u);
  let num_groups = model_dim / group_size;
  var block = channel_group;
  while (block < channel_groups) {
    let channel = block * SEPARABLE_CAUSAL_CONV_THREADS + thread_idx;
    if (channel < model_dim) {
      let group = channel / group_size;
      float start = 0.0;
      if (has_bias) {
        start = element_pointer(bias, uint2(channel, 0))[0].toFloat();
      }
      var token = token_group;
      while (token < sequence_length) {
        var acc = start;
        let taps = min(kernel_size, token + 1);
        for (uint tap = 0; tap < taps; tap++) {
          let x = element_pointer(input, element_index(token - tap, model_dim, channel))[0].toFloat();
          let w = element_pointer(weights, element_index(channel, kernel_size, kernel_size - 1 - tap))[0].toFloat();
          let row = element_pointer(coefficient_deltas, element_index(token, coefficient_row_stride, 0));
          let d = element_pointer(row, element_index(tap, num_groups, group))[0].toFloat();
          acc = ordered_sum(acc, exact_product(x, ordered_sum(w, d)));
        }
        element_pointer(output, element_index(token, model_dim, channel))[0] = T.fromFloat(acc);
        if (sequence_length - token <= groups.y) {
          break;
        }
        token += groups.y;
      }
    }
    if (channel_groups - block <= groups.x) {
      break;
    }
    block += groups.x;
  }
}
