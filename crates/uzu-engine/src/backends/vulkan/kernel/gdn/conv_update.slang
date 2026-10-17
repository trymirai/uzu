import definitions;
import common.activation;
import common.storage_address;
import generated.activation_type;

// The CPU DeltaNetConvUpdate of one token in place. Channel c accumulates in FP32 from the bias (or +0) over taps k
// below kernel_size - 1 of state[c, k] conv_weight[c, k], then x conv_weight[c, kernel_size - 1] with x the widened
// in_out[c], each product and sum an IEEE FP32 operation never fused (exact_product, ordered_sum), and stores SiLU in
// FP32 as T once. Its state row then shifts down one tap in ascending order and takes x last. One invocation owns a
// channel and reads all of its taps before writing; rows are disjoint only when state_stride holds kernel_size - 1
// taps, which the second precondition requires for several channels, and the first reproduces the CPU's own failures
// (kernel_size - 1 underflows for 0, and so does channel 0's last tap for 1). Element offsets are 64-bit words.
// Workgroups stride over channel blocks by the actual NumWorkgroups, which the host clamps to the device limit,
// advancing only while more than one step remains.

static const uint DELTA_NET_CONV_UPDATE_THREADS = 64;

[[Kernel]] [[Public]]
[[Precondition("encode", "kernel_size >= 2 || kernel_size == 1 && conv_dim == 0")]]
[[Precondition("encode", "conv_dim <= 1 || state_stride >= kernel_size - 1")]]
func DeltaNetConvUpdate<T: IModelFloat>(
  Ptr<float, Access.Read> conv_weight,
  [[Optional("has_bias")]] Ptr<float, Access.Read> bias,
  Ptr<T, Access.ReadWrite> in_out,
  Ptr<float, Access.ReadWrite> state,
  uint kernel_size,
  uint conv_dim,
  uint state_stride,
  [[Groups("conv_dim.div_ceil(64).min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[0])")]] uint channel_group,
  [[Threads("64")]] uint thread_idx,
  [[Specialize]] bool has_bias,
) {
  let groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
  let channel_groups = conv_dim / DELTA_NET_CONV_UPDATE_THREADS + (conv_dim % DELTA_NET_CONV_UPDATE_THREADS != 0 ? 1u : 0u);
  var block = channel_group;
  while (block < channel_groups) {
    let channel = block * DELTA_NET_CONV_UPDATE_THREADS + thread_idx;
    if (channel < conv_dim) {
      let taps = kernel_size - 1;
      let x = element_pointer(in_out, uint2(channel, 0))[0].toFloat();
      var acc = 0.0;
      if (has_bias) {
        acc = element_pointer(bias, uint2(channel, 0))[0];
      }
      for (uint tap = 0; tap < taps; tap++) {
        let sample = element_pointer(state, element_index(channel, state_stride, tap))[0];
        acc = ordered_sum(acc, exact_product(sample, element_pointer(conv_weight, element_index(channel, kernel_size, tap))[0]));
      }
      acc = ordered_sum(acc, exact_product(x, element_pointer(conv_weight, element_index(channel, kernel_size, taps))[0]));
      element_pointer(in_out, uint2(channel, 0))[0] = T.fromFloat(activate(acc, ActivationType.SILU));
      for (uint tap = 1; tap < taps; tap++) {
        element_pointer(state, element_index(channel, state_stride, tap - 1))[0] =
          element_pointer(state, element_index(channel, state_stride, tap))[0];
      }
      element_pointer(state, element_index(channel, state_stride, taps - 1))[0] = x;
    }
    if (channel_groups - block <= groups.x) {
      break;
    }
    block += groups.x;
  }
}
