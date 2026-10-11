import definitions;
import common.activation;
import common.storage_address;
import generated.activation_type;

// The CPU DeltaNetConvScan. Output (t, c), for t < suffix_len and c < conv_dim, accumulates in FP32 from the bias (or
// +0) over taps k ascending of conv_weight[c, k] conv_padded[t + k, c], each product and sum an IEEE FP32 operation
// never fused (exact_product, ordered_sum), then SiLU in FP32 and one T store into in_proj[t, c] at out_stride; the
// columns past conv_dim are never touched. Every channel also copies the state_stride padded rows from suffix_len on
// into state_out[c, :] as raw FP32, also without tokens. Element offsets are 64-bit words formed one term at a time, so
// no row sum t + k or suffix_len + k is formed in 32 bits. Workgroups stride over channel blocks and rows by the actual
// NumWorkgroups, which the host clamps to the device limits, advancing only while more than one step remains. Tokens
// write disjoint rows only when out_stride holds conv_dim, which the precondition requires whenever several exist.

static const uint DELTA_NET_CONV_SCAN_THREADS = 64;

[[Kernel]] [[Public]]
[[Precondition("encode", "suffix_len <= 1 || conv_dim <= out_stride")]]
func DeltaNetConvScan<T: IModelFloat>(
  Ptr<float, Access.Read> conv_padded,
  Ptr<float, Access.Read> conv_weight,
  [[Optional("has_bias")]] Ptr<float, Access.Read> bias,
  Ptr<T, Access.ReadWrite> in_proj,
  Ptr<float, Access.ReadWrite> state_out,
  uint suffix_len,
  uint kernel_size,
  uint row_stride,
  uint state_stride,
  uint conv_dim,
  uint out_stride,
  [[Groups("if conv_dim == 0 || suffix_len.max(state_stride) == 0 { 0 } else { conv_dim.div_ceil(64).min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[0]) }")]] uint channel_group,
  [[Groups("suffix_len.max(state_stride).min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[1])")]] uint row_group,
  [[Threads("64")]] uint thread_idx,
  [[Specialize]] bool has_bias,
) {
  let groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
  let channel_groups = conv_dim / DELTA_NET_CONV_SCAN_THREADS + (conv_dim % DELTA_NET_CONV_SCAN_THREADS != 0 ? 1u : 0u);
  let rows = max(suffix_len, state_stride);
  var block = channel_group;
  while (block < channel_groups) {
    let channel = block * DELTA_NET_CONV_SCAN_THREADS + thread_idx;
    if (channel < conv_dim) {
      var row = row_group;
      while (row < rows) {
        if (row < suffix_len) {
          var acc = 0.0;
          if (has_bias) {
            acc = element_pointer(bias, uint2(channel, 0))[0];
          }
          if (kernel_size > 0) {
            let window = element_pointer(conv_padded, element_index(row, row_stride, 0));
            for (uint tap = 0; tap < kernel_size; tap++) {
              let sample = element_pointer(window, element_index(tap, row_stride, channel))[0];
              acc = ordered_sum(acc, exact_product(element_pointer(conv_weight, element_index(channel, kernel_size, tap))[0], sample));
            }
          }
          element_pointer(in_proj, element_index(row, out_stride, channel))[0] = T.fromFloat(activate(acc, ActivationType.SILU));
        }
        if (row < state_stride) {
          let history = element_pointer(conv_padded, element_index(suffix_len, row_stride, 0));
          element_pointer(state_out, element_index(channel, state_stride, row))[0] =
            element_pointer(history, element_index(row, row_stride, channel))[0];
        }
        if (rows - row <= groups.y) {
          break;
        }
        row += groups.y;
      }
    }
    if (channel_groups - block <= groups.x) {
      break;
    }
    block += groups.x;
  }
}
