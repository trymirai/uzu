import definitions;
import common.activation;
import common.storage_address;
import generated.activation_type;

// The CPU ConvTreeScan. Channel c of tree node `row`, below conv_dim, accumulates in FP32 from the bias (or +0) over
// history taps h ascending of sample_h conv_weight[c, K - 1 - h], each product and sum an IEEE FP32 operation never
// fused (exact_product, ordered_sum), then stores T(silu(acc)) into out_proj[row, c]. sample_0 is in_proj[row, c]; each
// later tap follows the walk src <- parents[src] while src >= 0, then src - 1, reading base_state[c, K - 1 - |src|] once
// src is negative; the first K - 1 samples are recorded raw into suffix_state[row, c, K - 2 - h]. Channels from conv_dim
// on copy their T bits unchanged. The CPU's read of parents after the last tap is not performed. Walks read only the
// inputs, so rows need no ordering; out_proj must not alias in_proj. The data contract keeps every parent -1 or a row
// index and every walk within K - 1 steps of the base state. Element offsets are 64-bit words formed one term at a time.
// Workgroups stride over channel blocks and rows by the actual NumWorkgroups, which the host clamps to the device
// limits, advancing only while more than one step remains.

static const uint CONV_TREE_SCAN_THREADS = 64;

[[Kernel]] [[Public]]
[[Precondition("new", "kernel_size >= 1")]]
func ConvTreeScan<T: IModelFloat>(
  Ptr<T, Access.Read> in_proj,
  Ptr<float, Access.Read> conv_weight,
  [[Optional("has_bias")]] Ptr<float, Access.Read> bias,
  Ptr<float, Access.Read> base_state,
  Ptr<int, Access.Read> parents,
  Ptr<T, Access.ReadWrite> out_proj,
  Ptr<float, Access.ReadWrite> suffix_state,
  uint suffix_len,
  uint total_proj_dim,
  uint conv_dim,
  [[Groups("if suffix_len == 0 || total_proj_dim == 0 { 0 } else { total_proj_dim.div_ceil(64).min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[0]) }")]] uint channel_group,
  [[Groups("suffix_len.min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[1])")]] uint row_group,
  [[Threads("64")]] uint thread_idx,
  [[Specialize]] uint kernel_size,
  [[Specialize]] bool has_bias,
) {
  let groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
  let blocks = total_proj_dim / CONV_TREE_SCAN_THREADS + (total_proj_dim % CONV_TREE_SCAN_THREADS != 0 ? 1u : 0u);
  let taps = kernel_size - 1;
  var block = channel_group;
  while (block < blocks) {
    if (thread_idx < total_proj_dim - block * CONV_TREE_SCAN_THREADS) {
      let channel = block * CONV_TREE_SCAN_THREADS + thread_idx;
      var row = row_group;
      while (row < suffix_len) {
        let output = element_pointer(out_proj, element_index(row, total_proj_dim, channel));
        if (channel >= conv_dim) {
          output[0] = element_pointer(in_proj, element_index(row, total_proj_dim, channel))[0];
        } else {
          var acc = 0.0;
          if (has_bias) {
            acc = element_pointer(bias, uint2(channel, 0))[0];
          }
          let state_row = element_pointer(suffix_state, byte_offset(element_index(row, conv_dim, channel), taps));
          var source = int(row);
          for (uint h = 0; h < kernel_size; h++) {
            var sample = 0.0;
            if (source >= 0) {
              sample = element_pointer(in_proj, element_index(uint(source), total_proj_dim, channel))[0].toFloat();
            } else {
              sample = element_pointer(base_state, element_index(channel, taps, taps - uint(-source)))[0];
            }
            let weight = element_pointer(conv_weight, element_index(channel, kernel_size, taps - h))[0];
            acc = ordered_sum(acc, exact_product(sample, weight));
            if (h < taps) {
              element_pointer(state_row, uint2(taps - 1 - h, 0))[0] = sample;
              source = source >= 0 ? element_pointer(parents, uint2(uint(source), 0))[0] : source - 1;
            }
          }
          output[0] = T.fromFloat(activate(acc, ActivationType.SILU));
        }
        if (suffix_len - row <= groups.y) {
          break;
        }
        row += groups.y;
      }
    }
    if (blocks - block <= groups.x) {
      break;
    }
    block += groups.x;
  }
}
