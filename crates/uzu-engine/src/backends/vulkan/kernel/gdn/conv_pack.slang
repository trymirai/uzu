import definitions;
import common.storage_address;

// The CPU conv1d pack. Padded rows hold the state_stride state samples of each channel, then the suffix tokens, at
// row_stride; a state row holds a channel's samples, oldest first. State rows copy stored bits; token rows convert
// through FP32. Element offsets are 64-bit words, and a pointer to a row is formed only where that row is written.
// Workgroups stride over their dimension by the actual NumWorkgroups, which the host clamps to the device limits,
// advancing only while more than one step remains.

static const uint CONV1D_PACK_THREADS = 256;

// The workgroup count of a dimension, rounded up.
uint conv1d_groups(uint count, uint threads) {
  return count / threads + (count % threads != 0 ? 1u : 0u);
}

[[Kernel]] [[Public]] func Conv1dPack<StateT: IModelFloat, InputT: IModelFloat>(
  Ptr<StateT, Access.Read> state_in,
  Ptr<InputT, Access.Read> x,
  Ptr<StateT, Access.ReadWrite> padded,
  uint state_stride,
  uint row_stride,
  uint suffix_len,
  uint num_channels,
  [[Groups("num_channels.div_ceil(256).min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[0])")]] uint channel_group,
  [[Groups("state_stride.max(suffix_len).min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[1])")]] uint row_group,
  [[Threads("256")]] uint thread_idx,
) {
  let groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
  let channel_groups = conv1d_groups(num_channels, CONV1D_PACK_THREADS);
  var group = channel_group;
  while (group < channel_groups) {
    let channel = group * CONV1D_PACK_THREADS + thread_idx;
    if (channel < num_channels) {
      // The state rows copy stored bits.
      var row = row_group;
      while (row < state_stride) {
        element_pointer(padded, element_index(row, row_stride, channel))[0] =
          element_pointer(state_in, element_index(channel, state_stride, row))[0];
        if (state_stride - row <= groups.y) {
          break;
        }
        row += groups.y;
      }
      // The token rows, from row state_stride on, convert through FP32.
      var token = row_group;
      if (token < suffix_len) {
        let token_rows = element_pointer(padded, element_index(state_stride, row_stride, 0));
        while (true) {
          let input = element_pointer(x, element_index(token, row_stride, channel))[0];
          element_pointer(token_rows, element_index(token, row_stride, channel))[0] = StateT.fromFloat(input.toFloat());
          if (suffix_len - token <= groups.y) {
            break;
          }
          token += groups.y;
        }
      }
    }
    if (channel_groups - group <= groups.x) {
      break;
    }
    group += groups.x;
  }
}
