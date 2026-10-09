import definitions;
import common.activation;
import common.storage_address;
import generated.activation_type;

// The CPU conv1d kernels of Mamba2. Padded rows hold the state_stride state samples of each channel, then the suffix
// tokens, at row_stride; a state row holds a channel's samples, oldest first. Convolutions accumulate in FP32 from the
// bias (or +0) over the taps in order, each product and sum rounded as the CPU does (exact_product, ordered_sum), and
// the accumulator is rounded to T before the activation. Raw state copies move stored bits; a sample that passes
// through FP32 (Pack's tokens, Decode's new tail, Scan's state) is converted. Element offsets are 64-bit words, and a
// pointer to a row is formed only where that row is read or written. Workgroups stride over their dimension by the
// actual NumWorkgroups, which the host clamps to the device limits, advancing only while more than one step remains.

static const uint CONV1D_PACK_THREADS = 256;
static const uint CONV1D_THREADS = 64;

// T(acc), then the activation on that T as FP32, rounded to T again: the CPU's activation.activate(T::from(acc)).
T conv1d_value<T: IStorageFloat>(float acc, ActivationType activation) {
  let rounded = T.fromFloat(acc);
  if (activation == ActivationType.IDENTITY) {
    return rounded;
  }
  return T.fromFloat(activate(rounded.toFloat(), activation));
}

// Routes channel `channel` of token `token` to x_out, b_out or c_out by subtraction, so inner_dim + 2 proj_dim never
// wraps; later channels are not stored.
void conv1d_store<T: IStorageFloat>(
  Ptr<T, Access.ReadWrite> x_out,
  Ptr<T, Access.ReadWrite> b_out,
  Ptr<T, Access.ReadWrite> c_out,
  uint token,
  uint channel,
  uint inner_dim,
  uint proj_dim,
  T value,
) {
  if (channel < inner_dim) {
    element_pointer(x_out, element_index(token, inner_dim, channel))[0] = value;
  } else if (channel - inner_dim < proj_dim) {
    element_pointer(b_out, element_index(token, proj_dim, channel - inner_dim))[0] = value;
  } else if (channel - inner_dim - proj_dim < proj_dim) {
    element_pointer(c_out, element_index(token, proj_dim, channel - inner_dim - proj_dim))[0] = value;
  }
}

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

// A stored state sample: from next_state in place, where the original state is absent and never addressed, otherwise
// from the original state.
T conv1d_state<T: IStorageFloat, let IN_PLACE : bool>(
  Ptr<T, Access.Read> state,
  Ptr<T, Access.ReadWrite> next_state,
  uint2 element,
) {
  if (IN_PLACE) {
    return element_pointer(next_state, element)[0];
  }
  return element_pointer(state, element)[0];
}

// One channel of Decode: tokens in order, each convolving the state taps and its own input. In place each token reads
// the state the previous one stored; otherwise every token reads the original state and rewrites next_state.
void conv1d_decode_channel<T: IStorageFloat, let IN_PLACE : bool>(
  Ptr<T, Access.Read> x,
  Ptr<T, Access.Read> w,
  float bias,
  Ptr<T, Access.Read> state,
  Ptr<T, Access.ReadWrite> x_out,
  Ptr<T, Access.ReadWrite> b_out,
  Ptr<T, Access.ReadWrite> c_out,
  Ptr<T, Access.ReadWrite> next_state,
  uint kernel_size,
  uint row_stride,
  uint state_stride,
  uint suffix_len,
  uint inner_dim,
  uint proj_dim,
  ActivationType activation,
  uint channel,
) {
  let taps = kernel_size > 0 ? kernel_size - 1 : 0;
  for (uint token = 0; token < suffix_len; token++) {
    let current = element_pointer(x, element_index(token, row_stride, channel))[0].toFloat();
    var acc = bias;
    for (uint tap = 0; tap < taps; tap++) {
      let sample = conv1d_state<T, IN_PLACE>(state, next_state, element_index(channel, state_stride, tap));
      acc = ordered_sum(acc, exact_product(element_pointer(w, element_index(channel, kernel_size, tap))[0].toFloat(), sample.toFloat()));
    }
    if (taps < kernel_size) {
      acc = ordered_sum(acc, exact_product(element_pointer(w, element_index(channel, kernel_size, taps))[0].toFloat(), current));
    }
    conv1d_store(x_out, b_out, c_out, token, channel, inner_dim, proj_dim, conv1d_value<T>(acc, activation));
    if (taps > 0) {
      for (uint tap = 0; tap + 1 < taps; tap++) {
        let shifted = conv1d_state<T, IN_PLACE>(state, next_state, element_index(channel, state_stride, tap + 1));
        element_pointer(next_state, element_index(channel, state_stride, tap))[0] = shifted;
      }
      element_pointer(next_state, element_index(channel, state_stride, taps - 1))[0] = T.fromFloat(current);
    }
  }
}

[[Kernel]] [[Public]] func Conv1dDecode<T: IStorageFloat>(
  Ptr<T, Access.Read> x,
  Ptr<T, Access.Read> w,
  [[Optional("has_bias")]] Ptr<T, Access.Read> b,
  [[Optional("!state_in_place")]] Ptr<T, Access.Read> state,
  Ptr<T, Access.ReadWrite> x_out,
  Ptr<T, Access.ReadWrite> b_out,
  Ptr<T, Access.ReadWrite> c_out,
  Ptr<T, Access.ReadWrite> next_state,
  uint kernel_size,
  uint row_stride,
  uint state_stride,
  uint num_channels,
  uint suffix_len,
  uint inner_dim,
  uint proj_dim,
  [[PipelineVariants]] ActivationType activation_type,
  [[Groups("if suffix_len == 0 { 0 } else { num_channels.div_ceil(64).min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[0]) }")]] uint channel_group,
  [[Threads("64")]] uint thread_idx,
  [[Specialize]] bool has_bias,
  [[Specialize]] bool state_in_place,
) {
  let groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
  let channel_groups = conv1d_groups(num_channels, CONV1D_THREADS);
  var group = channel_group;
  while (group < channel_groups) {
    let channel = group * CONV1D_THREADS + thread_idx;
    if (channel < num_channels) {
      float bias = 0.0;
      if (has_bias) {
        bias = element_pointer(b, uint2(channel, 0))[0].toFloat();
      }
      if (state_in_place) {
        conv1d_decode_channel<T, true>(
          x, w, bias, state, x_out, b_out, c_out, next_state, kernel_size, row_stride, state_stride, suffix_len,
          inner_dim, proj_dim, activation_type, channel
        );
      } else {
        conv1d_decode_channel<T, false>(
          x, w, bias, state, x_out, b_out, c_out, next_state, kernel_size, row_stride, state_stride, suffix_len,
          inner_dim, proj_dim, activation_type, channel
        );
      }
    }
    if (channel_groups - group <= groups.x) {
      break;
    }
    group += groups.x;
  }
}

// Rows below suffix_len are outputs convolving padded rows token..token + kernel_size - 1; the kernel_size - 1 rows
// from suffix_len on convert into each channel's first state elements, also for an empty suffix.
[[Kernel]] [[Public]] func Conv1dScan<T: IStorageFloat>(
  Ptr<T, Access.Read> padded,
  Ptr<T, Access.Read> w,
  [[Optional("has_bias")]] Ptr<T, Access.Read> b,
  Ptr<T, Access.ReadWrite> x_out,
  Ptr<T, Access.ReadWrite> b_out,
  Ptr<T, Access.ReadWrite> c_out,
  Ptr<T, Access.ReadWrite> state_out,
  uint suffix_len,
  uint kernel_size,
  uint row_stride,
  uint state_stride,
  uint num_channels,
  uint inner_dim,
  uint proj_dim,
  [[PipelineVariants]] ActivationType activation_type,
  [[Groups("num_channels.div_ceil(64).min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[0])")]] uint channel_group,
  [[Groups("suffix_len.max(kernel_size.saturating_sub(1)).min(command_buffer.context().physical_device().properties.limits.max_compute_work_group_count[1])")]] uint row_group,
  [[Threads("64")]] uint thread_idx,
  [[Specialize]] bool has_bias,
) {
  let groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
  let channel_groups = conv1d_groups(num_channels, CONV1D_THREADS);
  let taps = kernel_size > 0 ? kernel_size - 1 : 0;
  var group = channel_group;
  while (group < channel_groups) {
    let channel = group * CONV1D_THREADS + thread_idx;
    if (channel < num_channels) {
      float bias = 0.0;
      if (has_bias) {
        bias = element_pointer(b, uint2(channel, 0))[0].toFloat();
      }
      var token = row_group;
      while (token < suffix_len) {
        var acc = bias;
        if (kernel_size > 0) {
          let window = element_pointer(padded, element_index(token, row_stride, channel));
          for (uint tap = 0; tap < kernel_size; tap++) {
            let sample = element_pointer(window, element_index(tap, row_stride, 0))[0].toFloat();
            acc = ordered_sum(acc, exact_product(element_pointer(w, element_index(channel, kernel_size, tap))[0].toFloat(), sample));
          }
        }
        conv1d_store(x_out, b_out, c_out, token, channel, inner_dim, proj_dim, conv1d_value<T>(acc, activation_type));
        if (suffix_len - token <= groups.y) {
          break;
        }
        token += groups.y;
      }
      var tap = row_group;
      if (tap < taps) {
        let state_rows = element_pointer(padded, element_index(suffix_len, row_stride, channel));
        while (true) {
          let sample = element_pointer(state_rows, element_index(tap, row_stride, 0))[0];
          element_pointer(state_out, element_index(channel, state_stride, tap))[0] = T.fromFloat(sample.toFloat());
          if (taps - tap <= groups.y) {
            break;
          }
          tap += groups.y;
        }
      }
    }
    if (channel_groups - group <= groups.x) {
      break;
    }
    group += groups.x;
  }
}
