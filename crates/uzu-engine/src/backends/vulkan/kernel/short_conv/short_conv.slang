import definitions;

T short_conv_step<T: IStorageFloat, WeightT: IStorageFloat, let SourceAccess : Access>(
  Ptr<T, Access.Read> in_proj,
  Ptr<WeightT, Access.Read> w,
  float bias,
  Ptr<T, SourceAccess> source,
  uint source_offset,
  Ptr<T, Access.ReadWrite> destination,
  uint destination_offset,
  uint in_proj_idx,
  uint channel_idx,
  uint kernel_size,
  uint model_dim,
) {
  let tap_count = kernel_size > 0 ? kernel_size - 1 : 0;
  let w_row = channel_idx * kernel_size;
  let x = in_proj[in_proj_idx + 2 * model_dim].toFloat() * in_proj[in_proj_idx].toFloat();
  float acc = bias;
  for (uint tap = 0; tap < tap_count; ++tap) {
    acc += w[w_row + tap].toFloat() * source[source_offset + tap].toFloat();
  }
  acc += w[w_row + tap_count].toFloat() * x;
  let post_gate = in_proj[in_proj_idx + model_dim].toFloat();
  if (tap_count > 0) {
    for (uint tap = 0; tap + 1 < tap_count; ++tap) {
      destination[destination_offset + tap] = source[source_offset + tap + 1];
    }
    destination[destination_offset + tap_count - 1] = T.fromFloat(x);
  }
  return T.fromFloat(acc * post_gate);
}

[[Variants("T", "float, half, bf16")]]
[[Kernel]] [[Public]] func ShortConvPack<T: IStorageFloat>(
  Ptr<T, Access.Read> state_in,
  Ptr<T, Access.Read> in_proj,
  Ptr<T, Access.ReadWrite> padded,
  uint state_stride,
  uint suffix_len,
  uint in_proj_stride,
  uint model_dim,
  [[Axis("model_dim", "256")]] uint channel_idx,
  [[Axis("state_stride + suffix_len", "1")]] uint row_idx,
) {
  let padded_offset = row_idx * model_dim + channel_idx;
  if (row_idx < state_stride) {
    padded[padded_offset] = state_in[channel_idx * state_stride + row_idx];
  } else {
    let in_proj_idx = (row_idx - state_stride) * in_proj_stride + channel_idx;
    padded[padded_offset] = T.fromFloat(in_proj[in_proj_idx + 2 * model_dim].toFloat() * in_proj[in_proj_idx].toFloat());
  }
}

[[Variants("T", "float, half, bf16")]]
[[Variants("WeightT", "float")]]
[[Kernel]] [[Public]] func ShortConvPrefill<T: IStorageFloat, WeightT: IStorageFloat>(
  Ptr<T, Access.Read> padded,
  Ptr<T, Access.Read> in_proj,
  Ptr<WeightT, Access.Read> w,
  [[Optional("has_bias")]] Ptr<WeightT, Access.Read> b,
  Ptr<T, Access.ReadWrite> out,
  Ptr<T, Access.ReadWrite> state_out,
  uint suffix_len,
  uint kernel_size,
  uint in_proj_stride,
  uint state_stride,
  uint model_dim,
  [[Axis("model_dim", "64")]] uint channel_idx,
  [[Axis("suffix_len + kernel_size.saturating_sub(1)", "4")]] uint row_idx,
  [[Specialize]] bool has_bias,
) {
  if (row_idx >= suffix_len) {
    state_out[channel_idx * state_stride + row_idx - suffix_len] = padded[row_idx * model_dim + channel_idx];
    return;
  }
  float acc = 0.0;
  if (has_bias) {
    acc = b[channel_idx].toFloat();
  }
  for (uint tap = 0; tap < kernel_size; ++tap) {
    acc += w[channel_idx * kernel_size + tap].toFloat() * padded[(row_idx + tap) * model_dim + channel_idx].toFloat();
  }
  let post_gate = in_proj[row_idx * in_proj_stride + channel_idx + model_dim].toFloat();
  out[row_idx * model_dim + channel_idx] = T.fromFloat(acc * post_gate);
}

[[Variants("T", "float, half, bf16")]]
[[Variants("WeightT", "float")]]
[[Kernel]] [[Public]] func ShortConvDecode<T: IStorageFloat, WeightT: IStorageFloat>(
  Ptr<T, Access.Read> in_proj,
  Ptr<WeightT, Access.Read> w,
  [[Optional("has_bias")]] Ptr<WeightT, Access.Read> b,
  [[Optional("!state_in_place")]] Ptr<T, Access.Read> state,
  Ptr<T, Access.ReadWrite> out,
  Ptr<T, Access.ReadWrite> next_state,
  uint suffix_len,
  uint kernel_size,
  uint in_proj_stride,
  uint state_stride,
  uint model_dim,
  [[Axis("model_dim", "64")]] uint channel_idx,
  [[Specialize]] bool has_bias,
  [[Specialize]] bool state_in_place,
) {
  float bias = 0.0;
  if (has_bias) {
    bias = b[channel_idx].toFloat();
  }
  let state_offset = channel_idx * state_stride;
  for (uint token_idx = 0; token_idx < suffix_len; ++token_idx) {
    let in_proj_idx = token_idx * in_proj_stride + channel_idx;
    T value;
    if (!state_in_place && token_idx == 0) {
      value = short_conv_step(
        in_proj, w, bias, state, state_offset, next_state, state_offset, in_proj_idx, channel_idx, kernel_size, model_dim
      );
    } else {
      value = short_conv_step(
        in_proj, w, bias, next_state, state_offset, next_state, state_offset, in_proj_idx, channel_idx, kernel_size,
        model_dim
      );
    }
    out[token_idx * model_dim + channel_idx] = value;
  }
}

[[Variants("T", "float, half, bf16")]]
[[Variants("WeightT", "float")]]
[[Kernel]] [[Public]] func ShortConvTrie<T: IStorageFloat, WeightT: IStorageFloat>(
  Ptr<T, Access.Read> in_proj,
  Ptr<WeightT, Access.Read> w,
  [[Optional("has_bias")]] Ptr<WeightT, Access.Read> b,
  Ptr<T, Access.Read> base_state,
  Ptr<int, Access.Read> parents,
  Ptr<T, Access.ReadWrite> out,
  Ptr<T, Access.ReadWrite> suffix_state,
  uint suffix_len,
  uint kernel_size,
  uint in_proj_stride,
  uint state_stride,
  uint model_dim,
  [[Axis("model_dim", "64")]] uint channel_idx,
  [[Specialize]] bool has_bias,
) {
  float bias = 0.0;
  if (has_bias) {
    bias = b[channel_idx].toFloat();
  }
  for (uint node = 0; node < suffix_len; ++node) {
    let parent = parents[node];
    let in_proj_idx = node * in_proj_stride + channel_idx;
    let node_offset = (node * model_dim + channel_idx) * state_stride;
    T value;
    if (parent < 0) {
      value = short_conv_step(
        in_proj, w, bias, base_state, channel_idx * state_stride, suffix_state, node_offset, in_proj_idx, channel_idx,
        kernel_size, model_dim
      );
    } else {
      value = short_conv_step(
        in_proj, w, bias, suffix_state, (uint(parent) * model_dim + channel_idx) * state_stride, suffix_state,
        node_offset, in_proj_idx, channel_idx, kernel_size, model_dim
      );
    }
    out[node * model_dim + channel_idx] = value;
  }
}
