import definitions;

// An in_proj row holds each channel's pre-gate, post-gate and input at offsets 0, model_dim and 2 model_dim; a state row
// holds one channel's samples, oldest first. Convolutions accumulate in FP32. Prefill convolves the samples Pack stored
// in T; Decode and Trie convolve the current gated input in FP32 and round it to T only to store it. State copies and
// shifts move stored T values bit for bit.

static const uint SHORT_CONV_PREFILL_THREADS = 64;
static const uint SHORT_CONV_PREFILL_ROWS = 4;
static const uint SHORT_CONV_RECURRENT_THREADS = 64;

// acc + a · b with both roundings of the CPU kernels (NoContraction): drivers may otherwise fuse them into one FMA.
float short_conv_mad(float acc, float a, float b) {
  return spirv_asm {
    %product:$$float = OpFMul $a $b;
    OpDecorate %product NoContraction;
    %sum:$$float = OpFAdd $acc %product;
    OpDecorate %sum NoContraction;
    result:$$float = OpCopyObject %sum;
  };
}

// One token of one channel for Decode and Trie: convolves the tap_count samples at `source` and the current gated input,
// then stores the shifted samples and T(x) at `destination`, which may be the same samples. Returns the gated output.
// Pointers are indexed and never offset or cast: Slang lowers both through Int64, which the M2 driver does not expose.
T short_conv_step<T: IStorageFloat, WeightT: IFloat32, let SourceAccess : Access>(
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
    acc = short_conv_mad(acc, w[w_row + tap].toFloat(), source[source_offset + tap].toFloat());
  }
  acc = short_conv_mad(acc, w[w_row + tap_count].toFloat(), x);
  let post_gate = in_proj[in_proj_idx + model_dim].toFloat();
  if (tap_count > 0) {
    for (uint tap = 0; tap + 1 < tap_count; ++tap) {
      destination[destination_offset + tap] = source[source_offset + tap + 1];
    }
    destination[destination_offset + tap_count - 1] = T.fromFloat(x);
  }
  return T.fromFloat(acc * post_gate);
}

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

// Rows below suffix_len are outputs, convolving padded rows row..row + kernel_size - 1; the kernel_size - 1 rows after
// them copy padded rows suffix_len.. into each channel's first state elements, also for an empty suffix.
[[Kernel]] [[Public]] func ShortConvPrefill<T: IStorageFloat, WeightT: IFloat32>(
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
  [[Groups("model_dim.div_ceil(64)")]] uint channel_group,
  [[Groups("(suffix_len + kernel_size.saturating_sub(1)).div_ceil(4)")]] uint row_group,
  [[Threads("64")]] uint thread_idx,
  [[Threads("4")]] uint row_thread,
  [[Specialize]] bool has_bias,
) {
  let channel_idx = channel_group * SHORT_CONV_PREFILL_THREADS + thread_idx;
  let row_idx = row_group * SHORT_CONV_PREFILL_ROWS + row_thread;
  if (channel_idx >= model_dim || row_idx >= suffix_len + (kernel_size > 0 ? kernel_size - 1 : 0)) {
    return;
  }
  if (row_idx >= suffix_len) {
    state_out[channel_idx * state_stride + row_idx - suffix_len] = padded[row_idx * model_dim + channel_idx];
    return;
  }
  float acc = 0.0;
  if (has_bias) {
    acc = b[channel_idx].toFloat();
  }
  for (uint tap = 0; tap < kernel_size; ++tap) {
    acc = short_conv_mad(acc, w[channel_idx * kernel_size + tap].toFloat(), padded[(row_idx + tap) * model_dim + channel_idx].toFloat());
  }
  let post_gate = in_proj[row_idx * in_proj_stride + channel_idx + model_dim].toFloat();
  out[row_idx * model_dim + channel_idx] = T.fromFloat(acc * post_gate);
}

// Tokens run in order; each one after the first continues the state the previous one stored.
[[Kernel]] [[Public]] func ShortConvDecode<T: IStorageFloat, WeightT: IFloat32>(
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
  [[Groups("if suffix_len == 0 { 0 } else { model_dim.div_ceil(64) }")]] uint channel_group,
  [[Threads("64")]] uint thread_idx,
  [[Specialize]] bool has_bias,
  [[Specialize]] bool state_in_place,
) {
  let channel_idx = channel_group * SHORT_CONV_RECURRENT_THREADS + thread_idx;
  if (channel_idx >= model_dim) {
    return;
  }
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

// Nodes run in order; a negative parent continues base_state, any other an earlier node's suffix_state.
[[Kernel]] [[Public]] func ShortConvTrie<T: IStorageFloat, WeightT: IFloat32>(
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
  [[Groups("if suffix_len == 0 { 0 } else { model_dim.div_ceil(64) }")]] uint channel_group,
  [[Threads("64")]] uint thread_idx,
  [[Specialize]] bool has_bias,
) {
  let channel_idx = channel_group * SHORT_CONV_RECURRENT_THREADS + thread_idx;
  if (channel_idx >= model_dim) {
    return;
  }
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
