import definitions;
import common.workgroup_reduction;
import generated.trie;

static const uint NGRAM_GATE_THREADS = 256;

[[Variants("T", "float, half, bf16")]]
[[Kernel]] [[Public]] func NGramGate<T: IStorageFloat>(
  Ptr<T, Access.Read> keys,
  Ptr<T, Access.Read> stream,
  Ptr<T, Access.Read> values,
  Ptr<float, Access.Read> key_scales,
  Ptr<float, Access.Read> query_scales,
  Ptr<float, Access.Read> conv_scales,
  Ptr<T, Access.ReadWrite> gated,
  Ptr<T, Access.ReadWrite> conv_input,
  uint num_tokens,
  uint stream_count,
  uint model_dim,
  float epsilon,
  float scale_offset,
  [[Groups("num_tokens * stream_count")]] uint row,
  [[Threads("256")]] uint thread_in_row,
) {
  static_assert(NGRAM_GATE_THREADS == 256, "Threads(256)");
  let token = row / stream_count;
  let stream_index = row % stream_count;
  let row_offset = row * model_dim;
  let value_offset = token * model_dim;
  let scale_row_offset = stream_index * model_dim;
  let dim = float(model_dim);

  float key_square_sum = 0.0;
  float query_square_sum = 0.0;
  float scaled_dot = 0.0;
  float value_square_sum = 0.0;
  for (uint i = thread_in_row; i < model_dim; i += NGRAM_GATE_THREADS) {
    let key = keys[row_offset + i].toFloat();
    let query = stream[row_offset + i].toFloat();
    let value = values[value_offset + i].toFloat();
    let key_scale = key_scales[scale_row_offset + i] + scale_offset;
    let query_scale = query_scales[scale_row_offset + i] + scale_offset;
    key_square_sum += key * key;
    query_square_sum += query * query;
    scaled_dot += key * key_scale * query * query_scale;
    value_square_sum += value * value;
  }

  let sums = workgroup_reduce<VectorSum<4>, NGRAM_GATE_THREADS>(
    float4(key_square_sum, query_square_sum, scaled_dot, value_square_sum)
  );

  let key_rms_inv = rsqrt(sums.x / dim + epsilon);
  let query_rms_inv = rsqrt(sums.y / dim + epsilon);
  let logit = key_rms_inv * query_rms_inv * sums.z / sqrt(dim);
  let signed_root = float(sign(logit)) * sqrt(max(abs(logit), 1e-6));
  let gate = 1.0 / (1.0 + exp(-signed_root));
  let conv_rms_inv = rsqrt(gate * gate * sums.w / dim + epsilon);

  for (uint i = thread_in_row; i < model_dim; i += NGRAM_GATE_THREADS) {
    let gated_value = gate * values[value_offset + i].toFloat();
    let conv_scale = conv_scales[scale_row_offset + i] + scale_offset;
    gated[row_offset + i] = T.fromFloat(gated_value);
    conv_input[row_offset + i] = T.fromFloat(gated_value * conv_rms_inv * conv_scale);
  }
}

float ngram_conv_sample<T: IStorageFloat>(
  Ptr<T, Access.Read> conv_input,
  Ptr<T, Access.Read> state,
  uint padded_row,
  uint state_rows,
  uint channels,
  uint channel,
) {
  if (padded_row < state_rows) {
    return state[padded_row * channels + channel].toFloat();
  }
  return conv_input[(padded_row - state_rows) * channels + channel].toFloat();
}

[[Variants("T", "float, half, bf16")]]
[[Kernel]] [[Public]] func NGramConvInject<T: IStorageFloat>(
  Ptr<T, Access.Read> conv_input,
  Ptr<T, Access.Read> state,
  Ptr<float, Access.Read> weights,
  Ptr<T, Access.Read> gated,
  Ptr<T, Access.ReadWrite> stream,
  Ptr<T, Access.ReadWrite> next_state,
  uint num_tokens,
  uint channels,
  uint kernel_size,
  [[Axis("channels", "256")]] uint channel,
  [[Axis("num_tokens + kernel_size - 1", "1")]] uint row,
) {
  let state_rows = kernel_size - 1;

  if (row < num_tokens) {
    float accumulator = 0.0;
    for (uint tap = 0; tap < kernel_size; ++tap) {
      accumulator += weights[channel * kernel_size + tap] *
                     ngram_conv_sample(conv_input, state, row + tap, state_rows, channels, channel);
    }
    let activated = accumulator / (1.0 + exp(-accumulator));
    let index = row * channels + channel;
    stream[index] = T.fromFloat(stream[index].toFloat() + gated[index].toFloat() + activated);
  } else {
    let state_row = row - num_tokens;
    next_state[state_row * channels + channel] =
        T.fromFloat(ngram_conv_sample(conv_input, state, num_tokens + state_row, state_rows, channels, channel));
  }
}

// `state` and `node_states` have different access modes, so they can't be unified into one
// pointer. Branch rather than use `?:`, which may evaluate both reads, and the unused one can
// be out of bounds.
T ngram_parent_state<T: IStorageFloat>(
  Ptr<T, Access.Read> state,
  Ptr<T, Access.ReadWrite> node_states,
  int parent,
  uint row,
  uint state_rows,
  uint channels,
  uint channel,
) {
  if (parent < 0) {
    return state[row * channels + channel];
  }
  return node_states[(uint(parent) * state_rows + row) * channels + channel];
}

// One thread per channel walking the nodes in order: a child's slot is only ever read by the
// thread that wrote its parent's, so DFS order alone sequences the dependency.
[[Variants("T", "float, half, bf16")]]
[[Kernel]] [[Public]] func NGramConvInjectTrie<T: IStorageFloat>(
  Ptr<T, Access.Read> conv_input,
  Ptr<T, Access.Read> state,
  Ptr<TrieNode, Access.Read> trie,
  Ptr<float, Access.Read> weights,
  Ptr<T, Access.Read> gated,
  Ptr<T, Access.ReadWrite> stream,
  Ptr<T, Access.ReadWrite> node_states,
  uint num_tokens,
  uint channels,
  uint kernel_size,
  [[Axis("channels", "256")]] uint channel,
) {
  let state_rows = kernel_size - 1;
  let weight_row = channel * kernel_size;

  for (uint node = 0; node < num_tokens; ++node) {
    let parent = trie[node].parent;
    let own = conv_input[node * channels + channel].toFloat();
    let own_offset = node * state_rows * channels;

    float accumulator = weights[weight_row + state_rows] * own;
    for (uint tap = 0; tap < state_rows; ++tap) {
      let value = ngram_parent_state(state, node_states, parent, tap, state_rows, channels, channel);
      accumulator += weights[weight_row + tap] * value.toFloat();
    }
    let activated = accumulator / (1.0 + exp(-accumulator));
    let index = node * channels + channel;
    stream[index] = T.fromFloat(stream[index].toFloat() + gated[index].toFloat() + activated);

    for (uint row = 0; row + 1 < state_rows; ++row) {
      node_states[own_offset + row * channels + channel] =
          ngram_parent_state(state, node_states, parent, row + 1, state_rows, channels, channel);
    }
    if (state_rows > 0) {
      node_states[own_offset + (state_rows - 1) * channels + channel] = T.fromFloat(own);
    }
  }
}
