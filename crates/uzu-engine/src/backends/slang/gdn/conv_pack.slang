import definitions;
import storage;

[[Variants("StateT", "float")]]
[[Variants("InputT", "float, half, BFloat16")]]
[[Public]] [[Kernel]] func Conv1dPack<
    StateT : ITensorElement,
    InputT : ITensorElement>(
    StructuredBuffer<uint> state_in,
    StructuredBuffer<uint> x,
    RWStructuredBuffer<Atomic<uint>> padded,
    uint state_stride,
    uint row_stride,
    uint suffix_len,
    uint num_channels,
    [[Axis("num_channels", "32")]] uint channel_idx,
    [[Axis("state_stride + suffix_len", "1")]] uint row_idx,
) {
  uint3 gid = uint3(row_idx * num_channels + channel_idx, 0, 0);

  let index = gid.x;
  if (index >= (state_stride + suffix_len) * num_channels) {
    return;
  }
  let channel = index % num_channels;
  let row = index / num_channels;
  var value = 0.0;
  if (row < state_stride) {
    value = tensorLoad<StateT>(state_in, channel * state_stride + row);
  } else {
    value = tensorLoad<InputT>(x, (row - state_stride) * row_stride + channel);
  }
  tensorStore<StateT>(padded, row * row_stride + channel, value);
}
