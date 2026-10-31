import definitions;
import common.activation;
import generated.activation_type;

// In place on in_proj, one invocation per (token tile, channel c < conv_dim): its window starts as the tile's packed rows
// (see DeltaNetConvPack), so tiles are independent. Each token of the tile accumulates in FP32 from the bias (or +0) over
// the window taps then the token, stores SiLU as T once and slides the token into the window; the last tile writes the
// window to state_out[c, :]. Columns past conv_dim are untouched.

static const uint DELTA_NET_CONV_SCAN_THREADS = 64;
static const uint DELTA_NET_CONV_SCAN_TAPS = 3;

[[Variants("T", "float, bf16")]]
[[Kernel]] [[Public]]
[[Precondition("new", "KERNEL_SIZE == 4")]]
[[Precondition("encode", "suffix_len <= 1 || conv_dim <= out_stride")]]
func DeltaNetConvScan<T: IModelFloat>(
  Ptr<float, Access.Read> packed,
  Ptr<float, Access.Read> conv_weight,
  [[Optional("has_bias")]] Ptr<float, Access.Read> bias,
  Ptr<T, Access.ReadWrite> in_proj,
  Ptr<float, Access.ReadWrite> state_out,
  uint suffix_len,
  uint tile_tokens,
  uint conv_dim,
  uint out_stride,
  [[Groups("conv_dim.div_ceil(64)")]] uint channel_group,
  [[Groups("suffix_len.div_ceil(tile_tokens)")]] uint tile,
  [[Threads("64")]] uint thread_idx,
  [[Specialize]] uint KERNEL_SIZE,
  [[Specialize]] bool has_bias,
) {
  let channel = channel_group * DELTA_NET_CONV_SCAN_THREADS + thread_idx;
  if (channel >= conv_dim) {
    return;
  }
  float weights[DELTA_NET_CONV_SCAN_TAPS + 1];
  float window[DELTA_NET_CONV_SCAN_TAPS];
  [ForceUnroll]
  for (uint tap = 0; tap < DELTA_NET_CONV_SCAN_TAPS; tap++) {
    weights[tap] = conv_weight[channel * KERNEL_SIZE + tap];
    window[tap] = packed[(tile * DELTA_NET_CONV_SCAN_TAPS + tap) * conv_dim + channel];
  }
  weights[DELTA_NET_CONV_SCAN_TAPS] = conv_weight[channel * KERNEL_SIZE + DELTA_NET_CONV_SCAN_TAPS];
  var bias_value = 0.0;
  if (has_bias) {
    bias_value = bias[channel];
  }

  let tile_begin = tile * tile_tokens;
  let tile_end = min(tile_begin + tile_tokens, suffix_len);
  for (uint token = tile_begin; token < tile_end; token++) {
    let index = token * out_stride + channel;
    let sample = in_proj[index].toFloat();
    var acc = bias_value;
    [ForceUnroll]
    for (uint tap = 0; tap < DELTA_NET_CONV_SCAN_TAPS; tap++) {
      acc = fma(weights[tap], window[tap], acc);
    }
    acc = fma(weights[DELTA_NET_CONV_SCAN_TAPS], sample, acc);
    in_proj[index] = T.fromFloat(activate(acc, ActivationType.SILU));
    [ForceUnroll]
    for (uint tap = 0; tap + 1 < DELTA_NET_CONV_SCAN_TAPS; tap++) {
      window[tap] = window[tap + 1];
    }
    window[DELTA_NET_CONV_SCAN_TAPS - 1] = sample;
  }

  if (tile_end == suffix_len) {
    [ForceUnroll]
    for (uint tap = 0; tap < DELTA_NET_CONV_SCAN_TAPS; tap++) {
      state_out[channel * DELTA_NET_CONV_SCAN_TAPS + tap] = window[tap];
    }
  }
}
