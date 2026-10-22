#include <metal_stdlib>
#include "../activation/activations.h"
#include "../common/dsl.h"

using namespace metal;

// Causal conv1d + SiLU, in place on in_proj. A thread owns CHANNELS channels of one token tile; its first window
// comes from DeltaNetConvPack, so tiles are independent. The last tile writes state_out.
//
// packed:      [num_tiles, KERNEL_SIZE - 1, conv_dim]
// conv_weight: [conv_dim, KERNEL_SIZE]
// bias:        [conv_dim] (optional)
// in_proj:     [suffix_len, out_stride], first conv_dim channels overwritten
// state_out:   [conv_dim, KERNEL_SIZE - 1]
// conv_dim and out_stride must be multiples of CHANNELS.

constant constexpr uint CHANNELS = 8;
constant constexpr uint VECTOR_WIDTH = 4;
constant constexpr uint ROWS_IN_FLIGHT = 4;

template <typename T>
METAL_FUNC void load_channels(const device T* source, thread float (&values)[CHANNELS]) {
  const device vec<T, VECTOR_WIDTH>* vectors = reinterpret_cast<const device vec<T, VECTOR_WIDTH>*>(source);
  for (uint vector_idx = 0; vector_idx < CHANNELS / VECTOR_WIDTH; ++vector_idx) {
    const vec<float, VECTOR_WIDTH> chunk = vec<float, VECTOR_WIDTH>(vectors[vector_idx]);
    for (uint i = 0; i < VECTOR_WIDTH; ++i) {
      values[vector_idx * VECTOR_WIDTH + i] = chunk[i];
    }
  }
}

template <typename T>
METAL_FUNC void store_channels(device T* destination, const thread float (&values)[CHANNELS]) {
  device vec<T, VECTOR_WIDTH>* vectors = reinterpret_cast<device vec<T, VECTOR_WIDTH>*>(destination);
  for (uint vector_idx = 0; vector_idx < CHANNELS / VECTOR_WIDTH; ++vector_idx) {
    vec<float, VECTOR_WIDTH> chunk;
    for (uint i = 0; i < VECTOR_WIDTH; ++i) {
      chunk[i] = values[vector_idx * VECTOR_WIDTH + i];
    }
    vectors[vector_idx] = vec<T, VECTOR_WIDTH>(chunk);
  }
}

template <typename T, uint KERNEL_SIZE>
VARIANTS(T, float, bfloat)
VARIANTS(KERNEL_SIZE, 4)
PUBLIC KERNEL(DeltaNetConvScan)(
    device const float* packed,
    device const float* conv_weight,
    device const float* bias OPTIONAL(has_bias),
    device T* in_proj,
    device float* state_out,
    constant const uint& suffix_len,
    constant const uint& tile_tokens,
    constant const uint& conv_dim,
    constant const uint& out_stride,
    const bool has_bias SPECIALIZE,
    const uint channel_group AXIS(conv_dim / 8, 32),
    const uint tile_idx AXIS(suffix_len.div_ceil(tile_tokens), 4)
) {
  constexpr uint TAPS = KERNEL_SIZE - 1;
  const uint first_channel = channel_group * CHANNELS;

  float weights[CHANNELS * KERNEL_SIZE];
  const device float4* weight_vectors =
      reinterpret_cast<const device float4*>(conv_weight + first_channel * KERNEL_SIZE);
  for (uint i = 0; i < CHANNELS * KERNEL_SIZE / 4; ++i) {
    const float4 chunk = weight_vectors[i];
    for (uint lane = 0; lane < 4; ++lane) {
      weights[i * 4 + lane] = chunk[lane];
    }
  }
  float biases[CHANNELS];
  for (uint j = 0; j < CHANNELS; ++j) {
    biases[j] = has_bias ? bias[first_channel + j] : 0.0f;
  }

  float window[TAPS][CHANNELS];
  for (uint tap = 0; tap < TAPS; ++tap) {
    load_channels(packed + (tile_idx * TAPS + tap) * conv_dim + first_channel, window[tap]);
  }

  const uint tile_begin = tile_idx * tile_tokens;
  const uint tile_end = min(tile_begin + tile_tokens, suffix_len);
  for (uint token = tile_begin; token < tile_end; token += ROWS_IN_FLIGHT) {
    float rows[ROWS_IN_FLIGHT][CHANNELS];
    for (uint r = 0; r < ROWS_IN_FLIGHT; ++r) {
      if (token + r < tile_end) {
        load_channels(in_proj + (token + r) * size_t(out_stride) + first_channel, rows[r]);
      }
    }
    for (uint r = 0; r < ROWS_IN_FLIGHT; ++r) {
      if (token + r < tile_end) {
        float outputs[CHANNELS];
        for (uint j = 0; j < CHANNELS; ++j) {
          float acc = biases[j];
          for (uint tap = 0; tap < TAPS; ++tap) {
            acc = fma(weights[j * KERNEL_SIZE + tap], window[tap][j], acc);
          }
          acc = fma(weights[j * KERNEL_SIZE + TAPS], rows[r][j], acc);
          outputs[j] = activate_silu(acc);
        }
        store_channels(in_proj + (token + r) * size_t(out_stride) + first_channel, outputs);
        for (uint tap = 0; tap + 1 < TAPS; ++tap) {
          for (uint j = 0; j < CHANNELS; ++j) {
            window[tap][j] = window[tap + 1][j];
          }
        }
        for (uint j = 0; j < CHANNELS; ++j) {
          window[TAPS - 1][j] = rows[r][j];
        }
      }
    }
  }

  if (tile_end == suffix_len) {
    device float4* state_vectors = reinterpret_cast<device float4*>(state_out + first_channel * TAPS);
    for (uint i = 0; i < CHANNELS * TAPS / 4; ++i) {
      float4 chunk;
      for (uint lane = 0; lane < 4; ++lane) {
        const uint flat = i * 4 + lane;
        chunk[lane] = window[flat % TAPS][flat / TAPS];
      }
      state_vectors[i] = chunk;
    }
  }
}
