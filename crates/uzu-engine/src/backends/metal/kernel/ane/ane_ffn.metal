// GPU side of the ANE prefill FFN (Metal-only kernels; see backends/metal/ane).
//
// Layouts: activations are token-major [tokens, hidden] in uzu and channel-major
// [channels, row_stride] in ANE surfaces (tokens contiguous within a channel row).

#include <metal_stdlib>
#include "../common/dsl.h"
#include "../common/thread_context.h"
#include "../common/threadgroup_reduce.h"
#include "../hadamard_transform/hadamard_transform.h"

using namespace metal;

#define ANE_THREADS 256
#define ANE_CHUNK 4096
#define ANE_TILE 32

// Register-resident Walsh-Hadamard transforms over a 4096-value chunk held by a 256-thread
// threadgroup: thread t owns values [16 t, 16 t + 16), so simdgroup s owns [512 s, 512 s + 512).
// Stages of span 1..8 run within a thread, 16..256 across lanes (simd shuffles), 512..2048 across
// simdgroups (one threadgroup-memory exchange).
#define ANE_VALUES 16

static void hadamard_in_thread(thread float* values) {
  METAL_PRAGMA_UNROLL
  for (ushort span = 1; span < ANE_VALUES; span <<= 1) {
    METAL_PRAGMA_UNROLL
    for (ushort index = 0; index < ANE_VALUES; ++index) {
      if (!(index & span)) {
        const float left = values[index];
        const float right = values[index + span];
        values[index] = left + right;
        values[index + span] = left - right;
      }
    }
  }
}

// Spans 16 * lane_stride for lane_stride in [1, lane_strides).
static void hadamard_across_lanes(thread float* values, ushort lane, ushort lane_strides) {
  for (ushort stride = 1; stride < lane_strides; stride <<= 1) {
    METAL_PRAGMA_UNROLL
    for (ushort index = 0; index < ANE_VALUES; ++index) {
      const float partner = simd_shuffle_xor(values[index], stride);
      values[index] = (lane & stride) ? (partner - values[index]) : (partner + values[index]);
    }
  }
}

// Spans 512, 1024 and 2048: an 8-point transform across the simdgroups, through `shared`.
static void hadamard_across_simdgroups(thread float* values, threadgroup float* shared, uint thread_id) {
  METAL_PRAGMA_UNROLL
  for (ushort index = 0; index < ANE_VALUES; ++index) {
    shared[thread_id * ANE_VALUES + index] = values[index];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint position = thread_id; position < 512; position += ANE_THREADS) {
    float column[8];
    METAL_PRAGMA_UNROLL
    for (ushort group = 0; group < 8; ++group) {
      column[group] = shared[group * 512 + position];
    }
    METAL_PRAGMA_UNROLL
    for (ushort span = 1; span < 8; span <<= 1) {
      METAL_PRAGMA_UNROLL
      for (ushort group = 0; group < 8; ++group) {
        if (!(group & span)) {
          const float left = column[group];
          const float right = column[group + span];
          column[group] = left + right;
          column[group + span] = left - right;
        }
      }
    }
    METAL_PRAGMA_UNROLL
    for (ushort group = 0; group < 8; ++group) {
      shared[group * 512 + position] = column[group];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  METAL_PRAGMA_UNROLL
  for (ushort index = 0; index < ANE_VALUES; ++index) {
    values[index] = shared[thread_id * ANE_VALUES + index];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Normalized transform of every consecutive `block` (32, 64, ..., 4096) of the thread's chunk values.
static void block_hadamard(thread float* values, uint block, ushort lane, threadgroup float* shared, uint thread_id) {
  hadamard_in_thread(values);
  hadamard_across_lanes(values, lane, ushort(min(block, 512u) / ANE_VALUES));
  if (block > 512) {
    hadamard_across_simdgroups(values, shared, thread_id);
  }
  const float scale = rsqrt(float(block));
  METAL_PRAGMA_UNROLL
  for (ushort index = 0; index < ANE_VALUES; ++index) {
    values[index] *= scale;
  }
}

static float threadgroup_peak(float value, threadgroup float* partials, const thread ThreadContext& thread_context) {
  return threadgroup_cooperative_reduce<SimdReduceMax<float>, ANE_THREADS>(value, partials, thread_context);
}

static float values_peak(const thread float* values) {
  float peak = 0.0f;
  METAL_PRAGMA_UNROLL
  for (ushort index = 0; index < ANE_VALUES; ++index) {
    peak = max(peak, fabs(values[index]));
  }
  return peak;
}

static void store_codes(device char* destination, const thread float* values, float inverse) {
  METAL_PRAGMA_UNROLL
  for (ushort index = 0; index < ANE_VALUES; index += 4) {
    *reinterpret_cast<device packed_char4*>(destination + index) = char4(clamp(
        rint(float4(values[index], values[index + 1], values[index + 2], values[index + 3]) * inverse),
        -127.0f,
        127.0f
    ));
  }
}

// Per token: a = H_4096(signs * x), int8 codes with one scale per token (hidden must be 4096).
// Writes token-major codes (transposed into the ANE surface by AneTransposeCodes) and the
// token scale (x128, the ANE program's dequantize convention) into the `tx` surface.
KERNEL(AnePack)(
    const device bfloat* input,
    const device float* signs,
    device char* codes,
    device half* token_scales,
    constant uint& hidden,
    constant uint& tokens,
    threadgroup float shared[ANE_CHUNK],
    threadgroup float partials[METAL_SIMD_SIZE],
    const ThreadContext thread_context,
    const uint token GROUPS(tokens),
    const uint thread_id THREADS(ANE_THREADS)
) {
  const uint base = thread_id * ANE_VALUES;
  const device bfloat* row = input + ulong(token) * hidden + base;
  float values[ANE_VALUES];
  METAL_PRAGMA_UNROLL
  for (ushort index = 0; index < ANE_VALUES; ++index) {
    values[index] = float(row[index]) * signs[base + index];
  }
  block_hadamard(values, hidden, thread_context.simd_lane_id, shared, thread_id);
  const float peak = max(threadgroup_peak(values_peak(values), partials, thread_context), 1e-20f);
  store_codes(codes + ulong(token) * hidden + base, values, 127.0f / peak);
  if (thread_id == 0) {
    token_scales[token] = half(peak / 127.0f * 128.0f);
  }
}

// Token-major int8 [tokens, channels] -> channel-major surface rows [channels, row_stride].
KERNEL(AneTransposeCodes)(
    const device char* codes,
    device char* surface,
    constant uint& channels,
    constant uint& tokens,
    constant uint& row_stride,
    threadgroup char tile[ANE_TILE * (ANE_TILE + 1)],
    const uint channel_tile GROUPS(channels.div_ceil(32)),
    const uint token_tile GROUPS(tokens.div_ceil(32)),
    const uint thread_id THREADS(ANE_THREADS)
) {
  const uint channel_base = channel_tile * ANE_TILE;
  const uint token_base = token_tile * ANE_TILE;
  for (uint index = thread_id; index < ANE_TILE * ANE_TILE; index += ANE_THREADS) {
    const uint token = index / ANE_TILE;
    const uint channel = index % ANE_TILE;
    char value = 0;
    if (token_base + token < tokens && channel_base + channel < channels) {
      value = codes[ulong(token_base + token) * channels + channel_base + channel];
    }
    tile[channel * (ANE_TILE + 1) + token] = value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint index = thread_id; index < ANE_TILE * ANE_TILE; index += ANE_THREADS) {
    const uint channel = index / ANE_TILE;
    const uint token = index % ANE_TILE;
    if (token_base + token < tokens && channel_base + channel < channels) {
      surface[ulong(channel_base + channel) * row_stride + token_base + token] = tile[channel * (ANE_TILE + 1) + token];
    }
  }
}

// ANE output [hidden + 1, row_stride] fp16 (last row: per-token intermediate scale) -> token-major bf16
// [tokens, hidden]: y = signs * H32(y_ane * intermediate_scale * token_scale), the down projection's
// output rotation applied per 32-channel block.
KERNEL(AneUnpack)(
    const device half* surface,
    const device half* token_scales,
    const device float* signs,
    device bfloat* output,
    constant uint& hidden,
    constant uint& tokens,
    constant uint& row_stride,
    threadgroup float tile[ANE_TILE * (ANE_TILE + 1)],
    const ThreadContext thread_context,
    const uint channel_tile GROUPS(hidden / 32),
    const uint token_tile GROUPS(tokens.div_ceil(32)),
    const uint thread_id THREADS(ANE_THREADS)
) {
  const uint channel_base = channel_tile * ANE_TILE;
  const uint token_base = token_tile * ANE_TILE;
  for (uint index = thread_id; index < ANE_TILE * ANE_TILE; index += ANE_THREADS) {
    const uint channel = index / ANE_TILE;
    const uint token = index % ANE_TILE;
    float value = 0.0f;
    if (token_base + token < tokens) {
      value = float(surface[ulong(channel_base + channel) * row_stride + token_base + token]);
    }
    tile[channel * (ANE_TILE + 1) + token] = value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const ushort lane = thread_context.simd_lane_id;
  const float sign = signs[channel_base + lane];
  for (uint token = thread_context.simdgroup_index; token < ANE_TILE; token += ANE_THREADS / METAL_SIMD_SIZE) {
    const uint global_token = token_base + token;
    if (global_token >= tokens) {
      break;
    }
    const float scale = float(surface[ulong(hidden) * row_stride + global_token]) * float(token_scales[global_token]);
    const float value = simdgroup_hadamard_transform(lane, tile[lane * (ANE_TILE + 1) + token] * scale) * sign;
    output[ulong(global_token) * hidden + channel_base + lane] = bfloat(value);
  }
}

// One staged weight row for the ANE: dequantize a W4 asymmetric group-32 row (mirai "output_input"
// layout: codes [rows, columns/2] low nibble first, scales [groups, scale_stride] bf16, zero points
// [groups, scale_stride/2] low nibble = even row), then
//   r' = post_signs * H_block(pre_signs * H32(r))
// and quantize to int8 with one scale per row (written x128 into a [rows, 1] fp16 surface).
// The row (chunks * 4096 columns, chunks <= 3) stays in registers; block divides 4096.
KERNEL(AneStageRows)(
    const device uchar* weight_codes,
    const device bfloat* weight_scales,
    const device uchar* zero_points,
    const device float* pre_signs,
    const device float* post_signs OPTIONAL(has_post_signs),
    device char* staged,
    device half* row_scales,
    constant uint& source_row_offset,
    constant uint& rows,
    constant uint& scale_stride,
    constant uint& staged_row_stride,
    constant uint& row_scale_stride,
    const uint block SPECIALIZE,
    const uint chunks SPECIALIZE,
    const bool has_post_signs SPECIALIZE,
    threadgroup float shared[ANE_CHUNK],
    threadgroup float partials[METAL_SIMD_SIZE],
    const ThreadContext thread_context,
    const uint row GROUPS(rows),
    const uint thread_id THREADS(ANE_THREADS)
) {
  const uint source_row = source_row_offset + row;
  const uint columns = chunks * ANE_CHUNK;
  const device uchar* row_codes = weight_codes + ulong(source_row) * (columns / 2);
  const ushort lane = thread_context.simd_lane_id;
  const uint zero_shift = (source_row & 1) * 4;

  float values[3][ANE_VALUES];
  float peak = 0.0f;
  for (uint chunk = 0; chunk < chunks; ++chunk) {
    const uint base = chunk * ANE_CHUNK + thread_id * ANE_VALUES;
    const uint group = base / 32;
    const float scale = float(weight_scales[ulong(group) * scale_stride + source_row]);
    const float zero = float((zero_points[ulong(group) * (scale_stride / 2) + source_row / 2] >> zero_shift) & 0xF);
    const uint2 packed = *reinterpret_cast<const device uint2*>(row_codes + base / 2);
    thread float* chunk_values = values[chunk];
    METAL_PRAGMA_UNROLL
    for (ushort index = 0; index < ANE_VALUES; ++index) {
      const uint word = index < 8 ? packed.x : packed.y;
      const uint code = (word >> ((index % 8) * 4)) & 0xF;
      chunk_values[index] = scale * (float(code) - zero);
    }
    block_hadamard(chunk_values, 32, lane, shared, thread_id);
    METAL_PRAGMA_UNROLL
    for (ushort index = 0; index < ANE_VALUES; ++index) {
      chunk_values[index] *= pre_signs[base + index];
    }
    block_hadamard(chunk_values, block, lane, shared, thread_id);
    if (has_post_signs) {
      METAL_PRAGMA_UNROLL
      for (ushort index = 0; index < ANE_VALUES; ++index) {
        chunk_values[index] *= post_signs[base + index];
      }
    }
    peak = max(peak, values_peak(chunk_values));
  }
  peak = max(threadgroup_peak(peak, partials, thread_context), 1e-20f);
  device char* destination = staged + ulong(row) * staged_row_stride;
  for (uint chunk = 0; chunk < chunks; ++chunk) {
    store_codes(destination + chunk * ANE_CHUNK + thread_id * ANE_VALUES, values[chunk], 127.0f / peak);
  }
  if (thread_id == 0) {
    row_scales[ulong(row) * row_scale_stride] = half(peak / 127.0f * 128.0f);
  }
}
