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

// Normalized Walsh-Hadamard transform of every consecutive `block` values of `values[0..count)`,
// in place in threadgroup memory. Each stage pairs indices within 2h-aligned chunks, so blocks never mix.
static void threadgroup_hadamard(threadgroup float* values, uint count, uint block, uint thread_id) {
  for (uint half_span = 1; half_span < block; half_span <<= 1) {
    for (uint pair = thread_id; pair < count / 2; pair += ANE_THREADS) {
      const uint index = (pair / half_span) * 2 * half_span + pair % half_span;
      const float left = values[index];
      const float right = values[index + half_span];
      values[index] = left + right;
      values[index + half_span] = left - right;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float scale = rsqrt(float(block));
  for (uint index = thread_id; index < count; index += ANE_THREADS) {
    values[index] *= scale;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

static float threadgroup_peak(float value, threadgroup float* partials, const thread ThreadContext& thread_context) {
  return threadgroup_cooperative_reduce<SimdReduceMax<float>, ANE_THREADS>(value, partials, thread_context);
}

// Per token: a = H_hidden(signs * x), int8 codes with one scale per token.
// Writes token-major codes (transposed into the ANE surface by AneTransposeCodes) and the
// token scale (x128, the ANE program's dequantize convention) into the `tx` surface.
KERNEL(AnePack)(
    const device bfloat* input,
    const device float* signs,
    device char* codes,
    device half* token_scales,
    constant uint& hidden,
    constant uint& tokens,
    threadgroup float values[ANE_CHUNK],
    threadgroup float partials[METAL_SIMD_SIZE],
    const ThreadContext thread_context,
    const uint token GROUPS(tokens),
    const uint thread_id THREADS(ANE_THREADS)
) {
  const device bfloat* row = input + ulong(token) * hidden;
  for (uint index = thread_id; index < hidden; index += ANE_THREADS) {
    values[index] = float(row[index]) * signs[index];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  threadgroup_hadamard(values, hidden, hidden, thread_id);

  float peak = 0.0f;
  for (uint index = thread_id; index < hidden; index += ANE_THREADS) {
    peak = max(peak, fabs(values[index]));
  }
  peak = max(threadgroup_peak(peak, partials, thread_context), 1e-20f);
  const float inverse = 127.0f / peak;
  device char* destination = codes + ulong(token) * hidden;
  for (uint index = thread_id; index < hidden; index += ANE_THREADS) {
    destination[index] = char(clamp(rint(values[index] * inverse), -127.0f, 127.0f));
  }
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
// Rows longer than ANE_CHUNK are processed in ANE_CHUNK pieces (block must divide ANE_CHUNK), twice:
// once for the row peak, once to write the codes.
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
    constant uint& columns,
    constant uint& scale_stride,
    constant uint& staged_row_stride,
    constant uint& row_scale_stride,
    const uint block SPECIALIZE,
    const bool has_post_signs SPECIALIZE,
    threadgroup float values[ANE_CHUNK],
    threadgroup float partials[METAL_SIMD_SIZE],
    const ThreadContext thread_context,
    const uint row GROUPS(rows),
    const uint thread_id THREADS(ANE_THREADS)
) {
  const uint source_row = source_row_offset + row;
  const device uchar* row_codes = weight_codes + ulong(source_row) * (columns / 2);
  const uint chunk = min(columns, uint(ANE_CHUNK));
  const uint chunks = columns / chunk;
  float peak = 0.0f;
  device char* destination = staged + ulong(row) * staged_row_stride;

  for (uint pass = 0; pass < 2; ++pass) {
    const float inverse = pass == 0 ? 0.0f : 127.0f / peak;
    for (uint piece = 0; piece < chunks; ++piece) {
      const uint base = piece * chunk;
      if (pass == 1 && chunks == 1) {
        // The single chunk is still in threadgroup memory.
      } else {
        for (uint index = thread_id; index < chunk; index += ANE_THREADS) {
          const uint column = base + index;
          const uint group = column / 32;
          const uint code = (row_codes[column / 2] >> ((column & 1) * 4)) & 0xF;
          const uint zero =
              (zero_points[ulong(group) * (scale_stride / 2) + source_row / 2] >> ((source_row & 1) * 4)) & 0xF;
          const float scale = float(weight_scales[ulong(group) * scale_stride + source_row]);
          values[index] = scale * (float(code) - float(zero));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup_hadamard(values, chunk, 32, thread_id);
        for (uint index = thread_id; index < chunk; index += ANE_THREADS) {
          values[index] *= pre_signs[base + index];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup_hadamard(values, chunk, block, thread_id);
        if (has_post_signs) {
          for (uint index = thread_id; index < chunk; index += ANE_THREADS) {
            values[index] *= post_signs[base + index];
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
      }
      if (pass == 0) {
        for (uint index = thread_id; index < chunk; index += ANE_THREADS) {
          peak = max(peak, fabs(values[index]));
        }
      } else {
        for (uint index = thread_id; index < chunk; index += ANE_THREADS) {
          destination[base + index] = char(clamp(rint(values[index] * inverse), -127.0f, 127.0f));
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (pass == 0) {
      peak = max(threadgroup_peak(peak, partials, thread_context), 1e-20f);
    }
  }
  if (thread_id == 0) {
    row_scales[ulong(row) * row_scale_stride] = half(peak / 127.0f * 128.0f);
  }
}
