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
#define ANE_DOWN_BLOCK 512

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
    const float sign = (lane & stride) ? -1.0f : 1.0f;
    METAL_PRAGMA_UNROLL
    for (ushort index = 0; index < ANE_VALUES; ++index) {
      values[index] = fma(values[index], sign, simd_shuffle_xor(values[index], stride));
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

// Spans 512, 1024 and 2048 for a 4096-value chunk as a single threadgroup-memory transpose. Thread
// (simdgroup s, lane l) writes its 16 values transposed (value v of thread t at v * 256 + t, so both
// the writes and the reads below are bank-conflict-free), then owns the 8-point columns
// {g * 512 + 16 l + 2 s + j}, g in [0, 8), j in [0, 2). Returns them as values[2 * g + j]; the chunk
// stays in this layout (see store_codes_columns) so no transpose back is needed.
static void hadamard_across_simdgroups_columns(
    thread float* values,
    threadgroup float* shared,
    uint thread_id,
    ushort lane,
    ushort simdgroup
) {
  METAL_PRAGMA_UNROLL
  for (ushort index = 0; index < ANE_VALUES; ++index) {
    shared[index * ANE_THREADS + thread_id] = values[index];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  METAL_PRAGMA_UNROLL
  for (ushort j = 0; j < 2; ++j) {
    float column[8];
    METAL_PRAGMA_UNROLL
    for (ushort group = 0; group < 8; ++group) {
      column[group] = shared[(2 * simdgroup + j) * ANE_THREADS + group * METAL_SIMD_SIZE + lane];
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
      values[2 * group + j] = column[group];
    }
  }
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

// Unnormalized transform of every 4096 values for block 4096, leaving the chunk in the
// hadamard_across_simdgroups_columns layout.
static void block_hadamard_columns(
    thread float* values,
    ushort lane,
    ushort simdgroup,
    threadgroup float* shared,
    uint thread_id
) {
  hadamard_in_thread(values);
  hadamard_across_lanes(values, lane, 512 / ANE_VALUES);
  hadamard_across_simdgroups_columns(values, shared, thread_id, lane, simdgroup);
}

// `value` negated where bit `index` of `bits` is set (the +-1 sign vectors are stored as bit masks).
static float flip_sign(float value, uint bits, ushort index) {
  return as_type<float>(as_type<uint>(value) ^ ((bits << (31 - index)) & 0x80000000u));
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

// round-to-nearest-even int8 codes by adding 1.5 * 2^23: the sum's low byte is the code in two's complement.
#define ANE_ROUNDING_MAGIC 12582912.0f

static void store_codes(device char* destination, const thread float* values, float inverse) {
  METAL_PRAGMA_UNROLL
  for (ushort index = 0; index < ANE_VALUES; index += 4) {
    const uint4 bits = as_type<uint4>(
        fma(float4(values[index], values[index + 1], values[index + 2], values[index + 3]),
            float4(inverse),
            float4(ANE_ROUNDING_MAGIC))
    );
    uint packed = bits.w << 24;
    packed = insert_bits(packed, bits.z, 16, 8);
    packed = insert_bits(packed, bits.y, 8, 8);
    packed = insert_bits(packed, bits.x, 0, 8);
    *reinterpret_cast<device uint*>(destination + index) = packed;
  }
}

static void store_codes_columns(
    device char* destination,
    const thread float* values,
    float inverse,
    ushort lane,
    ushort simdgroup
) {
  METAL_PRAGMA_UNROLL
  for (ushort group = 0; group < 8; ++group) {
    const uint2 bits = as_type<uint2>(
        fma(float2(values[2 * group], values[2 * group + 1]), float2(inverse), float2(ANE_ROUNDING_MAGIC))
    );
    *reinterpret_cast<device ushort*>(destination + group * 512 + lane * ANE_VALUES + simdgroup * 2) =
        ushort(insert_bits(bits.x, bits.y, 8, 8));
  }
}

// Per token: a = H_4096(signs * x), int8 codes with one scale per token (hidden must be 4096).
// Writes token-major codes (transposed into the ANE surface by AneTransposeCodes) and the
// token scale (x128, the ANE program's dequantize convention) into the `tx` surface. With
// `half_values` the codes are not rounded: `codes` holds the fp16 values a / peak * 127 / 128, what the
// program's dequantize makes of the int8 codes, without the rounding error.
KERNEL(AnePack)(
    const device bfloat* input,
    const device float* signs,
    device char* codes,
    device half* token_scales,
    constant uint& hidden,
    constant uint& tokens,
    const bool half_values SPECIALIZE,
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
  const float inverse = 127.0f / peak;
  if (half_values) {
    device half* destination = reinterpret_cast<device half*>(codes) + ulong(token) * hidden + base;
    METAL_PRAGMA_UNROLL
    for (ushort index = 0; index < ANE_VALUES; ++index) {
      destination[index] = half(values[index] * inverse * (1.0f / 128.0f));
    }
  } else {
    store_codes(codes + ulong(token) * hidden + base, values, inverse);
  }
  if (thread_id == 0) {
    token_scales[token] = half(peak / 127.0f * 128.0f);
  }
}

// Token-major int8 (or, with `wide`, fp16) [tokens, channels] -> channel-major surface rows
// [channels, row_stride], `row_stride` in elements.
KERNEL(AneTransposeCodes)(
    const device char* codes,
    device char* surface,
    constant uint& channels,
    constant uint& tokens,
    constant uint& row_stride,
    const bool wide SPECIALIZE,
    threadgroup ushort tile[ANE_TILE * (ANE_TILE + 1)],
    const uint channel_tile GROUPS(channels.div_ceil(32)),
    const uint token_tile GROUPS(tokens.div_ceil(32)),
    const uint thread_id THREADS(ANE_THREADS)
) {
  const uint channel_base = channel_tile * ANE_TILE;
  const uint token_base = token_tile * ANE_TILE;
  for (uint index = thread_id; index < ANE_TILE * ANE_TILE; index += ANE_THREADS) {
    const uint token = index / ANE_TILE;
    const uint channel = index % ANE_TILE;
    ushort value = 0;
    if (token_base + token < tokens && channel_base + channel < channels) {
      const ulong source = ulong(token_base + token) * channels + channel_base + channel;
      value = wide ? reinterpret_cast<const device ushort*>(codes)[source] : ushort(uchar(codes[source]));
    }
    tile[channel * (ANE_TILE + 1) + token] = value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint index = thread_id; index < ANE_TILE * ANE_TILE; index += ANE_THREADS) {
    const uint channel = index / ANE_TILE;
    const uint token = index % ANE_TILE;
    if (token_base + token < tokens && channel_base + channel < channels) {
      const ulong destination = ulong(channel_base + channel) * row_stride + token_base + token;
      const ushort value = tile[channel * (ANE_TILE + 1) + token];
      if (wide) {
        reinterpret_cast<device ushort*>(surface)[destination] = value;
      } else {
        surface[destination] = char(value);
      }
    }
  }
}

// ANE output [hidden (+ 1), row_stride] fp16 -> token-major bf16 [tokens, hidden]:
// y = signs * H32(y_ane * intermediate_scale * token_scale), the projection's output rotation applied
// per 32-channel block. With `scale_row` the output carries a per-token intermediate scale in its last
// row (the FFN's down projection); without it (a single linear layer) that factor is 1.
KERNEL(AneUnpack)(
    const device half* surface,
    const device half* token_scales,
    const device float* signs,
    device bfloat* output,
    constant uint& hidden,
    constant uint& tokens,
    constant uint& row_stride,
    const bool scale_row SPECIALIZE,
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
    float scale = float(token_scales[global_token]);
    if (scale_row) {
      scale *= float(surface[ulong(hidden) * row_stride + global_token]);
    }
    const float value = simdgroup_hadamard_transform(lane, tile[lane * (ANE_TILE + 1) + token] * scale) * sign;
    output[ulong(global_token) * hidden + channel_base + lane] = bfloat(value);
  }
}

// Dequantize and transform one thread's 16 values of a W4 asymmetric group-32 row chunk:
// H32(scale * (code - zero)) up to the 1/sqrt(32) normalization, which is returned as the factor
// (scale / sqrt(32)) the caller folds into its sign multiply. The zero point only moves the DC term,
// so the transform runs on the raw (exact) integer codes.
static float dequantized_h32(thread float* values, uint2 packed, float scale, float zero, ushort lane) {
  METAL_PRAGMA_UNROLL
  for (ushort index = 0; index < ANE_VALUES; ++index) {
    const uint word = index < 8 ? packed.x : packed.y;
    values[index] = float((word >> ((index % 8) * 4)) & 0xF);
  }
  hadamard_in_thread(values);
  hadamard_across_lanes(values, lane, 2);
  values[0] -= (lane & 1) ? 0.0f : 32.0f * zero;
  return scale * rsqrt(32.0f);
}

// Staged weight rows for the ANE: dequantize a W4 asymmetric group-32 row (mirai "output_input"
// layout: codes [rows, columns/2] low nibble first, scales [groups, scale_stride] bf16, zero points
// [groups, scale_stride/2] low nibble = even row), then
//   r' = post_signs * H_block(pre_signs * H32(r))
// and quantize to int8 with one scale per row (written x128 into a [rows, 1] fp16 surface). The
// second transform's normalization is folded into the quantization (peak and inverse scale).

// Up projection: a 4096-column row, block 4096, no post signs. Float pre signs (their 16 KB per row
// stay cache resident, and the float multiply is what keeps this kernel ALU-light).
KERNEL(AneStageUpRows)(
    const device uchar* weight_codes,
    const device bfloat* weight_scales,
    const device uchar* zero_points,
    const device float* pre_signs,
    device char* staged,
    device half* row_scales,
    constant uint& source_row_offset,
    constant uint& rows,
    constant uint& scale_stride,
    constant uint& staged_row_stride,
    constant uint& row_scale_stride,
    threadgroup float shared[ANE_CHUNK],
    threadgroup float partials[METAL_SIMD_SIZE],
    const ThreadContext thread_context,
    const uint row GROUPS(rows),
    const uint thread_id THREADS(ANE_THREADS)
) {
  const uint source_row = source_row_offset + row;
  const ushort lane = thread_context.simd_lane_id;
  const ushort simdgroup = thread_context.simdgroup_index;
  const uint base = thread_id * ANE_VALUES;
  const uint group = base / 32;
  const float scale = float(weight_scales[ulong(group) * scale_stride + source_row]);
  const float zero =
      float((zero_points[ulong(group) * (scale_stride / 2) + source_row / 2] >> ((source_row & 1) * 4)) & 0xF);
  const uint2 packed =
      *reinterpret_cast<const device uint2*>(weight_codes + ulong(source_row) * (ANE_CHUNK / 2) + base / 2);

  float values[ANE_VALUES];
  const float group_scale = dequantized_h32(values, packed, scale, zero, lane);
  METAL_PRAGMA_UNROLL
  for (ushort index = 0; index < ANE_VALUES; index += 4) {
    const float4 signs = *reinterpret_cast<const device float4*>(pre_signs + base + index) * group_scale;
    values[index] *= signs.x;
    values[index + 1] *= signs.y;
    values[index + 2] *= signs.z;
    values[index + 3] *= signs.w;
  }
  // Unnormalized, and left in the hadamard_across_simdgroups_columns layout.
  block_hadamard_columns(values, lane, simdgroup, shared, thread_id);

  const float normalization = rsqrt(float(ANE_CHUNK));
  const float row_peak = max(threadgroup_peak(values_peak(values), partials, thread_context) * normalization, 1e-20f);
  store_codes_columns(
      staged + ulong(row) * staged_row_stride,
      values,
      127.0f * normalization / row_peak,
      lane,
      simdgroup
  );
  if (thread_id == 0) {
    row_scales[ulong(row) * row_scale_stride] = half(row_peak / 127.0f * 128.0f);
  }
}

// Down projection: a `chunks` * 4096-column row (chunks <= 3, held in registers), block 512, with post
// signs. The +-1 sign vectors are bit masks: the float signs of a 12288-column row (96 KB with both)
// did not stay cache resident and dominated the kernel.
KERNEL(AneStageDownRows)(
    const device uchar* weight_codes,
    const device bfloat* weight_scales,
    const device uchar* zero_points,
    const device ushort* pre_sign_bits,
    const device ushort* post_sign_bits,
    device char* staged,
    device half* row_scales,
    constant uint& source_row_offset,
    constant uint& rows,
    constant uint& scale_stride,
    constant uint& staged_row_stride,
    constant uint& row_scale_stride,
    const uint chunks SPECIALIZE,
    threadgroup float partials[METAL_SIMD_SIZE],
    const ThreadContext thread_context,
    const uint row GROUPS(rows),
    const uint thread_id THREADS(ANE_THREADS)
) {
  const uint source_row = source_row_offset + row;
  const device uchar* row_codes = weight_codes + ulong(source_row) * (chunks * ANE_CHUNK / 2);
  const ushort lane = thread_context.simd_lane_id;
  const uint zero_shift = (source_row & 1) * 4;

  float values[3][ANE_VALUES];
  float peak = 0.0f;
  METAL_PRAGMA_UNROLL
  for (uint chunk = 0; chunk < chunks; ++chunk) {
    const uint base = chunk * ANE_CHUNK + thread_id * ANE_VALUES;
    const uint group = base / 32;
    const float scale = float(weight_scales[ulong(group) * scale_stride + source_row]);
    const float zero = float((zero_points[ulong(group) * (scale_stride / 2) + source_row / 2] >> zero_shift) & 0xF);
    const uint2 packed = *reinterpret_cast<const device uint2*>(row_codes + base / 2);
    thread float* chunk_values = values[chunk];
    const float group_scale = dequantized_h32(chunk_values, packed, scale, zero, lane);
    const uint pre_bits = pre_sign_bits[base / ANE_VALUES];
    METAL_PRAGMA_UNROLL
    for (ushort index = 0; index < ANE_VALUES; ++index) {
      chunk_values[index] = flip_sign(chunk_values[index], pre_bits, index) * group_scale;
    }
    hadamard_in_thread(chunk_values);
    hadamard_across_lanes(chunk_values, lane, ANE_DOWN_BLOCK / ANE_VALUES);
    const uint post_bits = post_sign_bits[base / ANE_VALUES];
    METAL_PRAGMA_UNROLL
    for (ushort index = 0; index < ANE_VALUES; ++index) {
      chunk_values[index] = flip_sign(chunk_values[index], post_bits, index);
    }
    peak = max(peak, values_peak(chunk_values));
  }
  const float normalization = rsqrt(float(ANE_DOWN_BLOCK));
  const float row_peak = max(threadgroup_peak(peak, partials, thread_context) * normalization, 1e-20f);
  const float inverse = 127.0f * normalization / row_peak;
  device char* destination = staged + ulong(row) * staged_row_stride;
  METAL_PRAGMA_UNROLL
  for (uint chunk = 0; chunk < chunks; ++chunk) {
    store_codes(destination + chunk * ANE_CHUNK + thread_id * ANE_VALUES, values[chunk], inverse);
  }
  if (thread_id == 0) {
    row_scales[ulong(row) * row_scale_stride] = half(row_peak / 127.0f * 128.0f);
  }
}
