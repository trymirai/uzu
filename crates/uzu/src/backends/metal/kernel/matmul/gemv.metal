// SPDX-License-Identifier: MIT

#include <metal_stdlib>
using namespace metal;

template <
    typename T,
    typename AccT,
    int ROWS_PER_SIMDGROUP,
    int ELEMS_PER_THREAD>
METAL_FUNC void gemv_multirow(
    const device T* __restrict weights,
    const device T* __restrict input,
    device T* __restrict output,
    const int input_dim,
    const int output_dim,
    const int weight_stride,
    const int batch_idx,
    const int input_batch_stride,
    const int output_batch_stride,
    uint row_base,
    uint simd_lid
) {

  constexpr int BLOCK_K = 32 * ELEMS_PER_THREAD;

  input += batch_idx * input_batch_stride;
  output += batch_idx * output_batch_stride;

  if (row_base >= static_cast<uint>(output_dim)) {
    return;
  }

  const int rows_to_compute =
      min(ROWS_PER_SIMDGROUP, output_dim - static_cast<int>(row_base));

  AccT accumulators[ROWS_PER_SIMDGROUP];
#pragma unroll(ROWS_PER_SIMDGROUP)
  for (int r = 0; r < ROWS_PER_SIMDGROUP; ++r) {
    accumulators[r] = AccT(0);
  }

  AccT input_cache[ELEMS_PER_THREAD];

  const device T* weight_row_ptrs[ROWS_PER_SIMDGROUP];
#pragma unroll(ROWS_PER_SIMDGROUP)
  for (int r = 0; r < ROWS_PER_SIMDGROUP; ++r) {
    weight_row_ptrs[r] = weights + (row_base + r) * weight_stride;
  }

  int k = 0;
  for (; k + BLOCK_K <= input_dim; k += BLOCK_K) {
    const int thread_k = k + simd_lid * ELEMS_PER_THREAD;

#pragma unroll(ELEMS_PER_THREAD)
    for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
      input_cache[e] = static_cast<AccT>(input[thread_k + e]);
    }

#pragma unroll(ROWS_PER_SIMDGROUP)
    for (int r = 0; r < ROWS_PER_SIMDGROUP; ++r) {
      AccT dot = 0;
#pragma unroll(ELEMS_PER_THREAD)
      for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
        dot += input_cache[e] *
               static_cast<AccT>(weight_row_ptrs[r][thread_k + e]);
      }
      accumulators[r] += dot;
    }
  }

  if (k < input_dim) {
    const int thread_k = k + simd_lid * ELEMS_PER_THREAD;
    const int remaining = input_dim - k;

#pragma unroll(ELEMS_PER_THREAD)
    for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
      const int idx = simd_lid * ELEMS_PER_THREAD + e;
      input_cache[e] =
          (idx < remaining) ? static_cast<AccT>(input[thread_k + e]) : AccT(0);
    }

#pragma unroll(ROWS_PER_SIMDGROUP)
    for (int r = 0; r < ROWS_PER_SIMDGROUP; ++r) {
      AccT dot = 0;
#pragma unroll(ELEMS_PER_THREAD)
      for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
        const int idx = simd_lid * ELEMS_PER_THREAD + e;
        if (idx < remaining) {
          dot += input_cache[e] *
                 static_cast<AccT>(weight_row_ptrs[r][thread_k + e]);
        }
      }
      accumulators[r] += dot;
    }
  }

#pragma unroll(ROWS_PER_SIMDGROUP)
  for (int r = 0; r < ROWS_PER_SIMDGROUP; ++r) {
    accumulators[r] = simd_sum(accumulators[r]);
  }

  if (simd_lid == 0) {
#pragma unroll(ROWS_PER_SIMDGROUP)
    for (int r = 0; r < rows_to_compute; ++r) {
      output[row_base + r] = static_cast<T>(accumulators[r]);
    }
  }
}

#define DECL_GEMV_FAST(NAME, TYPE, ACC)                                        \
  [[kernel]] void NAME(                                                        \
      const device TYPE* weights [[buffer(0)]],                                \
      const device TYPE* input [[buffer(1)]],                                  \
      device TYPE* output [[buffer(2)]],                                       \
      constant int& input_dim [[buffer(3)]],                                   \
      constant int& output_dim [[buffer(4)]],                                  \
      constant int& weight_stride [[buffer(5)]],                               \
      constant int& input_batch_stride [[buffer(6)]],                          \
      constant int& output_batch_stride [[buffer(7)]],                         \
      uint3 tgid [[threadgroup_position_in_grid]],                             \
      uint simd_lid [[thread_index_in_simdgroup]]                              \
  ) {                                                                          \
    constexpr int ROWS_PER_SIMDGROUP = 4;                                      \
    constexpr int ELEMS_PER_THREAD = 4;                                        \
    const uint row_base = tgid.x * ROWS_PER_SIMDGROUP;                         \
    gemv_multirow<TYPE, ACC, ROWS_PER_SIMDGROUP, ELEMS_PER_THREAD>(            \
        weights,                                                               \
        input,                                                                 \
        output,                                                                \
        input_dim,                                                             \
        output_dim,                                                            \
        weight_stride,                                                         \
        tgid.z,                                                                \
        input_batch_stride,                                                    \
        output_batch_stride,                                                   \
        row_base,                                                              \
        simd_lid                                                               \
    );                                                                         \
  }

DECL_GEMV_FAST(gemv_fast_f16, half, float)
DECL_GEMV_FAST(gemv_fast_bf16, bfloat, float)
DECL_GEMV_FAST(gemv_fast_f32, float, float)

#define DECL_GEMV_FAST_BIAS(NAME, TYPE, ACC)                                   \
  [[kernel]] void NAME(                                                        \
      const device TYPE* weights [[buffer(0)]],                                \
      const device TYPE* input [[buffer(1)]],                                  \
      const device TYPE* bias [[buffer(2)]],                                   \
      device TYPE* output [[buffer(3)]],                                       \
      constant int& input_dim [[buffer(4)]],                                   \
      constant int& output_dim [[buffer(5)]],                                  \
      constant int& weight_stride [[buffer(6)]],                               \
      constant int& input_batch_stride [[buffer(7)]],                          \
      constant int& output_batch_stride [[buffer(8)]],                         \
      uint3 tgid [[threadgroup_position_in_grid]],                             \
      uint simd_lid [[thread_index_in_simdgroup]]                              \
  ) {                                                                          \
    constexpr int ROWS_PER_SIMDGROUP = 4;                                      \
    constexpr int ELEMS_PER_THREAD = 4;                                        \
    const uint row_base = tgid.x * ROWS_PER_SIMDGROUP;                         \
    gemv_multirow<TYPE, ACC, ROWS_PER_SIMDGROUP, ELEMS_PER_THREAD>(            \
        weights,                                                               \
        input,                                                                 \
        output,                                                                \
        input_dim,                                                             \
        output_dim,                                                            \
        weight_stride,                                                         \
        tgid.z,                                                                \
        input_batch_stride,                                                    \
        output_batch_stride,                                                   \
        row_base,                                                              \
        simd_lid                                                               \
    );                                                                         \
    if (simd_lid == 0 && row_base < static_cast<uint>(output_dim)) {           \
      const int rows_to_compute =                                              \
          min(ROWS_PER_SIMDGROUP, output_dim - static_cast<int>(row_base));    \
      device TYPE* out_ptr = output + tgid.z * output_batch_stride;            \
      for (int r = 0; r < rows_to_compute; ++r) {                              \
        const uint row = row_base + static_cast<uint>(r);                      \
        out_ptr[row] = out_ptr[row] + bias[row];                               \
      }                                                                        \
    }                                                                          \
  }

DECL_GEMV_FAST_BIAS(gemv_fast_f16_bias, half, float)
DECL_GEMV_FAST_BIAS(gemv_fast_bf16_bias, bfloat, float)
DECL_GEMV_FAST_BIAS(gemv_fast_f32_bias, float, float)

template <typename T, typename AccT, ushort ROWS_PER_THREADGROUP, ushort TILE_K>
METAL_FUNC void gemv_tiled(
    const device T* __restrict weights,
    const device T* __restrict input,
    device T* __restrict output,
    constant uint& input_dim,
    constant uint& output_dim,
    constant uint& weight_stride,
    constant uint& input_stride,
    constant uint& output_stride,
    uint3 tgid,
    uint simd_gid,
    uint simd_lid,
    uint tid,
    threadgroup AccT* input_tile
) {

  const ushort THREADS_PER_THREADGROUP = ROWS_PER_THREADGROUP * 32;
  if (simd_gid >= ROWS_PER_THREADGROUP) {
    return;
  }

  const uint row = tgid.x * ROWS_PER_THREADGROUP + simd_gid;
  if (row >= output_dim) {
    return;
  }

  const uint batch = tgid.z;
  const device T* weight_row = weights + row * weight_stride;
  const device T* input_ptr = input + batch * input_stride;
  device T* output_ptr = output + batch * output_stride;

  AccT accumulator = AccT(0);

  for (uint k_block = 0; k_block < input_dim; k_block += TILE_K) {
    const uint remaining = input_dim - k_block;
    const uint tile_size = remaining < TILE_K ? remaining : TILE_K;

    if (tid < tile_size && tid < THREADS_PER_THREADGROUP) {
      input_tile[tid] = static_cast<AccT>(input_ptr[k_block + tid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = simd_lid; idx < tile_size; idx += 32) {
      accumulator +=
          static_cast<AccT>(weight_row[k_block + idx]) * input_tile[idx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (ushort offset = 16; offset > 0; offset >>= 1) {
    accumulator += simd_shuffle_down(accumulator, offset);
  }

  if (simd_lid == 0) {
    output_ptr[row] = static_cast<T>(accumulator);
  }
}

#define DECL_GEMV_TILED(NAME, TYPE, ACC, ROWS, TILE_K)                         \
  [[kernel, max_total_threads_per_threadgroup(ROWS * 32)]] void NAME(          \
      const device TYPE* weights [[buffer(0)]],                                \
      const device TYPE* input [[buffer(1)]],                                  \
      device TYPE* output [[buffer(2)]],                                       \
      constant uint& input_dim [[buffer(3)]],                                  \
      constant uint& output_dim [[buffer(4)]],                                 \
      constant uint& weight_stride [[buffer(5)]],                              \
      constant uint& input_stride [[buffer(6)]],                               \
      constant uint& output_stride [[buffer(7)]],                              \
      uint3 tgid [[threadgroup_position_in_grid]],                             \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                        \
      uint simd_lid [[thread_index_in_simdgroup]],                             \
      uint tid [[thread_index_in_threadgroup]]                                 \
  ) {                                                                          \
    threadgroup ACC input_tile[TILE_K];                                        \
    gemv_tiled<TYPE, ACC, ROWS, TILE_K>(                                       \
        weights,                                                               \
        input,                                                                 \
        output,                                                                \
        input_dim,                                                             \
        output_dim,                                                            \
        weight_stride,                                                         \
        input_stride,                                                          \
        output_stride,                                                         \
        tgid,                                                                  \
        simd_gid,                                                              \
        simd_lid,                                                              \
        tid,                                                                   \
        input_tile                                                             \
    );                                                                         \
  }

#define DECL_GEMV_TILED_BIAS(NAME, TYPE, ACC, ROWS, TILE_K)                    \
  [[kernel, max_total_threads_per_threadgroup(ROWS * 32)]] void NAME(          \
      const device TYPE* weights [[buffer(0)]],                                \
      const device TYPE* input [[buffer(1)]],                                  \
      const device TYPE* bias [[buffer(2)]],                                   \
      device TYPE* output [[buffer(3)]],                                       \
      constant uint& input_dim [[buffer(4)]],                                  \
      constant uint& output_dim [[buffer(5)]],                                 \
      constant uint& weight_stride [[buffer(6)]],                              \
      constant uint& input_stride [[buffer(7)]],                               \
      constant uint& output_stride [[buffer(8)]],                              \
      uint3 tgid [[threadgroup_position_in_grid]],                             \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                        \
      uint simd_lid [[thread_index_in_simdgroup]],                             \
      uint tid [[thread_index_in_threadgroup]]                                 \
  ) {                                                                          \
    threadgroup ACC input_tile[TILE_K];                                        \
    gemv_tiled<TYPE, ACC, ROWS, TILE_K>(                                       \
        weights,                                                               \
        input,                                                                 \
        output,                                                                \
        input_dim,                                                             \
        output_dim,                                                            \
        weight_stride,                                                         \
        input_stride,                                                          \
        output_stride,                                                         \
        tgid,                                                                  \
        simd_gid,                                                              \
        simd_lid,                                                              \
        tid,                                                                   \
        input_tile                                                             \
    );                                                                         \
    const uint row = tgid.x * ROWS + simd_gid;                                 \
    if (simd_lid == 0 && simd_gid < ROWS && row < output_dim) {                \
      device TYPE* out_ptr = output + tgid.z * output_stride;                  \
      out_ptr[row] = out_ptr[row] + bias[row];                                 \
    }                                                                          \
  }

DECL_GEMV_TILED_BIAS(gemv_f16_rows2_bias, half, float, 2, 128)
DECL_GEMV_TILED_BIAS(gemv_f16_rows4_bias, half, float, 4, 128)
DECL_GEMV_TILED_BIAS(gemv_f16_rows8_bias, half, float, 8, 128)
DECL_GEMV_TILED_BIAS(gemv_f16_rows16_bias, half, float, 16, 128)

DECL_GEMV_TILED_BIAS(gemv_bf16_rows2_bias, bfloat, float, 2, 128)
DECL_GEMV_TILED_BIAS(gemv_bf16_rows4_bias, bfloat, float, 4, 128)
DECL_GEMV_TILED_BIAS(gemv_bf16_rows8_bias, bfloat, float, 8, 128)
DECL_GEMV_TILED_BIAS(gemv_bf16_rows16_bias, bfloat, float, 16, 128)

DECL_GEMV_TILED_BIAS(gemv_f32_rows2_bias, float, float, 2, 128)
DECL_GEMV_TILED_BIAS(gemv_f32_rows4_bias, float, float, 4, 128)
DECL_GEMV_TILED_BIAS(gemv_f32_rows8_bias, float, float, 8, 128)
DECL_GEMV_TILED_BIAS(gemv_f32_rows16_bias, float, float, 16, 128)

DECL_GEMV_TILED(gemv_f16_rows2, half, float, 2, 128)
DECL_GEMV_TILED(gemv_f16_rows4, half, float, 4, 128)
DECL_GEMV_TILED(gemv_f16_rows8, half, float, 8, 128)
DECL_GEMV_TILED(gemv_f16_rows16, half, float, 16, 128)

DECL_GEMV_TILED(gemv_bf16_rows2, bfloat, float, 2, 128)
DECL_GEMV_TILED(gemv_bf16_rows4, bfloat, float, 4, 128)
DECL_GEMV_TILED(gemv_bf16_rows8, bfloat, float, 8, 128)
DECL_GEMV_TILED(gemv_bf16_rows16, bfloat, float, 16, 128)

DECL_GEMV_TILED(gemv_f32_rows2, float, float, 2, 128)
DECL_GEMV_TILED(gemv_f32_rows4, float, float, 4, 128)
DECL_GEMV_TILED(gemv_f32_rows8, float, float, 8, 128)
DECL_GEMV_TILED(gemv_f32_rows16, float, float, 16, 128)

template <typename T, typename AccT, ushort KPARTS>
METAL_FUNC void gemv_split_k_rows16_impl(
    const device T* __restrict weights,
    const device T* __restrict input,
    const device T* __restrict bias,
    device T* __restrict output,
    const int input_dim,
    const int output_dim,
    const int weight_stride,
    const int input_batch_stride,
    const int output_batch_stride,
    uint3 tgid,
    uint simd_gid,
    uint simd_lid,
    threadgroup AccT* partial_sums
) {
  constexpr ushort rows_per_threadgroup = 16;
  constexpr ushort rows_per_rowgroup = 2;
  constexpr ushort rowgroups = rows_per_threadgroup / rows_per_rowgroup;
  constexpr ushort elems_per_thread = 4;
  constexpr int block_k = 32 * elems_per_thread;

  const uint rowgroup_idx = simd_gid / KPARTS;
  const uint kpart_idx = simd_gid - rowgroup_idx * KPARTS;

  const uint batch = tgid.z;
  const uint row_base =
      tgid.x * rows_per_threadgroup + rowgroup_idx * rows_per_rowgroup;

  const bool valid_rowgroup = row_base < static_cast<uint>(output_dim);
  const uint rows_to_compute = valid_rowgroup
      ? min(static_cast<uint>(rows_per_rowgroup),
            static_cast<uint>(output_dim) - row_base)
      : 0;

  const device T* input_ptr = input + batch * input_batch_stride;
  device T* output_ptr = output + batch * output_batch_stride;

  AccT accumulators[rows_per_rowgroup] = {AccT(0), AccT(0)};

  if (valid_rowgroup) {
    const device T* weight_row_ptrs[rows_per_rowgroup];
    weight_row_ptrs[0] = weights + (row_base + 0) * weight_stride;
    weight_row_ptrs[1] = weights + (row_base + 1) * weight_stride;

    for (int k_block = static_cast<int>(kpart_idx) * block_k;
         k_block < input_dim;
         k_block += block_k * static_cast<int>(KPARTS))
    {
      AccT input_cache[elems_per_thread];

#pragma unroll(elems_per_thread)
      for (int e = 0; e < elems_per_thread; ++e) {
        const int col = k_block + static_cast<int>(simd_lid) * elems_per_thread + e;
        input_cache[e] =
            (col < input_dim) ? static_cast<AccT>(input_ptr[col]) : AccT(0);
      }

#pragma unroll(rows_per_rowgroup)
      for (int r = 0; r < rows_per_rowgroup; ++r) {
        AccT dot = 0;
#pragma unroll(elems_per_thread)
        for (int e = 0; e < elems_per_thread; ++e) {
          const int col =
              k_block + static_cast<int>(simd_lid) * elems_per_thread + e;
          if (col < input_dim) {
            dot += input_cache[e] * static_cast<AccT>(weight_row_ptrs[r][col]);
          }
        }
        accumulators[r] += dot;
      }
    }
  }

#pragma unroll(rows_per_rowgroup)
  for (int r = 0; r < rows_per_rowgroup; ++r) {
    accumulators[r] = simd_sum(accumulators[r]);
  }

  const uint partial_base =
      rowgroup_idx * (KPARTS * rows_per_rowgroup) + kpart_idx * rows_per_rowgroup;

  if (simd_lid == 0) {
    partial_sums[partial_base + 0] = accumulators[0];
    partial_sums[partial_base + 1] = accumulators[1];
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (kpart_idx == 0 && simd_lid == 0 && rows_to_compute > 0) {
    for (uint r = 0; r < rows_to_compute; ++r) {
      AccT sum = 0;
      for (uint kp = 0; kp < KPARTS; ++kp) {
        sum += partial_sums[rowgroup_idx * (KPARTS * rows_per_rowgroup) +
                            kp * rows_per_rowgroup + r];
      }
      const uint row = row_base + r;
      if (bias != nullptr) {
        sum += static_cast<AccT>(bias[row]);
      }
      output_ptr[row] = static_cast<T>(sum);
    }
  }
}

#define DECL_GEMV_SPLIT_K_KPARTS2(NAME, TYPE, ACC)                              \
  [[kernel, max_total_threads_per_threadgroup(512)]] void NAME(                 \
      const device TYPE* weights [[buffer(0)]],                                 \
      const device TYPE* input [[buffer(1)]],                                   \
      device TYPE* output [[buffer(2)]],                                        \
      constant int& input_dim [[buffer(3)]],                                    \
      constant int& output_dim [[buffer(4)]],                                   \
      constant int& weight_stride [[buffer(5)]],                                \
      constant int& input_batch_stride [[buffer(6)]],                           \
      constant int& output_batch_stride [[buffer(7)]],                          \
      uint3 tgid [[threadgroup_position_in_grid]],                              \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                         \
      uint simd_lid [[thread_index_in_simdgroup]]                               \
  ) {                                                                           \
    threadgroup ACC partial_sums[8 * 2 * 2];                                    \
    gemv_split_k_rows16_impl<TYPE, ACC, 2>(                                     \
        weights,                                                                \
        input,                                                                  \
        nullptr,                                                                \
        output,                                                                 \
        input_dim,                                                              \
        output_dim,                                                             \
        weight_stride,                                                          \
        input_batch_stride,                                                     \
        output_batch_stride,                                                    \
        tgid,                                                                   \
        simd_gid,                                                               \
        simd_lid,                                                               \
        partial_sums                                                            \
    );                                                                          \
  }

#define DECL_GEMV_SPLIT_K_KPARTS4(NAME, TYPE, ACC)                              \
  [[kernel, max_total_threads_per_threadgroup(1024)]] void NAME(                \
      const device TYPE* weights [[buffer(0)]],                                 \
      const device TYPE* input [[buffer(1)]],                                   \
      device TYPE* output [[buffer(2)]],                                        \
      constant int& input_dim [[buffer(3)]],                                    \
      constant int& output_dim [[buffer(4)]],                                   \
      constant int& weight_stride [[buffer(5)]],                                \
      constant int& input_batch_stride [[buffer(6)]],                           \
      constant int& output_batch_stride [[buffer(7)]],                          \
      uint3 tgid [[threadgroup_position_in_grid]],                              \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                         \
      uint simd_lid [[thread_index_in_simdgroup]]                               \
  ) {                                                                           \
    threadgroup ACC partial_sums[8 * 4 * 2];                                    \
    gemv_split_k_rows16_impl<TYPE, ACC, 4>(                                     \
        weights,                                                                \
        input,                                                                  \
        nullptr,                                                                \
        output,                                                                 \
        input_dim,                                                              \
        output_dim,                                                             \
        weight_stride,                                                          \
        input_batch_stride,                                                     \
        output_batch_stride,                                                    \
        tgid,                                                                   \
        simd_gid,                                                               \
        simd_lid,                                                               \
        partial_sums                                                            \
    );                                                                          \
  }

#define DECL_GEMV_SPLIT_K_KPARTS2_BIAS(NAME, TYPE, ACC)                         \
  [[kernel, max_total_threads_per_threadgroup(512)]] void NAME(                 \
      const device TYPE* weights [[buffer(0)]],                                 \
      const device TYPE* input [[buffer(1)]],                                   \
      const device TYPE* bias [[buffer(2)]],                                    \
      device TYPE* output [[buffer(3)]],                                        \
      constant int& input_dim [[buffer(4)]],                                    \
      constant int& output_dim [[buffer(5)]],                                   \
      constant int& weight_stride [[buffer(6)]],                                \
      constant int& input_batch_stride [[buffer(7)]],                           \
      constant int& output_batch_stride [[buffer(8)]],                          \
      uint3 tgid [[threadgroup_position_in_grid]],                              \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                         \
      uint simd_lid [[thread_index_in_simdgroup]]                               \
  ) {                                                                           \
    threadgroup ACC partial_sums[8 * 2 * 2];                                    \
    gemv_split_k_rows16_impl<TYPE, ACC, 2>(                                     \
        weights,                                                                \
        input,                                                                  \
        bias,                                                                   \
        output,                                                                 \
        input_dim,                                                              \
        output_dim,                                                             \
        weight_stride,                                                          \
        input_batch_stride,                                                     \
        output_batch_stride,                                                    \
        tgid,                                                                   \
        simd_gid,                                                               \
        simd_lid,                                                               \
        partial_sums                                                            \
    );                                                                          \
  }

#define DECL_GEMV_SPLIT_K_KPARTS4_BIAS(NAME, TYPE, ACC)                         \
  [[kernel, max_total_threads_per_threadgroup(1024)]] void NAME(                \
      const device TYPE* weights [[buffer(0)]],                                 \
      const device TYPE* input [[buffer(1)]],                                   \
      const device TYPE* bias [[buffer(2)]],                                    \
      device TYPE* output [[buffer(3)]],                                        \
      constant int& input_dim [[buffer(4)]],                                    \
      constant int& output_dim [[buffer(5)]],                                   \
      constant int& weight_stride [[buffer(6)]],                                \
      constant int& input_batch_stride [[buffer(7)]],                           \
      constant int& output_batch_stride [[buffer(8)]],                          \
      uint3 tgid [[threadgroup_position_in_grid]],                              \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                         \
      uint simd_lid [[thread_index_in_simdgroup]]                               \
  ) {                                                                           \
    threadgroup ACC partial_sums[8 * 4 * 2];                                    \
    gemv_split_k_rows16_impl<TYPE, ACC, 4>(                                     \
        weights,                                                                \
        input,                                                                  \
        bias,                                                                   \
        output,                                                                 \
        input_dim,                                                              \
        output_dim,                                                             \
        weight_stride,                                                          \
        input_batch_stride,                                                     \
        output_batch_stride,                                                    \
        tgid,                                                                   \
        simd_gid,                                                               \
        simd_lid,                                                               \
        partial_sums                                                            \
    );                                                                          \
  }

DECL_GEMV_SPLIT_K_KPARTS2(gemv_split_k_f16_rows16_kparts2, half, float)
DECL_GEMV_SPLIT_K_KPARTS4(gemv_split_k_f16_rows16_kparts4, half, float)
DECL_GEMV_SPLIT_K_KPARTS2_BIAS(gemv_split_k_f16_rows16_kparts2_bias, half, float)
DECL_GEMV_SPLIT_K_KPARTS4_BIAS(gemv_split_k_f16_rows16_kparts4_bias, half, float)

DECL_GEMV_SPLIT_K_KPARTS2(gemv_split_k_bf16_rows16_kparts2, bfloat, float)
DECL_GEMV_SPLIT_K_KPARTS4(gemv_split_k_bf16_rows16_kparts4, bfloat, float)
DECL_GEMV_SPLIT_K_KPARTS2_BIAS(gemv_split_k_bf16_rows16_kparts2_bias, bfloat, float)
DECL_GEMV_SPLIT_K_KPARTS4_BIAS(gemv_split_k_bf16_rows16_kparts4_bias, bfloat, float)

DECL_GEMV_SPLIT_K_KPARTS2(gemv_split_k_f32_rows16_kparts2, float, float)
DECL_GEMV_SPLIT_K_KPARTS4(gemv_split_k_f32_rows16_kparts4, float, float)
DECL_GEMV_SPLIT_K_KPARTS2_BIAS(gemv_split_k_f32_rows16_kparts2_bias, float, float)
DECL_GEMV_SPLIT_K_KPARTS4_BIAS(gemv_split_k_f32_rows16_kparts4_bias, float, float)