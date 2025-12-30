#include <metal_stdlib>
using namespace metal;

// Tiled GEMV with vector blocking in threadgroup memory.
// Signature matches existing Rust dispatch (no bias/axpby).
template <typename T, typename AccT, ushort ROWS_PER_TG, ushort BK>
METAL_FUNC void gemv_impl_tiled(
    const device T* __restrict matrix [[buffer(0)]],
    const device T* __restrict vector [[buffer(1)]],
    device T* __restrict output [[buffer(2)]],
    constant uint& k [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& ldb [[buffer(5)]],
    constant uint& lda [[buffer(6)]],
    constant uint& ldd [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint tid [[thread_index_in_threadgroup]],
    threadgroup AccT* vec_tile) {
    const ushort TG_THREADS = ROWS_PER_TG * 32;
    if (simd_gid >= ROWS_PER_TG) {
        return;
    }
    const uint row = tgid.x * ROWS_PER_TG + simd_gid;
    if (row >= n) {
        return;
    }
    const uint batch = tgid.z;

    const device T* row_ptr = matrix + row * ldb;
    const device T* vec_ptr = vector + batch * lda;
    device T* out_ptr = output + batch * ldd;

    AccT acc = static_cast<AccT>(0);

    // Cooperative load of vector tile into threadgroup memory.
    for (uint k_block = 0; k_block < k; k_block += BK) {
        const uint remaining = k - k_block;
        const uint tile_elems = remaining < BK ? remaining : BK;

        // Each thread loads at most one element of the vector tile.
        if (tid < tile_elems && tid < TG_THREADS) {
            vec_tile[tid] = static_cast<AccT>(vec_ptr[k_block + tid]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate over this tile.
        for (uint idx = simd_lid; idx < tile_elems; idx += 32) {
            acc += static_cast<AccT>(row_ptr[k_block + idx]) * vec_tile[idx];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Reduce within simdgroup.
    for (ushort offset = 16; offset > 0; offset >>= 1) {
        acc += simd_shuffle_down(acc, offset);
    }

    if (simd_lid == 0) {
        out_ptr[row] = static_cast<T>(acc);
    }
}

#define DECL_GEMV(NAME, TYPE, ACC, ROWS, BK)                                      \
    [[kernel, max_total_threads_per_threadgroup(ROWS * 32)]] void NAME(           \
        const device TYPE* matrix [[buffer(0)]],                                  \
        const device TYPE* vector [[buffer(1)]],                                  \
        device TYPE* output [[buffer(2)]],                                        \
        constant uint& k [[buffer(3)]],                                           \
        constant uint& n [[buffer(4)]],                                           \
        constant uint& ldb [[buffer(5)]],                                         \
        constant uint& lda [[buffer(6)]],                                         \
        constant uint& ldd [[buffer(7)]],                                         \
        uint3 tgid [[threadgroup_position_in_grid]],                              \
        uint simd_gid [[simdgroup_index_in_threadgroup]],                         \
        uint simd_lid [[thread_index_in_simdgroup]],                              \
        uint tid [[thread_index_in_threadgroup]]) {                               \
        threadgroup ACC vec_tile[BK];                                             \
        gemv_impl_tiled<TYPE, ACC, ROWS, BK>(                                     \
            matrix, vector, output, k, n, ldb, lda, ldd,                          \
            tgid, simd_gid, simd_lid, tid, vec_tile);                             \
    }

// half variants
DECL_GEMV(gemv_f16_rows2, half, float, 2, 128)
DECL_GEMV(gemv_f16_rows4, half, float, 4, 128)
DECL_GEMV(gemv_f16_rows8, half, float, 8, 128)

// bfloat16 variants
DECL_GEMV(gemv_bf16_rows2, bfloat, float, 2, 128)
DECL_GEMV(gemv_bf16_rows4, bfloat, float, 4, 128)
DECL_GEMV(gemv_bf16_rows8, bfloat, float, 8, 128)
#include <metal_stdlib>
using namespace metal;

#define MLX_MTL_PRAGMA_UNROLL _Pragma("unroll")

// Simplified MLX-style GEMV without bias/axpby; compact signature.
template <
    typename T,
    const int BM,
    const int BN,
    const int SM,
    const int SN,
    const int TM,
    const int TN>
struct GEMVKernel {
  enum : int {
    threadsM = BM * SM,
    threadsN = BN * SN,
    blockM = threadsM * TM,
    blockN = threadsN * TN,
  };

  static_assert(SM * SN == 32, "simdgroup must be 32 threads");
  static_assert(SN == 4 || SN == 8 || SN == 16 || SN == 32, "SN must divide 32");

  template <typename U = T>
  static METAL_FUNC void load_safe(
      const device T* src,
      thread U dst[TN],
      const int src_offset,
      const int src_size) {
    if (src_offset + TN <= src_size) {
      MLX_MTL_PRAGMA_UNROLL
      for (int tn = 0; tn < TN; tn++) {
        dst[tn] = static_cast<U>(src[src_offset + tn]);
      }
    } else {
      MLX_MTL_PRAGMA_UNROLL
      for (int tn = 0; tn < TN; tn++) {
        dst[tn] = src_offset + tn < src_size
            ? static_cast<U>(src[src_offset + tn])
            : U(0);
      }
    }
  }

  static METAL_FUNC void run(
      const device T* mat [[buffer(0)]],
      const device T* in_vec [[buffer(1)]],
      device T* out_vec [[buffer(2)]],
      const constant int& in_vec_size [[buffer(3)]],
      const constant int& out_vec_size [[buffer(4)]],
      const constant int& matrix_ld [[buffer(5)]],
      threadgroup float* tgp_memory [[threadgroup(0)]],
      uint3 tid [[threadgroup_position_in_grid]],
      uint3 lid [[thread_position_in_threadgroup]],
      uint simd_gid [[simdgroup_index_in_threadgroup]],
      uint simd_lid [[thread_index_in_simdgroup]]) {
    (void)lid;

    thread float result[TM] = {0};
    thread T inter[TN];
    thread float v_coeff[TN];

    const int thrM = SN != 32 ? simd_lid / SN : 0;
    const int thrN = SN != 32 ? simd_lid % SN : int(simd_lid);

    const int sgN = BN != 1 ? (simd_gid % BN) : 0;

    const int simdM = BN != 1 ? SM * (simd_gid / BN) : int(SM * simd_gid);
    const int simdN = BN != 1 ? SN * (simd_gid % BN) : 0;

    int bm = (simdM + thrM) * TM;
    int bn = (simdN + thrN) * TN;

    int out_row = tid.x * blockM + bm;
    if (out_row >= out_vec_size) {
      return;
    }
    out_row = out_row + TM <= out_vec_size ? out_row : out_vec_size - TM;

    mat += out_row * matrix_ld;

    const int loop_stride = blockN;
    const int in_size = in_vec_size;
    const int n_iter = in_size / loop_stride;
    const int leftover = in_size - loop_stride * n_iter;

    for (int i = 0; i < n_iter; ++i) {
      load_safe<float>(in_vec, v_coeff, bn, TN);

      int mat_offset = 0;
      MLX_MTL_PRAGMA_UNROLL
      for (int tm = 0; tm < TM; tm++) {
        load_safe(mat, inter, mat_offset + bn, TN);
        MLX_MTL_PRAGMA_UNROLL
        for (int tn = 0; tn < TN; tn++) {
          result[tm] += inter[tn] * v_coeff[tn];
        }
        mat_offset += matrix_ld;
      }

      bn += blockN;
    }

    if (leftover > 0) {
      load_safe<float>(in_vec, v_coeff, bn, in_size);

      MLX_MTL_PRAGMA_UNROLL
      for (int tm = 0; tm < TM; tm++) {
        load_safe(&mat[tm * matrix_ld], inter, bn, in_size);
        MLX_MTL_PRAGMA_UNROLL
        for (int tn = 0; tn < TN; tn++) {
          result[tm] += inter[tn] * v_coeff[tn];
        }
      }
    }

    MLX_MTL_PRAGMA_UNROLL
    for (int tm = 0; tm < TM; tm++) {
      MLX_MTL_PRAGMA_UNROLL
      for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) {
        result[tm] += simd_shuffle_down(result[tm], sn);
      }
    }

    if (BN > 1) {
      threadgroup float* tgp_results = tgp_memory + sgN * (blockM + TM) + bm;
      if (thrN == 0) {
        MLX_MTL_PRAGMA_UNROLL
        for (int tm = 0; tm < TM; tm++) {
          tgp_results[tm] = result[tm];
        }

        threadgroup_barrier(mem_flags::mem_none);

        if (sgN == 0) {
          MLX_MTL_PRAGMA_UNROLL
          for (int sgn = 1; sgn < BN; sgn++) {
            MLX_MTL_PRAGMA_UNROLL
            for (int tm = 0; tm < TM; tm++) {
              result[tm] += tgp_results[sgn * (blockM + TM) + tm];
            }
          }
        }
      }
    }

    if (simdN == 0 && thrN == 0) {
      MLX_MTL_PRAGMA_UNROLL
      for (int tm = 0; tm < TM; tm++) {
        out_vec[out_row + tm] = static_cast<T>(result[tm]);
      }
    }
  }
};

#define DECL_GEMV(name, itype, bm, bn, sm, sn, tm, tn)                  \
  [[kernel, max_total_threads_per_threadgroup(bm * sm * bn * sn)]] void \
  name(                                                                 \
      const device itype* mat [[buffer(0)]],                            \
      const device itype* in_vec [[buffer(1)]],                         \
      device itype* out_vec [[buffer(2)]],                              \
      const constant int& in_vec_size [[buffer(3)]],                    \
      const constant int& out_vec_size [[buffer(4)]],                   \
      const constant int& matrix_ld [[buffer(5)]],                      \
      uint3 tid [[threadgroup_position_in_grid]],                       \
      uint3 lid [[thread_position_in_threadgroup]],                     \
      uint simd_gid [[simdgroup_index_in_threadgroup]],                 \
      uint simd_lid [[thread_index_in_simdgroup]],                      \
      threadgroup float* tgp_memory [[threadgroup(0)]]) {               \
    GEMVKernel<itype, bm, bn, sm, sn, tm, tn>::run(                     \
        mat, in_vec, out_vec, in_vec_size, out_vec_size,                \
        matrix_ld, tgp_memory,                                         \
        tid, lid, simd_gid, simd_lid);                                  \
  }

#define DECL_GEMV_BLOCKS(name, itype)                                   \
  DECL_GEMV(name##_bm2_bn1_sm4_sn8_tm1_tn4, itype, 2, 1, 4, 8, 1, 4)    \
  DECL_GEMV(name##_bm2_bn1_sm4_sn8_tm4_tn4, itype, 2, 1, 4, 8, 4, 4)    \
  DECL_GEMV(name##_bm2_bn1_sm2_sn16_tm1_tn4, itype, 2, 1, 2, 16, 1, 4)  \
  DECL_GEMV(name##_bm2_bn1_sm2_sn16_tm4_tn4, itype, 2, 1, 2, 16, 4, 4)  \
  DECL_GEMV(name##_bm4_bn1_sm2_sn16_tm4_tn4, itype, 4, 1, 2, 16, 4, 4)

DECL_GEMV_BLOCKS(gemv_f32, float)
DECL_GEMV_BLOCKS(gemv_f16, half)
DECL_GEMV_BLOCKS(gemv_bf16, bfloat)
