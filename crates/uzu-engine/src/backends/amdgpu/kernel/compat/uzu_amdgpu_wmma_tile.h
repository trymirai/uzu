// RDNA3 WMMA (v_wmma_f32_16x16x16_bf16) for uzu's ThreadgroupTile with bf16 operands: the GEMM
// accumulator of `matmul/common/threadgroup_tile.h`, included at its end when UZU_AMDGPU_WMMA is set.
//
// Each SIMD group owns a (BLOCK_ROWS / SIMDGROUPS_PER_ROW) x (BLOCK_COLS / SIMDGROUPS_PER_COLUMN)
// block, computed as 16x16 WMMA tiles. Operands come straight from the staged threadgroup tiles in the
// gfx11 wave32 layout: lane l holds row l % 16 of A and column l % 16 of B (16 K values each), and the
// 8 accumulators of lane l hold rows 2 * i + l / 16 of column l % 16. Blocks shorter than 16 rows
// (Tile8x32x32) zero the missing A rows and never store or read them.
#pragma once

namespace uzu {
namespace matmul {

typedef short __uzu_wmma_bf16x16 __attribute__((ext_vector_type(16)));
typedef float __uzu_wmma_f32x8 __attribute__((ext_vector_type(8)));

template <
    typename DT,
    int BLOCK_ROWS,
    int BLOCK_COLS,
    int BLOCK_DEPTH,
    int SIMDGROUPS_PER_ROW,
    int SIMDGROUPS_PER_COLUMN,
    bool transpose_a,
    bool transpose_b,
    ushort THREADGROUP_LEADING_DIMENSION_A,
    ushort THREADGROUP_LEADING_DIMENSION_B,
    typename Epilogue>
struct ThreadgroupTile<
    bfloat,
    bfloat,
    DT,
    BLOCK_ROWS,
    BLOCK_COLS,
    BLOCK_DEPTH,
    SIMDGROUPS_PER_ROW,
    SIMDGROUPS_PER_COLUMN,
    transpose_a,
    transpose_b,
    THREADGROUP_LEADING_DIMENSION_A,
    THREADGROUP_LEADING_DIMENSION_B,
    float,
    Epilogue> {
  using AccumulatorType = float;
  UZU_CONST int WMMA_SIZE = 16;
  UZU_CONST int SIMDGROUP_ROWS = BLOCK_ROWS / SIMDGROUPS_PER_ROW;
  UZU_CONST int SIMDGROUP_COLS = BLOCK_COLS / SIMDGROUPS_PER_COLUMN;
  UZU_CONST int TILES_M = (SIMDGROUP_ROWS + WMMA_SIZE - 1) / WMMA_SIZE;
  UZU_CONST int TILES_N = SIMDGROUP_COLS / WMMA_SIZE;
  static_assert(SIMDGROUP_COLS % WMMA_SIZE == 0, "WMMA tile: simdgroup columns must be a multiple of 16");
  static_assert(BLOCK_DEPTH % WMMA_SIZE == 0, "WMMA tile: block depth must be a multiple of 16");

  UZU_CONST ushort A_STRIDE_ROW = transpose_a ? 1 : THREADGROUP_LEADING_DIMENSION_A;
  UZU_CONST ushort A_STRIDE_INNER = transpose_a ? THREADGROUP_LEADING_DIMENSION_A : 1;
  UZU_CONST ushort B_STRIDE_INNER = transpose_b ? 1 : THREADGROUP_LEADING_DIMENSION_B;
  UZU_CONST ushort B_STRIDE_COL = transpose_b ? THREADGROUP_LEADING_DIMENSION_B : 1;

  __uzu_wmma_f32x8 accumulators[TILES_M * TILES_N];
  ushort lane;
  ushort block_row;
  ushort block_col;
  // First output row / column of this lane, as in the simdgroup tile: output (i, tm, tn) sits at
  // row simdgroup_row_offset + tm * 16 + 2 * i and column simdgroup_col_offset + tn * 16.
  ushort simdgroup_row_offset;
  ushort simdgroup_col_offset;

  METAL_FUNC ThreadgroupTile(const thread ThreadContext& thread_context) {
    lane = thread_context.simd_lane_id;
    block_row = SIMDGROUP_ROWS * (thread_context.simdgroup_index / SIMDGROUPS_PER_COLUMN);
    block_col = SIMDGROUP_COLS * (thread_context.simdgroup_index % SIMDGROUPS_PER_COLUMN);
    simdgroup_row_offset = block_row + lane / WMMA_SIZE;
    simdgroup_col_offset = block_col + lane % WMMA_SIZE;
#pragma unroll
    for (int tile = 0; tile < TILES_M * TILES_N; ++tile) {
      accumulators[tile] = __uzu_wmma_f32x8(0.0f);
    }
  }

  // 16 K values of one operand row / column. Contiguous K in a staged tile whose rows are 16-byte aligned
  // (leading dimension a multiple of 8 elements; K steps of 16) loads as two 16-byte LDS reads instead of
  // sixteen 2-byte ones.
  template <ushort STRIDE, ushort LEADING_DIMENSION>
  static METAL_FUNC __uzu_wmma_bf16x16 load_operand(const threadgroup bfloat* source) {
    __uzu_wmma_bf16x16 operand;
    if constexpr (STRIDE == 1 && LEADING_DIMENSION % 8 == 0) {
      typedef short __attribute__((ext_vector_type(8))) short8;
      const threadgroup short8* vectors = reinterpret_cast<const threadgroup short8*>(source);
      const short8 low = vectors[0];
      const short8 high = vectors[1];
      operand = __builtin_shufflevector(low, high, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    } else {
#pragma unroll
      for (int j = 0; j < WMMA_SIZE; ++j) {
        operand[j] = __builtin_bit_cast(short, source[j * STRIDE]);
      }
    }
    return operand;
  }

  METAL_FUNC void matmul(const threadgroup bfloat* a_shared, const threadgroup bfloat* b_shared) {
    const ushort lane16 = lane % WMMA_SIZE;
#pragma unroll
    for (ushort k0 = 0; k0 < BLOCK_DEPTH; k0 += WMMA_SIZE) {
      __uzu_wmma_bf16x16 a_operands[TILES_M];
      __uzu_wmma_bf16x16 b_operands[TILES_N];
#pragma unroll
      for (int tm = 0; tm < TILES_M; ++tm) {
        const int row = tm * WMMA_SIZE + lane16;
        const threadgroup bfloat* source = a_shared + (block_row + row) * A_STRIDE_ROW + k0 * A_STRIDE_INNER;
        if (row >= SIMDGROUP_ROWS) {
          a_operands[tm] = __uzu_wmma_bf16x16(0);
        } else {
          a_operands[tm] = load_operand<A_STRIDE_INNER, THREADGROUP_LEADING_DIMENSION_A>(source);
        }
      }
#pragma unroll
      for (int tn = 0; tn < TILES_N; ++tn) {
        const int col = block_col + tn * WMMA_SIZE + lane16;
        const threadgroup bfloat* source = b_shared + col * B_STRIDE_COL + k0 * B_STRIDE_INNER;
        b_operands[tn] = load_operand<B_STRIDE_INNER, THREADGROUP_LEADING_DIMENSION_B>(source);
      }
#pragma unroll
      for (int tm = 0; tm < TILES_M; ++tm) {
#pragma unroll
        for (int tn = 0; tn < TILES_N; ++tn) {
          accumulators[tm * TILES_N + tn] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
              a_operands[tm],
              b_operands[tn],
              accumulators[tm * TILES_N + tn]
          );
        }
      }
    }
  }

  // fn(row_offset, col_offset, k, value) for each output this lane owns, relative to
  // (simdgroup_row_offset, simdgroup_col_offset); rows past the simdgroup block are skipped.
  template <class Fn>
  METAL_FUNC void for_each_output(Fn fn) {
    // vector elements cannot bind to references: address the accumulators as floats
    float* data = reinterpret_cast<float*>(accumulators);
    const ushort lane_row = lane / WMMA_SIZE;
#pragma unroll
    for (int tm = 0; tm < TILES_M; ++tm) {
#pragma unroll
      for (int tn = 0; tn < TILES_N; ++tn) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const ushort row_offset = ushort(tm * WMMA_SIZE + 2 * i);
          if (row_offset + lane_row < SIMDGROUP_ROWS) {
            fn(row_offset, ushort(tn * WMMA_SIZE), ushort(0), data[(tm * TILES_N + tn) * 8 + i]);
          }
        }
      }
    }
  }

  METAL_FUNC void store_result(device DT* D, const int leading_dimension_d) {
    D += simdgroup_row_offset * leading_dimension_d + simdgroup_col_offset;
    for_each_output([&](ushort row_offset, ushort col_offset, ushort, thread AccumulatorType& value) {
      D[row_offset * leading_dimension_d + col_offset] = static_cast<DT>(Epilogue::apply(value));
    });
  }

  METAL_FUNC void store_result_safe(device DT* D, const int leading_dimension_d, short2 destination_tile_dimensions) {
    D += simdgroup_row_offset * leading_dimension_d + simdgroup_col_offset;
    destination_tile_dimensions -= short2(simdgroup_col_offset, simdgroup_row_offset);
    for_each_output([&](ushort row_offset, ushort col_offset, ushort, thread AccumulatorType& value) {
      if (short(row_offset) < destination_tile_dimensions.y && short(col_offset) < destination_tile_dimensions.x) {
        D[row_offset * leading_dimension_d + col_offset] = static_cast<DT>(Epilogue::apply(value));
      }
    });
  }

  template <typename EpilogueOp>
  METAL_FUNC void apply_epilogue(const device DT* C, const int ld_c, const int cstride_c, thread const EpilogueOp& op) {
    const device DT* c_ptr = C + simdgroup_row_offset * ld_c + simdgroup_col_offset * cstride_c;
    for_each_output([&](ushort row_offset, ushort col_offset, ushort k, thread AccumulatorType& v) {
      v = op.apply(v, static_cast<AccumulatorType>(c_ptr[row_offset * ld_c + (col_offset + k) * cstride_c]));
    });
  }

  template <typename EpilogueOp>
  METAL_FUNC void apply_epilogue_safe(
      const device DT* C,
      const int ld_c,
      const int cstride_c,
      short2 tile_dimensions,
      thread const EpilogueOp& op
  ) {
    const device DT* c_ptr = C + simdgroup_row_offset * ld_c + simdgroup_col_offset * cstride_c;
    tile_dimensions -= short2(simdgroup_col_offset, simdgroup_row_offset);
    for_each_output([&](ushort row_offset, ushort col_offset, ushort k, thread AccumulatorType& v) {
      if (short(row_offset) < tile_dimensions.y && short(col_offset + k) < tile_dimensions.x) {
        v = op.apply(v, static_cast<AccumulatorType>(c_ptr[row_offset * ld_c + (col_offset + k) * cstride_c]));
      }
    });
  }

  METAL_FUNC void apply_bias(const device bfloat* bias) {
    const device bfloat* bias_ptr = bias + simdgroup_col_offset;
    for_each_output([&](ushort, ushort col_offset, ushort k, thread AccumulatorType& v) {
      v += static_cast<AccumulatorType>(bias_ptr[col_offset + k]);
    });
  }

  METAL_FUNC void apply_bias_safe(const device bfloat* bias, short2 tile_dimensions) {
    const device bfloat* bias_ptr = bias + simdgroup_col_offset;
    tile_dimensions -= short2(simdgroup_col_offset, simdgroup_row_offset);
    for_each_output([&](ushort row_offset, ushort col_offset, ushort k, thread AccumulatorType& v) {
      if (short(row_offset) < tile_dimensions.y && short(col_offset + k) < tile_dimensions.x) {
        v += static_cast<AccumulatorType>(bias_ptr[col_offset + k]);
      }
    });
  }
};

} // namespace matmul
} // namespace uzu
