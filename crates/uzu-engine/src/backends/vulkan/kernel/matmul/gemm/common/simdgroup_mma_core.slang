import definitions;
import common.matmul;
import common.storage_address;
import matmul.common.defines;
import matmul.common.loader;
import matmul.common.simdgroup_multiply_accumulate;
import matmul.common.threadgroup_tile;
import matmul.gemm.common.operands;

// Counterpart of Metal gemm/common/simdgroup_mma_core.h's SimdgroupMmaCore: one threadgroup computes a
// THREADGROUP_BLOCK_M x THREADGROUP_BLOCK_N tile of D = epilogue(A B) from full-precision A and the right operand B,
// staging each THREADGROUP_BLOCK_K slice of both in threadgroup memory (rows padded by 16 bytes of their storage type)
// and accumulating it in FP32 8 x 8 simdgroup fragments; full K slices load unsafe where M or N is aligned, a K tail
// loads safe, then the epilogue (scale and prior D), bias and store, unsafe only where M and N are both aligned.
//
// Alignment is a uniform specialization rather than Metal's integral_constant dispatch. Exceptional operands replay
// the CPU order before the epilogue. Split K, Morton tile order and fused RHT remain to be ported.

public static const uint GEMM_ALIGNMENT_M = 1;
public static const uint GEMM_ALIGNMENT_N = 2;
public static const uint GEMM_ALIGNMENT_K = 4;

// Metal's GemmParams and output transform as the Vulkan kernels pass them; alignment is GemmAlignment's bits.
public struct GemmParams {
  public uint M;
  public uint N;
  public uint K;
  public uint leading_dimension_a;
  public uint leading_dimension_d;
  public float ab_scale;
  public float soft_cap;
  public bool has_scale;
  public bool accumulate;
  public bool has_bias;
  public bool has_soft_cap;
  public uint alignment;
}

uint simdgroup_mma_k_loop<
  A : IModelFloat,
  B : IModelFloat,
  BLoader : IThreadgroupLoader<B>,
  SharedA : IThreadgroupStorage<A>,
  SharedB : IThreadgroupStorage<B>,
  let THREADGROUP_BLOCK_M : int,
  let THREADGROUP_BLOCK_N : int,
  let THREADGROUP_BLOCK_K : int,
  let SIMDGROUPS_PER_ROW : int,
  let SIMDGROUPS_PER_COLUMN : int
>(
  uint gemm_alignment,
  int aligned_k_iterations,
  inout ThreadgroupLoader<A, false> loader_a,
  inout BLoader loader_b,
  inout ThreadgroupTile<
    A, B, THREADGROUP_BLOCK_M, THREADGROUP_BLOCK_N, THREADGROUP_BLOCK_K, SIMDGROUPS_PER_ROW, SIMDGROUPS_PER_COLUMN
  > accumulator,
  int tile_block_rows,
  int tile_block_cols,
  int leftover_block_depth,
  bool participating,
  ThreadContext thread_context,
) {
  let tile_dimensions_a = int2(THREADGROUP_BLOCK_K, tile_block_rows);
  let tile_dimensions_b = int2(THREADGROUP_BLOCK_K, tile_block_cols);

  for (int k = 0; k < aligned_k_iterations; k++) {
    GroupMemoryBarrierWithGroupSync();
    if (participating) {
      if ((gemm_alignment & GEMM_ALIGNMENT_M) != 0) {
        loader_a.load_unsafe<SharedA>();
      } else {
        loader_a.load_safe<SharedA>(tile_dimensions_a);
      }
      if ((gemm_alignment & GEMM_ALIGNMENT_N) != 0) {
        loader_b.load_unsafe<SharedB>();
      } else {
        loader_b.load_safe<SharedB>(tile_dimensions_b);
      }
    }

    GroupMemoryBarrierWithGroupSync();
    accumulator.matmul<SharedA, SharedB>();

    loader_a.next();
    loader_b.next();
  }

  if ((gemm_alignment & GEMM_ALIGNMENT_K) == 0 && leftover_block_depth > 0) {
    GroupMemoryBarrierWithGroupSync();

    let last_tile_dimensions_a = int2(leftover_block_depth, tile_block_rows);
    let last_tile_dimensions_b = int2(leftover_block_depth, tile_block_cols);

    if (participating) {
      loader_a.load_safe<SharedA>(last_tile_dimensions_a);
      loader_b.load_safe<SharedB>(last_tile_dimensions_b);
    }

    GroupMemoryBarrierWithGroupSync();
    accumulator.matmul<SharedA, SharedB>();
  }
  return matmul_workgroup_fields(matmul_merge_fields(loader_a.exponent_fields(), loader_b.exponent_fields()));
}

void simdgroup_mma_finalize<
  A : IModelFloat,
  B : IModelFloat,
  D : IModelFloat,
  R : IRightOperand<B>,
  let THREADGROUP_BLOCK_M : int,
  let THREADGROUP_BLOCK_N : int,
  let THREADGROUP_BLOCK_K : int,
  let SIMDGROUPS_PER_ROW : int,
  let SIMDGROUPS_PER_COLUMN : int
>(
  inout ThreadgroupTile<
    A, B, THREADGROUP_BLOCK_M, THREADGROUP_BLOCK_N, THREADGROUP_BLOCK_K, SIMDGROUPS_PER_ROW, SIMDGROUPS_PER_COLUMN
  > accumulator,
  uint gemm_alignment,
  uint fields,
  DenseOperand<A> left,
  R right,
  Ptr<D, Access.ReadWrite> d,
  Ptr<B, Access.Read> bias_block,
  GemmParams params,
  uint block_row,
  uint block_col,
  int tile_block_rows,
  int tile_block_cols,
) {
  let tile_rows = THREADGROUP_BLOCK_M / (8 * SIMDGROUPS_PER_ROW);
  let tile_cols = THREADGROUP_BLOCK_N / (8 * SIMDGROUPS_PER_COLUMN);
  let aligned = (gemm_alignment & GEMM_ALIGNMENT_M) != 0 && (gemm_alignment & GEMM_ALIGNMENT_N) != 0;
  [unroll]
  for (int i = 0; i < tile_rows; i++) {
    let row = accumulator.simdgroup_row_offset + i * 8 * SIMDGROUPS_PER_ROW;
    [unroll]
    for (int j = 0; j < tile_cols; j++) {
      [unroll]
      for (int element = 0; element < 2; element++) {
        let col = accumulator.simdgroup_col_offset + j * 8 * SIMDGROUPS_PER_COLUMN + element;
        if (aligned || (row < tile_block_rows && col < tile_block_cols)) {
          let output = element_pointer(d, element_index(uint(row), params.leading_dimension_d, uint(col)));
          let prior = params.accumulate ? output[0].toFloat() : 0.0;
          let bias = params.has_bias ? bias_block[uint(col)].toFloat() : 0.0;
          var dot = accumulator.c_fragment.fragment_data[i * tile_cols + j][element];
          if ((asuint(dot) & 0x7f800000u) == 0x7f800000u || matmul_mode(fields, params.K, params.has_scale,
            params.ab_scale, params.accumulate, prior, params.has_bias, bias) != MATMUL_FAST) {
            dot = matmul_replay(left, block_row + uint(row), right, block_col + uint(col), params.K);
          }
          accumulator.c_fragment.fragment_data[i * tile_cols + j][element] = matmul_epilogue(dot, params.has_scale,
            params.ab_scale, params.accumulate, prior, params.has_bias, bias, params.has_soft_cap, params.soft_cap);
        }
      }
    }
  }

  if (aligned) {
    accumulator.store_result(d, params.leading_dimension_d);
  } else {
    accumulator.store_result_safe(d, params.leading_dimension_d, int2(tile_block_cols, tile_block_rows));
  }
}

// The tile at threadgroup_position (column tile, row tile). Every invocation of the workgroup must call it.
public void simdgroup_mma_core_run<
  A : IModelFloat,
  B : IModelFloat,
  D : IModelFloat,
  R : IRightOperand<B>,
  SharedA : IThreadgroupStorage<A>,
  SharedB : IThreadgroupStorage<B>,
  let THREADGROUP_BLOCK_M : int,
  let THREADGROUP_BLOCK_N : int,
  let THREADGROUP_BLOCK_K : int,
  let SIMDGROUPS_PER_ROW : int,
  let SIMDGROUPS_PER_COLUMN : int
>(
  DenseOperand<A> left,
  R right,
  Ptr<D, Access.ReadWrite> d,
  Ptr<B, Access.Read> output_bias,
  GemmParams params,
  uint2 threadgroup_position,
  ThreadContext thread_context,
) {
  let padding_a = 16 / int(sizeof(A));
  let padding_b = 16 / int(sizeof(B));
  let shared_stride_a = THREADGROUP_BLOCK_K + padding_a;
  let shared_stride_b = THREADGROUP_BLOCK_K + padding_b;
  let threadgroup_threads = SIMDGROUPS_PER_ROW * SIMDGROUPS_PER_COLUMN * int(SIMD_SIZE);
  let participating = int(thread_context.thread_index) < threadgroup_threads;

  let block_row = threadgroup_position.y * uint(THREADGROUP_BLOCK_M);
  let block_col = threadgroup_position.x * uint(THREADGROUP_BLOCK_N);

  let d_block = element_pointer(d, element_index(block_row, params.leading_dimension_d, block_col));

  var loader_a = ThreadgroupLoader<A, false>(
    element_pointer(left.values, element_index(block_row, params.leading_dimension_a, 0)),
    int(params.leading_dimension_a),
    THREADGROUP_BLOCK_M,
    THREADGROUP_BLOCK_K,
    shared_stride_a,
    true,
    threadgroup_threads,
    thread_context
  );
  var accumulator = ThreadgroupTile<
    A, B, THREADGROUP_BLOCK_M, THREADGROUP_BLOCK_N, THREADGROUP_BLOCK_K, SIMDGROUPS_PER_ROW, SIMDGROUPS_PER_COLUMN
  >(thread_context);

  let tile_block_rows = min(THREADGROUP_BLOCK_M, int(params.M) - int(block_row));
  let tile_block_cols = min(THREADGROUP_BLOCK_N, int(params.N) - int(block_col));
  let aligned_inner_iterations = int(params.K) / THREADGROUP_BLOCK_K;
  let leftover_block_depth = int(params.K) - aligned_inner_iterations * THREADGROUP_BLOCK_K;
  let bias_block = element_pointer(output_bias, uint2(block_col, 0));

  // Metal's dynamic_alignment_mask; its all_aligned fast path selects the same mask.
  let gemm_alignment =
    (params.alignment | (tile_block_rows == THREADGROUP_BLOCK_M ? GEMM_ALIGNMENT_M : 0u)
     | (tile_block_cols == THREADGROUP_BLOCK_N ? GEMM_ALIGNMENT_N : 0u))
    & (GEMM_ALIGNMENT_M | GEMM_ALIGNMENT_N | GEMM_ALIGNMENT_K);

  var loader_b = right.make_loader(
    block_col, THREADGROUP_BLOCK_N, THREADGROUP_BLOCK_K, shared_stride_b, threadgroup_threads, thread_context
  );
  let fields = simdgroup_mma_k_loop<
    A, B, R.Loader, SharedA, SharedB, THREADGROUP_BLOCK_M, THREADGROUP_BLOCK_N, THREADGROUP_BLOCK_K, SIMDGROUPS_PER_ROW,
    SIMDGROUPS_PER_COLUMN
  >(
    gemm_alignment,
    aligned_inner_iterations,
    loader_a,
    loader_b,
    accumulator,
    tile_block_rows,
    tile_block_cols,
    leftover_block_depth,
    participating,
    thread_context
  );
  if (participating) {
    simdgroup_mma_finalize<
      A, B, D, R, THREADGROUP_BLOCK_M, THREADGROUP_BLOCK_N, THREADGROUP_BLOCK_K, SIMDGROUPS_PER_ROW, SIMDGROUPS_PER_COLUMN
    >(
      accumulator,
      gemm_alignment,
      fields,
      left,
      right,
      d_block,
      bias_block,
      params,
      block_row,
      block_col,
      tile_block_rows,
      tile_block_cols
    );
  }
}
