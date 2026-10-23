import definitions;
import common.matmul;
import generated.gemm;
import matmul.common.simdgroup_multiply_accumulate;
import matmul.gemm.common.operands;
import matmul.gemm.common.simdgroup_mma_core;
import matmul.common.threadgroup_tile;

// The simdgroup tilings: each GemmTiling's block M x N x K and simdgroups per row x column. tiling is a specialization
// constant, so each pipeline keeps one branch.
public void gemm_simdgroup<A : IModelFloat, B : IModelFloat, D : IModelFloat, R : IRightOperand<B>, SA : IThreadgroupStorage<A>, SB : IThreadgroupStorage<B>>(
  GemmTiling tiling,
  DenseOperand<A> left,
  R right,
  Ptr<D, Access.ReadWrite> d,
  Ptr<B, Access.Read> output_bias,
  GemmParams params,
  uint2 threadgroup_position,
  ThreadContext thread_context,
) {
  if (tiling == GemmTiling.Tile8x32x32_Simdgroups1x1) {
    simdgroup_mma_core_run<A, B, D, R, SA, SB, 8, 32, 32, 1, 1>(
      left, right, d, output_bias, params, threadgroup_position, thread_context
    );
  } else if (tiling == GemmTiling.Tile64x32x32_Simdgroups2x2) {
    simdgroup_mma_core_run<A, B, D, R, SA, SB, 64, 32, 32, 2, 2>(
      left, right, d, output_bias, params, threadgroup_position, thread_context
    );
  } else if (tiling == GemmTiling.Tile64x64x16_Simdgroups2x2) {
    simdgroup_mma_core_run<A, B, D, R, SA, SB, 64, 64, 16, 2, 2>(
      left, right, d, output_bias, params, threadgroup_position, thread_context
    );
  } else if (tiling == GemmTiling.Tile64x64x32_Simdgroups2x2) {
    simdgroup_mma_core_run<A, B, D, R, SA, SB, 64, 64, 32, 2, 2>(
      left, right, d, output_bias, params, threadgroup_position, thread_context
    );
  } else if (tiling == GemmTiling.Tile32x32x32_Simdgroups2x2) {
    simdgroup_mma_core_run<A, B, D, R, SA, SB, 32, 32, 32, 2, 2>(
      left, right, d, output_bias, params, threadgroup_position, thread_context
    );
  }
}
