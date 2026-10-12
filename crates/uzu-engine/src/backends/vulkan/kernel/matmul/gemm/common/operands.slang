import definitions;
import common.matmul;
import matmul.common.loader;
import matmul.common.simdgroup_multiply_accumulate;
import matmul.gemm.common.quant_scale_bias;
import matmul.gemm.common.quant_scale_zero_point;
import matmul.gemm.common.schedules.staged;

// Counterpart of Metal gemm/common/operands.h for the simdgroup core: the right operand B as stored, full precision or
// U4/U8 with its scheme, the loader that stages it (Metal's compile-time choice in SimdgroupMmaCore::run) and, as an
// IMatmulOperand, the CPU's reading of one element for matmul_replay. The left operand of the simdgroup core is always
// full precision (Metal static_asserts !Left::QUANTIZED), common.matmul's DenseOperand. B is always stored as N rows of K
// (Metal's TRANSPOSE_RIGHT).
public interface IRightOperand<B> : IMatmulOperand {
  associatedtype Loader : IThreadgroupLoader<B>;

  public Loader make_loader(
    uint block_col,
    int block_n,
    int block_k,
    int shared_stride_b,
    int threadgroup_threads,
    ThreadContext thread_context,
  );
}

// Full-precision B, element (n, k) at n leading_dimension_b + k.
public struct FullPrecisionRight<B : IModelFloat> : IRightOperand<B> {
  public Ptr<B, Access.Read> dense;
  public uint leading_dimension_b;

  public typealias Loader = ThreadgroupLoader<B, true>;

  public float value(uint row, uint inner, uint base) {
    return dense[row * leading_dimension_b + inner].toFloat();
  }

  public Loader make_loader(
    uint block_col,
    int block_n,
    int block_k,
    int shared_stride_b,
    int threadgroup_threads,
    ThreadContext thread_context,
  ) {
    return make_full_precision_loader<B>(
      dense, leading_dimension_b, block_col, block_n, block_k, shared_stride_b, threadgroup_threads, thread_context
    );
  }
}

// U4/U8 ScaleBias B of K-element rows.
public struct ScaleBiasRight<B : IModelFloat> : IRightOperand<B> {
  public QuantizedWeights<B> weights;
  public uint k;

  public typealias Loader = QuantizedBlockLoaderScaleBias<B>;

  public float value(uint row, uint inner, uint base) {
    return weights.value(row, inner, base);
  }

  public Loader make_loader(
    uint block_col,
    int block_n,
    int block_k,
    int shared_stride_b,
    int threadgroup_threads,
    ThreadContext thread_context,
  ) {
    return make_staged_loader_scale_bias<B>(
      weights, k, block_col, block_n, block_k, shared_stride_b, threadgroup_threads, thread_context
    );
  }
}

// U4/U8 ScaleZeroPoint or ScaleSymmetric B of K-element rows.
public struct ScaleZeroPointRight<B : IModelFloat> : IRightOperand<B> {
  public QuantizedWeights<B> weights;
  public uint k;

  public typealias Loader = QuantizedBlockLoaderScaleZeroPoint<B>;

  public float value(uint row, uint inner, uint base) {
    return weights.value(row, inner, base);
  }

  public Loader make_loader(
    uint block_col,
    int block_n,
    int block_k,
    int shared_stride_b,
    int threadgroup_threads,
    ThreadContext thread_context,
  ) {
    return make_staged_loader_scale_zero_point<B>(
      weights, k, block_col, block_n, block_k, shared_stride_b, threadgroup_threads, thread_context
    );
  }
}
