import definitions;
import common.matmul;
import common.storage_address;
import generated.quantization_method;
import matmul.common.loader;
import matmul.common.simdgroup_multiply_accumulate;
import matmul.gemm.common.quant_scale_bias;
import matmul.gemm.common.quant_scale_zero_point;

// Counterpart of Metal gemm/common/schedules/staged.h's B loaders for the simdgroup core: make_full_precision_loader and
// make_staged_loader, the latter split by scheme because Slang has no deduced return type. Each starts at the
// threadgroup's first output column block_col and K offset 0 (the Vulkan kernels do not split K) and stages a
// block_n x block_k tile, rows shared_stride_b apart. StagedSchedule and stage() serve only the MXU core and have no
// counterpart.

// Dense B of N rows of K (Metal's TRANSPOSE_RIGHT), row block_col at block_col leading_dimension_b.
public ThreadgroupLoader<B, true> make_full_precision_loader<B : IModelFloat>(
  Ptr<B, Access.Read> dense,
  uint leading_dimension_b,
  uint block_col,
  int block_n,
  int block_k,
  int shared_stride_b,
  int threadgroup_threads,
  ThreadContext thread_context,
) {
  return ThreadgroupLoader<B, true>(
    element_pointer(dense, element_index(block_col, leading_dimension_b, 0)),
    int(leading_dimension_b),
    block_n,
    block_k,
    shared_stride_b,
    true,
    threadgroup_threads,
    thread_context
  );
}

// Codes of row block_col at block_col K bytes_per_pack / pack_factor, scales and biases at block_col scale_output_stride.
public QuantizedBlockLoaderScaleBias<B> make_staged_loader_scale_bias<B : IModelFloat>(
  QuantizedWeights<B> right,
  uint k,
  uint block_col,
  int block_n,
  int block_k,
  int shared_stride_b,
  int threadgroup_threads,
  ThreadContext thread_context,
) {
  let bits = right.u4 ? 4 : 8;
  let row_stride = k / uint(8 / bits);
  let params_offset = element_index(block_col, right.scale_output_stride, 0);
  return QuantizedBlockLoaderScaleBias<B>(
    element_pointer(right.codes, element_index(block_col, row_stride, 0)),
    element_pointer(right.scales, params_offset),
    element_pointer(right.biases, params_offset),
    right.signed_codes,
    int(k),
    int(right.scale_group_stride),
    int(right.scale_output_stride),
    block_n,
    block_k,
    shared_stride_b,
    threadgroup_threads,
    int(right.group_size),
    bits,
    thread_context
  );
}

// make_staged_loader_scale_bias's codes and scales, zero points (ScaleZeroPoint) from index block_col
// zero_point_output_stride of their plane, or the midpoint (ScaleSymmetric).
public QuantizedBlockLoaderScaleZeroPoint<B> make_staged_loader_scale_zero_point<B : IModelFloat>(
  QuantizedWeights<B> right,
  uint k,
  uint block_col,
  int block_n,
  int block_k,
  int shared_stride_b,
  int threadgroup_threads,
  ThreadContext thread_context,
) {
  let bits = right.u4 ? 4 : 8;
  let row_stride = k / uint(8 / bits);
  return QuantizedBlockLoaderScaleZeroPoint<B>(
    element_pointer(right.codes, element_index(block_col, row_stride, 0)),
    element_pointer(right.scales, element_index(block_col, right.scale_output_stride, 0)),
    right.zero_points,
    right.signed_codes,
    int(k),
    int(right.scale_group_stride),
    int(right.scale_output_stride),
    right.zero_point_output_stride,
    right.zero_point_group_stride,
    block_col * right.zero_point_output_stride,
    block_n,
    block_k,
    shared_stride_b,
    threadgroup_threads,
    int(right.group_size),
    bits,
    right.method == QuantizationMethod.ScaleSymmetric,
    thread_context
  );
}
