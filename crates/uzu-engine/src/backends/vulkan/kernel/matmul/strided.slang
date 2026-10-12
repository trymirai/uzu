import definitions;
import common.matmul;
import common.storage_address;
import matmul.gemv;

// StridedGemv and StridedGemm: matmul.gemv's and matmul.gemm's bodies over full-precision B read at element
// row b_row_stride + inner b_inner_stride, the CPU MatmulKernel's index for transposed B of any leading dimension
// (stride ld, 1) and for B that is not transposed (1, ld), in their own module so the contiguous modules' pipelines do
// not carry these entry points.

struct StridedOperand<T : IModelFloat> : IMatmulOperand {
  Ptr<T, Access.Read> values;
  uint row_stride;
  uint inner_stride;

  float value(uint row, uint inner, uint base) {
    return element_pointer(values, add_address(element_index(row, row_stride, 0), element_index(inner, inner_stride, 0)))[0].toFloat();
  }
}

[[Kernel]] func StridedGemv<A: IModelFloat, B: IModelFloat, D: IModelFloat>(
  Ptr<B, Access.Read> b,
  Ptr<A, Access.Read> a,
  Ptr<D, Access.ReadWrite> d,
  [[Optional("has_bias")]] Ptr<B, Access.Read> bias,
  [[Optional("gathered")]] Ptr<uint, Access.Read> gather,
  uint k,
  uint n,
  uint m,
  uint b_row_stride,
  uint b_inner_stride,
  [[Optional("has_scale")]] float ab_scale,
  [[Optional("has_soft_cap")]] float soft_cap,
  [[Groups("n.div_ceil(8)")]] uint tile,
  [[Groups("m")]] uint row,
  [[Threads("256")]] uint tid,
  [[Specialize]] bool has_scale,
  [[Specialize]] bool accumulate,
  [[Specialize]] bool has_bias,
  [[Specialize]] bool has_soft_cap,
  [[Specialize]] bool gathered,
) {
  StridedOperand<B> weights = { b, b_row_stride, b_inner_stride };
  DenseOperand<A> activations = { a };
  gemv_body<DenseOperand<A>, B, D>(
    weights, activations, d, bias, gather, k, n, m, ab_scale, soft_cap, tile, row, tid, has_scale, accumulate, has_bias,
    has_soft_cap, gathered);
}

[[Kernel]] func StridedGemm<A: IModelFloat, B: IModelFloat, D: IModelFloat>(
  Ptr<B, Access.Read> b,
  Ptr<A, Access.Read> a,
  Ptr<D, Access.ReadWrite> d,
  [[Optional("has_bias")]] Ptr<B, Access.Read> bias,
  uint k,
  uint n,
  uint m,
  uint b_row_stride,
  uint b_inner_stride,
  [[Optional("has_scale")]] float ab_scale,
  [[Optional("has_soft_cap")]] float soft_cap,
  [[Groups("n.div_ceil(64)")]] uint column_tile,
  [[Groups("m.div_ceil(64)")]] uint row_tile,
  [[Threads("256")]] uint tid,
  [[Specialize]] bool has_scale,
  [[Specialize]] bool accumulate,
  [[Specialize]] bool has_bias,
  [[Specialize]] bool has_soft_cap,
) {
  StridedOperand<B> weights = { b, b_row_stride, b_inner_stride };
  DenseOperand<A> activations = { a };
  gemm_body<DenseOperand<A>, B, D, StridedOperand<B>, CanonicalGemmStorage>(
    weights, activations, d, bias, k, n, m, ab_scale, soft_cap, column_tile, row_tile, tid, has_scale, accumulate,
    has_bias, has_soft_cap);
}
