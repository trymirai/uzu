import definitions;
import common.activation;
import common.matmul;
import generated.gemm;
import matmul.common.simdgroup_multiply_accumulate;
import matmul.gemm.common.gemm_tiling;
import matmul.gemm.common.operands;
import matmul.gemm.common.simdgroup_mma_core;
import matmul.common.loader;
[[SpecializationConstant]] public const uint __dsl_Gemm_shared_a_elements = 1;
[[SpecializationConstant]] public const uint __dsl_Gemm_shared_b_elements = 1;

// One typed array per operand. Specialization sets its active Metal tile's size before pipeline validation.
public struct GemmStorage<T, let ROLE : int> : IThreadgroupStorage<T> {
  static groupshared T data[ROLE == 0 ? __dsl_Gemm_shared_a_elements : __dsl_Gemm_shared_b_elements];
  public static T read(uint index) { return data[index]; }
  public static void write(uint index, T value) { data[index] = value; }
}


// Metal non-MXU GEMM.

[[Kernel]]
[[Precondition("new", "NUM_SIMDGROUPS == (if tiling as u32 == 0 { 1 } else { 4 })")]]
[[Precondition("new", "(tiling as u32) < 5")]]
[[Precondition("new", "alignment < 8")]]
[[Precondition("new", "shared_a_elements == [8, 64, 64, 64, 32][tiling as usize] * ([32, 32, 16, 32, 32][tiling as usize] + 16 / A.size_in_bytes() as u32)")]]
[[Precondition("new", "shared_b_elements == [32, 32, 64, 64, 32][tiling as usize] * ([32, 32, 16, 32, 32][tiling as usize] + 16 / B.size_in_bytes() as u32)")]]
[[Precondition("encode", "(alignment & 1) == 0 || m.is_multiple_of([8, 64, 64, 64, 32][tiling as usize])")]]
[[Precondition("encode", "(alignment & 2) == 0 || n.is_multiple_of([32, 32, 64, 64, 32][tiling as usize])")]]
[[Precondition("encode", "(alignment & 4) == 0 || k.is_multiple_of([32, 32, 16, 32, 32][tiling as usize])")]]
public func Gemm<A: IModelFloat, B: IModelFloat, D: IModelFloat>(
  Ptr<B, Access.Read> b,
  Ptr<A, Access.Read> a,
  Ptr<D, Access.ReadWrite> d,
  [[Optional("has_bias")]] Ptr<B, Access.Read> bias,
  uint k,
  uint n,
  uint m,
  [[Optional("has_scale")]] float ab_scale,
  [[Optional("has_soft_cap")]] float soft_cap,
  [[Groups("n.div_ceil([32, 32, 64, 64, 32][tiling as usize])")]] uint column_tile,
  [[Groups("m.div_ceil([8, 64, 64, 64, 32][tiling as usize])")]] uint row_tile,
  [[Threads("32")]] uint simd_lane,
  [[Threads("NUM_SIMDGROUPS")]] uint simd_group,
  [[Specialize]] bool has_scale,
  [[Specialize]] bool accumulate,
  [[Specialize]] bool has_bias,
  [[Specialize]] bool has_soft_cap,
  [[Specialize]] GemmTiling tiling,
  [[Specialize]] uint alignment,
  [[Specialize]] uint NUM_SIMDGROUPS,
  [[Specialize]] uint shared_a_elements,
  [[Specialize]] uint shared_b_elements,
) {
  let tid = simd_group * 32 + simd_lane;

  let thread_context = simdgroup_thread_context(tid, NUM_SIMDGROUPS);
  DenseOperand<A> left = { a };
  FullPrecisionRight<B> right = { b, k };
  GemmParams params = { m, n, k, k, n, ab_scale, soft_cap, has_scale, accumulate, has_bias, has_soft_cap, alignment };
  gemm_simdgroup<A, B, D, FullPrecisionRight<B>, GemmStorage<A, 0>, GemmStorage<B, 1>>(
    tiling, left, right, d, bias, params, uint2(column_tile, row_tile), thread_context
  );
}
