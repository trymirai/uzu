import definitions;
import common.matmul;
import generated.gemm;
import generated.quantization;
import generated.quantization_method;
import matmul.common.simdgroup_multiply_accumulate;
import matmul.gemm.common.gemm_tiling;
import matmul.gemm.common.operands;
import matmul.gemm.common.simdgroup_mma_core;
import matmul.common.loader;
[[SpecializationConstant]] public const uint __dsl_QuantizedGemm_shared_a_elements = 1;
[[SpecializationConstant]] public const uint __dsl_QuantizedGemm_shared_b_elements = 1;

// One typed array per operand. Specialization sets its active Metal tile's size before pipeline validation.
public struct QuantizedGemmStorage<T, let ROLE : int> : IThreadgroupStorage<T> {
  static groupshared T data[ROLE == 0 ? __dsl_QuantizedGemm_shared_a_elements : __dsl_QuantizedGemm_shared_b_elements];
  public static T read(uint index) { return data[index]; }
  public static void write(uint index, T value) { data[index] = value; }
}


[[SpecializationConstant]] public const bool __dsl_QuantizedGemm_METAL_LAYOUT = false;

// The reader-based route reserves its storage only when selected.
public struct QuantizedGemmCanonicalStorage : IThreadgroupStorage<float> {
  static groupshared float data[__dsl_QuantizedGemm_METAL_LAYOUT ? 1 : 2048];
  public static float read(uint index) { return data[index]; }
  public static void write(uint index, float value) { data[index] = value; }
}

// QuantizedGemm: U4/U8 quantized transposed B through the port of Metal's simdgroup GEMM (matmul.gemm.common), its
// staged loaders dequantizing each slice into threadgroup memory in B arithmetic, in its own module so the dense module's
// pipelines do not carry its entry points. tiling and alignment are as for Gemm. Metal's loaders require a group of
// whole K slices; U4 rows start on byte boundaries, so U4 K is even.

[[Kernel]]
[[Precondition("new", "group_size > 0")]]
[[Precondition("new", "NUM_SIMDGROUPS == (if !METAL_LAYOUT { 8 } else if tiling as u32 == 0 { 1 } else { 4 })")]]
[[Precondition("new", "(tiling as u32) < 5")]]
[[Precondition("new", "!METAL_LAYOUT || (alignment < 8)")]]
[[Precondition("new", "shared_a_elements == [8, 64, 64, 64, 32][tiling as usize] * ([32, 32, 16, 32, 32][tiling as usize] + 16 / A.size_in_bytes() as u32)")]]
[[Precondition("new", "shared_b_elements == [32, 32, 64, 64, 32][tiling as usize] * ([32, 32, 16, 32, 32][tiling as usize] + 16 / B.size_in_bytes() as u32)")]]
[[Precondition("new", "!METAL_LAYOUT || (group_size > 0 && group_size.is_multiple_of([32, 32, 16, 32, 32][tiling as usize]))")]]
[[Precondition("new", "quantization_mode == QuantizationMode::U4 || quantization_mode == QuantizationMode::U8")]]
[[Precondition("encode", "!METAL_LAYOUT || (quantization_mode != QuantizationMode::U4 || k.is_multiple_of(2))")]]
[[Precondition("encode", "!METAL_LAYOUT || ((alignment & 1) == 0 || m.is_multiple_of([8, 64, 64, 64, 32][tiling as usize]))")]]
[[Precondition("encode", "!METAL_LAYOUT || ((alignment & 2) == 0 || n.is_multiple_of([32, 32, 64, 64, 32][tiling as usize]))")]]
[[Precondition("encode", "!METAL_LAYOUT || ((alignment & 4) == 0 || k.is_multiple_of([32, 32, 16, 32, 32][tiling as usize]))")]]
public func QuantizedGemm<A: IModelFloat, B: IModelFloat, D: IModelFloat>(
  Ptr<uint8_t, Access.Read> codes,
  Ptr<B, Access.Read> scales,
  [[Optional("quantization_method == QuantizationMethod::ScaleBias")]] Ptr<B, Access.Read> biases,
  [[Optional("quantization_method == QuantizationMethod::ScaleZeroPoint")]] Ptr<uint8_t, Access.Read> zero_points,
  Ptr<A, Access.Read> a,
  Ptr<D, Access.ReadWrite> d,
  [[Optional("has_bias")]] Ptr<B, Access.Read> bias,
  uint k,
  uint n,
  uint m,
  uint scale_output_stride,
  uint scale_group_stride,
  [[Optional("quantization_method == QuantizationMethod::ScaleZeroPoint")]] uint zero_point_output_stride,
  [[Optional("quantization_method == QuantizationMethod::ScaleZeroPoint")]] uint zero_point_group_stride,
  [[Optional("has_scale")]] float ab_scale,
  [[Optional("has_soft_cap")]] float soft_cap,
  [[Groups("n.div_ceil(if METAL_LAYOUT { [32, 32, 64, 64, 32][tiling as usize] } else { 64 })")]] uint column_tile,
  [[Groups("m.div_ceil(if METAL_LAYOUT { [8, 64, 64, 64, 32][tiling as usize] } else { 64 })")]] uint row_tile,
  [[Threads("32")]] uint simd_lane,
  [[Threads("NUM_SIMDGROUPS")]] uint simd_group,
  [[Specialize]] bool has_scale,
  [[Specialize]] bool accumulate,
  [[Specialize]] bool has_bias,
  [[Specialize]] bool has_soft_cap,
  [[Specialize]] QuantizationMode quantization_mode,
  [[Specialize]] QuantizationMethod quantization_method,
  [[Specialize]] bool signed_codes,
  [[Specialize]] uint group_size,
  [[Specialize]] GemmTiling tiling,
  [[Specialize]] uint alignment,
  [[Specialize]] uint NUM_SIMDGROUPS,
  [[Specialize]] uint shared_a_elements,
  [[Specialize]] uint shared_b_elements,
  [[Specialize]] bool METAL_LAYOUT,
) {
  let tid = simd_group * 32 + simd_lane;


  QuantizedWeights<B> weights = {
    codes, scales, biases, zero_points, group_size, scale_output_stride, scale_group_stride,
    zero_point_output_stride, zero_point_group_stride, quantization_mode == QuantizationMode.U4, signed_codes,
    quantization_method
  };
  DenseOperand<A> left = { a };
  if (!METAL_LAYOUT) {
    gemm_body<DenseOperand<A>, B, D, QuantizedWeights<B>, QuantizedGemmCanonicalStorage>(weights, left, d, bias, k, n, m, ab_scale, soft_cap, column_tile, row_tile,
      tid, has_scale, accumulate, has_bias, has_soft_cap);
    return;
  }
  let thread_context = simdgroup_thread_context(tid, NUM_SIMDGROUPS);
  GemmParams params = { m, n, k, k, n, ab_scale, soft_cap, has_scale, accumulate, has_bias, has_soft_cap, alignment };
  let threadgroup_position = uint2(column_tile, row_tile);
  if (quantization_method == QuantizationMethod.ScaleBias) {
    ScaleBiasRight<B> right = { weights, k };
    gemm_simdgroup<A, B, D, ScaleBiasRight<B>, QuantizedGemmStorage<A, 0>, QuantizedGemmStorage<B, 1>>(
      tiling, left, right, d, bias, params, threadgroup_position, thread_context
    );
  } else {
    ScaleZeroPointRight<B> right = { weights, k };
    gemm_simdgroup<A, B, D, ScaleZeroPointRight<B>, QuantizedGemmStorage<A, 0>, QuantizedGemmStorage<B, 1>>(
      tiling, left, right, d, bias, params, threadgroup_position, thread_context
    );
  }
}
