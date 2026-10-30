import definitions;
import matmul.common.matmul;
import generated.gemm;
import generated.gemm_tiling;
import generated.gemm_b_layout;
import generated.quantization;
import generated.quantization_method;
import matmul.common.int8_operands;
import matmul.common.simdgroup_multiply_accumulate;
import matmul.common.tensor_tile;
import matmul.gemm.common.gemm_tiling;
import matmul.gemm.common.operands;
import matmul.gemm.common.simdgroup_mma_core;
import matmul.gemv.common.arguments;

using namespace linalg;

// D = epilogue(A B) over A [m, k] and B of n rows of k, full precision or U4/U8 codes (the B prologue). Each core is its
// own entry point so a module declares only its capabilities: Gemm (simdgroup fragments) on every device, GemmTensor
// (cooperative matrices) for 16-bit operands on tensor cores. Quantized B needs groups of whole K slices, U4 an even k.

[[SpecializationConstant]] public const uint __dsl_Gemm_shared_a_elements = 1;
[[SpecializationConstant]] public const uint __dsl_Gemm_shared_b_elements = 1;

// One typed array per operand, sized by specialization.
public struct GemmStorage<T, let ROLE : int> : IThreadgroupStorage<T> {
  static groupshared T data[ROLE == 0 ? __dsl_Gemm_shared_a_elements : __dsl_Gemm_shared_b_elements];
  public static T read(uint index) { return data[index]; }
  public static void write(uint index, T value) { data[index] = value; }
}

[[Kernel]]
[[Precondition("new", "!tiling.is_tensor() && NUM_SIMDGROUPS == tiling.simdgroups()")]]
[[Precondition("new", "alignment < 8")]]
[[Precondition("new", "shared_a_elements == tiling.scalar_staging(true, A.size_in_bytes() as u32)")]]
[[Precondition("new", "shared_b_elements == tiling.scalar_staging(false, B.size_in_bytes() as u32)")]]
[[Precondition("new", "b_prologue == GemmBPrologueKind::FullPrecision || (quantization_mode != QuantizationMode::I8 && group_size > 0 && group_size.is_multiple_of(tiling.block_k()))")]]
[[Precondition("encode", "b_prologue == GemmBPrologueKind::FullPrecision || quantization_mode != QuantizationMode::U4 || k.is_multiple_of(2)")]]
[[Precondition("encode", "(alignment & 1) == 0 || m.is_multiple_of(tiling.block_m())")]]
[[Precondition("encode", "(alignment & 2) == 0 || n.is_multiple_of(tiling.block_n())")]]
[[Precondition("encode", "(alignment & 4) == 0 || k.is_multiple_of(tiling.block_k())")]]
public func Gemm<A: IModelFloat, B: IModelFloat, D: IModelFloat>(
  Ptr<uint, Access.Read> b,
  [[Optional("b_prologue != GemmBPrologueKind::FullPrecision")]] Ptr<B, Access.Read> scales,
  [[Optional("b_prologue == GemmBPrologueKind::ScaleBiasDequant")]] Ptr<B, Access.Read> biases,
  [[Optional("b_prologue == GemmBPrologueKind::ScaleZeroPointDequant")]] Ptr<uint8_t, Access.Read> zero_points,
  Ptr<A, Access.Read> a,
  Ptr<D, Access.ReadWrite> d,
  [[Optional("has_bias")]] Ptr<B, Access.Read> bias,
  uint k,
  uint n,
  uint m,
  [[Optional("b_prologue != GemmBPrologueKind::FullPrecision")]] uint scale_output_stride,
  [[Optional("b_prologue != GemmBPrologueKind::FullPrecision")]] uint scale_group_stride,
  [[Optional("b_prologue == GemmBPrologueKind::ScaleZeroPointDequant")]] uint zero_point_output_stride,
  [[Optional("b_prologue == GemmBPrologueKind::ScaleZeroPointDequant")]] uint zero_point_group_stride,
  [[Optional("has_scale")]] float ab_scale,
  [[Optional("has_soft_cap")]] float soft_cap,
  [[Groups("n.div_ceil(tiling.block_n())")]] uint column_tile,
  [[Groups("m.div_ceil(tiling.block_m())")]] uint row_tile,
  [[Threads("32")]] uint simd_lane,
  [[Threads("NUM_SIMDGROUPS")]] uint simd_group,
  [[Specialize]] bool has_scale,
  [[Specialize]] bool has_bias,
  [[Specialize]] bool has_soft_cap,
  [[Specialize]] GemmBPrologueKind b_prologue,
  [[Specialize]] QuantizationMode quantization_mode,
  [[Specialize]] bool signed_codes,
  [[Specialize]] uint group_size,
  [[Specialize]] GemmTiling tiling,
  [[Specialize]] uint alignment,
  [[Specialize]] uint NUM_SIMDGROUPS,
  [[Specialize]] uint shared_a_elements,
  [[Specialize]] uint shared_b_elements,
) {
  ThreadContext thread_context = { simd_group * 32 + simd_lane, simd_group, simd_lane };
  DenseOperand<A> left = { a };
  GemmParams params = { m, n, k, k, n, ab_scale, soft_cap, has_scale, has_bias, has_soft_cap, alignment };
  let position = uint2(column_tile, row_tile);
  if (b_prologue == GemmBPrologueKind.FullPrecision) {
    FullPrecisionRight<B> right = { Ptr<B, Access.Read>(b), k };
    gemm_simdgroup<A, B, D, FullPrecisionRight<B>, GemmStorage<A, 0>, GemmStorage<B, 1>>(
      tiling, left, right, d, bias, params, position, thread_context
    );
  } else {
    QuantizedRight<B> right = {
      gemm_weights<B>(b, scales, biases, zero_points, group_size, scale_output_stride, scale_group_stride,
        zero_point_output_stride, zero_point_group_stride, quantization_mode, signed_codes, b_prologue, k,
        GemmBLayout.RowMajor),
      k
    };
    gemm_simdgroup<A, B, D, QuantizedRight<B>, GemmStorage<A, 0>, GemmStorage<B, 1>>(
      tiling, left, right, d, bias, params, position, thread_context
    );
  }
}

[[SpecializationConstant]] public const uint __dsl_GemmTensor_shared_words = 1;

// GemmTensor's two stages, reused by its epilogue.
public struct GemmTensorShared<N : ICoopElement> : ITensorShared<N> {
  static groupshared uint4 data[__dsl_GemmTensor_shared_words];
  public static uint4 read(uint word) { return data[word]; }
  public static void write(uint word, uint4 value) { data[word] = value; }
  public static CoopMat<N, MemoryScope.Subgroup, 16, 16, USE> load<
    let USE : CoopMatMatrixUse,
    let LAYOUT : CoopMatMatrixLayout
  >(uint word, uint stride) {
    return CoopMat<N, MemoryScope.Subgroup, 16, 16, USE>.Load<LAYOUT, uint4>(data, word, stride);
  }
  public static void store(
    CoopMat<float, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator> value,
    uint word,
    uint stride
) {
    value.Store<CoopMatMatrixLayout.RowMajor, uint4>(data, word, stride);
  }
}

// The cooperative tilings' instantiations; tiling is a specialization constant, so each pipeline keeps one branch.
void gemm_tensor_tiling<B : IModelFloat16, N : ICoopElement, D : IModelFloat>(
  GemmTiling tiling,
  DenseOperand<B> left,
  Ptr<B, Access.Read> dense,
  QuantizedWeights<B> weights,
  bool quantized,
  Ptr<D, Access.ReadWrite> d,
  Ptr<B, Access.Read> bias,
  GemmParams params,
  uint2 tile,
  uint tid,
) {
  typealias S = GemmTensorShared<N>;
  if (tiling == GemmTiling.Tile16x64x64_Simdgroups1x4x2) {
    gemm_tensor_run<B, N, D, S, 16, 64, 64, 1, 4, 2>(
      left, dense, weights, quantized, d, bias, params, tile, tid
    );
  } else if (tiling == GemmTiling.Tile32x64x64_Simdgroups1x4x2) {
    gemm_tensor_run<B, N, D, S, 32, 64, 64, 1, 4, 2>(
      left, dense, weights, quantized, d, bias, params, tile, tid
    );
  } else if (tiling == GemmTiling.Tile64x128x32_Simdgroups2x4x1) {
    gemm_tensor_run<B, N, D, S, 64, 128, 32, 2, 4, 1>(
      left, dense, weights, quantized, d, bias, params, tile, tid
    );
  } else if (tiling == GemmTiling.Tile128x128x32_Simdgroups2x4x1) {
    gemm_tensor_run<B, N, D, S, 128, 128, 32, 2, 4, 1>(
      left, dense, weights, quantized, d, bias, params, tile, tid
    );
  }
}

[[Kernel]]
[[RequiredSubgroupSize("32")]]
[[Precondition("new", "context.physical_device().capabilities.float_tensor_cores(B)")]]
[[Precondition("new", "tiling.is_tensor() && NUM_SIMDGROUPS == tiling.simdgroups()")]]
[[Precondition("new", "shared_words == tiling.tensor_staging()")]]
[[Precondition("new", "b_prologue == GemmBPrologueKind::FullPrecision || (quantization_mode != QuantizationMode::I8 && group_size > 0 && group_size.is_multiple_of(16))")]]
[[Precondition("encode", "b_prologue == GemmBPrologueKind::FullPrecision || quantization_mode != QuantizationMode::U4 || k.is_multiple_of(2)")]]
public func GemmTensor<B: IModelFloat16, D: IModelFloat>(
  Ptr<uint, Access.Read> b,
  [[Optional("b_prologue != GemmBPrologueKind::FullPrecision")]] Ptr<B, Access.Read> scales,
  [[Optional("b_prologue == GemmBPrologueKind::ScaleBiasDequant")]] Ptr<B, Access.Read> biases,
  [[Optional("b_prologue == GemmBPrologueKind::ScaleZeroPointDequant")]] Ptr<uint8_t, Access.Read> zero_points,
  Ptr<B, Access.Read> a,
  Ptr<D, Access.ReadWrite> d,
  [[Optional("has_bias")]] Ptr<B, Access.Read> bias,
  uint k,
  uint n,
  uint m,
  [[Optional("b_prologue != GemmBPrologueKind::FullPrecision")]] uint scale_output_stride,
  [[Optional("b_prologue != GemmBPrologueKind::FullPrecision")]] uint scale_group_stride,
  [[Optional("b_prologue == GemmBPrologueKind::ScaleZeroPointDequant")]] uint zero_point_output_stride,
  [[Optional("b_prologue == GemmBPrologueKind::ScaleZeroPointDequant")]] uint zero_point_group_stride,
  [[Optional("has_scale")]] float ab_scale,
  [[Optional("has_soft_cap")]] float soft_cap,
  [[Groups("n.div_ceil(tiling.block_n())")]] uint column_tile,
  [[Groups("m.div_ceil(tiling.block_m())")]] uint row_tile,
  [[Threads("32")]] uint simd_lane,
  [[Threads("NUM_SIMDGROUPS")]] uint simd_group,
  [[Specialize]] bool has_scale,
  [[Specialize]] bool has_bias,
  [[Specialize]] bool has_soft_cap,
  [[Specialize]] GemmBPrologueKind b_prologue,
  [[Specialize]] QuantizationMode quantization_mode,
  [[Specialize]] bool signed_codes,
  [[Specialize]] uint group_size,
  [[Specialize]] GemmTiling tiling,
  [[Specialize]] uint NUM_SIMDGROUPS,
  [[Specialize]] uint shared_words,
) {
  DenseOperand<B> left = { a };
  GemmParams params = { m, n, k, k, n, ab_scale, soft_cap, has_scale, has_bias, has_soft_cap, 0 };
  let dense = Ptr<B, Access.Read>(b);
  let weights = gemm_weights<B>(b, scales, biases, zero_points, group_size, scale_output_stride, scale_group_stride,
    zero_point_output_stride, zero_point_group_stride, quantization_mode, signed_codes, b_prologue, k,
    GemmBLayout.RowMajor);
  let tile = uint2(column_tile, row_tile);
  let tid = simd_group * 32 + simd_lane;
  let quantized = b_prologue != GemmBPrologueKind.FullPrecision;
  gemm_tensor_tiling<B, BFloat16, D>(tiling, left, dense, weights, quantized, d, bias, params, tile, tid);
}

[[SpecializationConstant]] public const uint __dsl_GemmInt8_shared_words = 1;
[[SpecializationConstant]] public const uint __dsl_GemmInt8K16_shared_words = 1056;
[[SpecializationConstant]] public const uint __dsl_GemmInt8K16_ROW_BLOCK = 16;
[[SpecializationConstant]] public const uint __dsl_GemmInt8Nv_shared_words = 1;

// GemmInt8's (ROLE 0) or GemmInt8Nv's (ROLE 1) two stages, and GemmInt8's per-subgroup fold scratch.
public struct GemmInt8Shared<let ROLE : int, let MMA_K : int> : IInt8Shared<MMA_K> {
  static groupshared uint4 data[ROLE == 0 ? __dsl_GemmInt8_shared_words : (ROLE == 1 ? __dsl_GemmInt8Nv_shared_words : __dsl_GemmInt8K16_shared_words)];
  public static uint4 read(uint word) { return data[word]; }
  public static void write(uint word, uint4 value) { data[word] = value; }
  public static CoopMat<int8_t, MemoryScope.Subgroup, 16, MMA_K, CoopMatMatrixUse.MatrixA> load_a(uint word, uint stride) {
    return CoopMat<int8_t, MemoryScope.Subgroup, 16, MMA_K, CoopMatMatrixUse.MatrixA>.Load<CoopMatMatrixLayout.RowMajor, uint4>(
      data, word, stride
    );
  }
  public static CoopMat<int8_t, MemoryScope.Subgroup, MMA_K, 16, CoopMatMatrixUse.MatrixB> load_b(uint word, uint stride) {
    return CoopMat<int8_t, MemoryScope.Subgroup, MMA_K, 16, CoopMatMatrixUse.MatrixB>.Load<
      CoopMatMatrixLayout.ColumnMajor,
      uint4
    >(data, word, stride);
  }
  public static void store(
    CoopMat<int, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator> value,
    uint word,
    uint stride
) {
    value.Store<CoopMatMatrixLayout.RowMajor, uint4>(data, word, stride);
  }
  public static void store_float(
    CoopMat<float, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator> value,
    uint word,
    uint stride
) {
    value.Store<CoopMatMatrixLayout.RowMajor, uint4>(data, word, stride);
  }
}

// Byte operands and INT32 results keep matrix accesses in their native component types. Output transforms reuse the
// byte staging after the operands are consumed; matrix products use the following INT32 scratch.
public struct GemmInt8K16Shared : IInt8Shared<16> {
  static const uint STAGES = 2 * (__dsl_GemmInt8K16_ROW_BLOCK + 64) * 5;
  static groupshared int8_t data[16 * STAGES];
  static groupshared int scratch[4 * (__dsl_GemmInt8K16_shared_words - STAGES)];
  public static uint4 read(uint word) {
    if (word < STAGES) {
      uint4 value = 0;
      [unroll]
      for (uint i = 0; i < 16; i++) {
        value[i / 4] |= uint(uint8_t(data[16 * word + i])) << (8 * (i % 4));
      }
      return value;
    }
    let offset = 4 * (word - STAGES);
    return asuint(int4(scratch[offset], scratch[offset + 1], scratch[offset + 2], scratch[offset + 3]));
  }
  public static void write(uint word, uint4 value) {
    [unroll]
    for (uint i = 0; i < 16; i++) { data[16 * word + i] = int8_t(value[i / 4] >> (8 * (i % 4))); }
  }
  public static CoopMat<int8_t, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixA> load_a(uint word, uint stride) {
    return CoopMat<int8_t, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixA>.Load<CoopMatMatrixLayout.RowMajor, int8_t>(data, 16 * word, 16 * stride);
  }
  public static CoopMat<int8_t, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixB> load_b(uint word, uint stride) {
    return CoopMat<int8_t, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixB>.Load<CoopMatMatrixLayout.ColumnMajor, int8_t>(data, 16 * word, 16 * stride);
  }
  public static void store(CoopMat<int, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator> value, uint word, uint stride) {
    value.Store<CoopMatMatrixLayout.RowMajor, int>(scratch, 4 * (word - STAGES), 4 * stride);
  }
  public static void store_float(CoopMat<float, MemoryScope.Subgroup, 16, 16, CoopMatMatrixUse.MatrixAccumulator> value, uint word, uint stride) {
    value.Store<CoopMatMatrixLayout.RowMajor, int>(scratch, 4 * (word - STAGES), 4 * stride);
  }
}

// The A8 body of the GemmInt8 entry points: ROW_BLOCK rows by 64 columns per workgroup, folding weight groups in
// registers where ELEMENT_FOLD.
void gemm_int8<let MMA_K : int, S : IInt8Shared<MMA_K>, B : IModelFloat, D : IModelFloat, let ELEMENT_FOLD : bool>(
  Ptr<uint, Access.Read> b,
  Ptr<B, Access.Read> scales,
  Ptr<B, Access.Read> biases,
  Ptr<uint8_t, Access.Read> zero_points,
  Ptr<int8_t, Access.Read> a,
  Ptr<float, Access.Read> a_scales,
  Ptr<int, Access.Read> a_group_sums,
  Ptr<D, Access.ReadWrite> d,
  Ptr<B, Access.Read> bias,
  Ptr<uint, Access.ReadWrite> partials,
  uint k,
  uint n,
  uint m,
  uint scale_output_stride,
  uint scale_group_stride,
  uint zero_point_output_stride,
  uint zero_point_group_stride,
  float ab_scale,
  float soft_cap,
  uint2 tile,
  uint tid,
  bool has_scale,
  bool has_bias,
  bool has_soft_cap,
  GemmBPrologueKind b_prologue,
  QuantizationMode quantization_mode,
  bool signed_codes,
  uint group_size,
  uint ROW_BLOCK,
  uint stage_depth,
  uint num_simdgroups,
  uint split_phase,
  uint splits,
  uint split_index,
  GemmBLayout b_layout,
  Ptr<int, Access.Read> rht_factors,
  bool has_rht,
) {
  let weights = gemm_weights<B>(b, scales, biases, zero_points, group_size, scale_output_stride, scale_group_stride,
    zero_point_output_stride, zero_point_group_stride, quantization_mode, signed_codes, b_prologue, k, b_layout
  );
  Int8Activations activations = { a, a_scales };
  // Under the output transform the bias adds after it.
  GemmParams params = { m, n, k, k, n, ab_scale, soft_cap, has_scale, has_bias && !has_rht, has_soft_cap, 0 };
  if (ROW_BLOCK == 16 && stage_depth == 128) {
    Int8Gemm<MMA_K, S, B, D, 16, 1, 4, ELEMENT_FOLD, 128>.run(
      activations, a_group_sums, weights, d, bias, partials, params, tile, tid, split_phase, splits, split_index,
      rht_factors, has_rht, has_bias
    );
  } else if (ROW_BLOCK == 16) {
    Int8Gemm<MMA_K, S, B, D, 16, 1, 4, ELEMENT_FOLD, 64>.run(
      activations, a_group_sums, weights, d, bias, partials, params, tile, tid, split_phase, splits, split_index,
      rht_factors, has_rht, has_bias
    );
  } else if (ROW_BLOCK == 32 && ELEMENT_FOLD && num_simdgroups == 8) {
    Int8Gemm<MMA_K, S, B, D, 32, 2, 4, ELEMENT_FOLD, 64>.run(
      activations, a_group_sums, weights, d, bias, partials, params, tile, tid, split_phase, splits, split_index,
      rht_factors, has_rht, has_bias
    );
  } else if (ROW_BLOCK == 32) {
    Int8Gemm<MMA_K, S, B, D, 32, 1, 4, ELEMENT_FOLD, 64>.run(
      activations, a_group_sums, weights, d, bias, partials, params, tile, tid, split_phase, splits, split_index,
      rht_factors, has_rht, has_bias
    );
  } else {
    Int8Gemm<MMA_K, S, B, D, 64, 4, 2, ELEMENT_FOLD, 64>.run(
      activations, a_group_sums, weights, d, bias, partials, params, tile, tid, split_phase, splits, split_index,
      rht_factors, has_rht, has_bias
    );
  }
}

// A8 activations (MatmulA::Int8Symmetric) on INT8 tensor cores against U4 codes or U8 symmetric codes. ROW_BLOCK is 16
// or 32 for verification batches, 64 beyond; a grid too narrow for the device splits K in two phases through `partials`.
[[Kernel]]
[[RequiredSubgroupSize("32")]]
[[Precondition("new", "b_prologue != GemmBPrologueKind::FullPrecision && [32, 64, 128].contains(&group_size)")]]
[[Precondition("new", "quantization_mode == QuantizationMode::U4 || (quantization_mode == QuantizationMode::U8 && b_prologue == GemmBPrologueKind::ScaleSymmetricDequant)")]]
[[Precondition("new", "[16, 32, 64].contains(&ROW_BLOCK)")]]
[[Precondition("new", "NUM_SIMDGROUPS == if ROW_BLOCK == 64 { 8 } else { 4 }")]]
[[Precondition("new", "context.physical_device().capabilities.contains(VkDeviceCapabilities::INT8_TENSOR_CORES_K32 | VkDeviceCapabilities::SUBGROUP_SIZE_32)")]]
[[Precondition("new", "shared_words == 2 * (ROW_BLOCK + 64) * 5 + 4 * 64 * (ROW_BLOCK / 16)")]]
[[Precondition("new", "split_phase < 3 && [1, 2, 4, 8].contains(&splits) && (split_phase == 0) == (splits == 1)")]]
[[Precondition("encode", "k.is_multiple_of(128 * splits)")]]
[[Precondition("encode", "b_layout == GemmBLayout::RowMajor || n.is_multiple_of(32)")]]
[[Precondition("encode", "!has_rht || n.is_multiple_of(32)")]]
public func GemmInt8<B: IModelFloat, D: IModelFloat>(
  Ptr<uint, Access.Read> b,
  Ptr<B, Access.Read> scales,
  [[Optional("b_prologue == GemmBPrologueKind::ScaleBiasDequant")]] Ptr<B, Access.Read> biases,
  [[Optional("b_prologue == GemmBPrologueKind::ScaleZeroPointDequant")]] Ptr<uint8_t, Access.Read> zero_points,
  Ptr<int8_t, Access.Read> a,
  Ptr<float, Access.Read> a_scales,
  [[Optional("b_prologue == GemmBPrologueKind::ScaleBiasDequant")]] Ptr<int, Access.Read> a_group_sums,
  Ptr<D, Access.ReadWrite> d,
  [[Optional("has_bias")]] Ptr<B, Access.Read> bias,
  [[Optional("split_phase != 0")]] Ptr<uint, Access.ReadWrite> partials,
  [[Optional("has_rht")]] Ptr<int, Access.Read> rht_factors,
  uint k,
  uint n,
  uint m,
  uint scale_output_stride,
  uint scale_group_stride,
  [[Optional("b_prologue == GemmBPrologueKind::ScaleZeroPointDequant")]] uint zero_point_output_stride,
  [[Optional("b_prologue == GemmBPrologueKind::ScaleZeroPointDequant")]] uint zero_point_group_stride,
  [[Optional("has_scale")]] float ab_scale,
  [[Optional("has_soft_cap")]] float soft_cap,
  [[Groups("n.div_ceil(64)")]] uint column_tile,
  [[Groups("m.div_ceil(ROW_BLOCK)")]] uint row_tile,
  [[Groups("if split_phase == 1 { splits } else { 1 }")]] uint split_index,
  [[Threads("32")]] uint simd_lane,
  [[Threads("NUM_SIMDGROUPS")]] uint simd_group,
  [[Specialize]] bool has_scale,
  [[Specialize]] bool has_bias,
  [[Specialize]] bool has_soft_cap,
  [[Specialize]] GemmBPrologueKind b_prologue,
  [[Specialize]] QuantizationMode quantization_mode,
  [[Specialize]] bool signed_codes,
  [[Specialize]] uint group_size,
  [[Specialize]] uint ROW_BLOCK,
  [[Specialize]] uint shared_words,
  [[Specialize]] uint split_phase,
  [[Specialize]] uint splits,
  [[Specialize]] GemmBLayout b_layout,
  [[Specialize]] uint NUM_SIMDGROUPS,
  [[Specialize]] bool has_rht,
) {
  gemm_int8<32, GemmInt8Shared<0, 32>, B, D, false>(
    b, scales, biases, zero_points, a, a_scales, a_group_sums, d, bias, partials, k, n, m, scale_output_stride,
    scale_group_stride, zero_point_output_stride, zero_point_group_stride, ab_scale, soft_cap, uint2(column_tile, row_tile),
    simd_group * 32 + simd_lane, has_scale, has_bias, has_soft_cap, b_prologue, quantization_mode, signed_codes, group_size,
    ROW_BLOCK, 64, NUM_SIMDGROUPS, split_phase, splits, split_index, b_layout, rht_factors, has_rht
  );
}

[[Kernel]]
[[RequiredSubgroupSize("32")]]
[[Precondition("new", "b_prologue != GemmBPrologueKind::FullPrecision && [32, 64, 128].contains(&group_size)")]]
[[Precondition("new", "quantization_mode == QuantizationMode::U4 || (quantization_mode == QuantizationMode::U8 && b_prologue == GemmBPrologueKind::ScaleSymmetricDequant)")]]
[[Precondition("new", "[16, 32, 64].contains(&ROW_BLOCK)")]]
[[Precondition("new", "NUM_SIMDGROUPS == if ROW_BLOCK == 64 { 8 } else { 4 }")]]
[[Precondition("new", "context.physical_device().capabilities.contains(VkDeviceCapabilities::INT8_TENSOR_CORES_K16 | VkDeviceCapabilities::SUBGROUP_SIZE_32)")]]
[[Precondition("new", "shared_words == 2 * (ROW_BLOCK + 64) * 5 + 4 * 64 * (ROW_BLOCK / 16)")]]
[[Precondition("new", "split_phase < 3 && [1, 2, 4, 8].contains(&splits) && (split_phase == 0) == (splits == 1)")]]
[[Precondition("encode", "k.is_multiple_of(128 * splits)")]]
[[Precondition("encode", "b_layout == GemmBLayout::RowMajor || n.is_multiple_of(32)")]]
[[Precondition("encode", "!has_rht || n.is_multiple_of(32)")]]
public func GemmInt8K16<B: IModelFloat, D: IModelFloat>(
  Ptr<uint, Access.Read> b,
  Ptr<B, Access.Read> scales,
  [[Optional("b_prologue == GemmBPrologueKind::ScaleBiasDequant")]] Ptr<B, Access.Read> biases,
  [[Optional("b_prologue == GemmBPrologueKind::ScaleZeroPointDequant")]] Ptr<uint8_t, Access.Read> zero_points,
  Ptr<int8_t, Access.Read> a,
  Ptr<float, Access.Read> a_scales,
  [[Optional("b_prologue == GemmBPrologueKind::ScaleBiasDequant")]] Ptr<int, Access.Read> a_group_sums,
  Ptr<D, Access.ReadWrite> d,
  [[Optional("has_bias")]] Ptr<B, Access.Read> bias,
  [[Optional("split_phase != 0")]] Ptr<uint, Access.ReadWrite> partials,
  [[Optional("has_rht")]] Ptr<int, Access.Read> rht_factors,
  uint k,
  uint n,
  uint m,
  uint scale_output_stride,
  uint scale_group_stride,
  [[Optional("b_prologue == GemmBPrologueKind::ScaleZeroPointDequant")]] uint zero_point_output_stride,
  [[Optional("b_prologue == GemmBPrologueKind::ScaleZeroPointDequant")]] uint zero_point_group_stride,
  [[Optional("has_scale")]] float ab_scale,
  [[Optional("has_soft_cap")]] float soft_cap,
  [[Groups("n.div_ceil(64)")]] uint column_tile,
  [[Groups("m.div_ceil(ROW_BLOCK)")]] uint row_tile,
  [[Groups("if split_phase == 1 { splits } else { 1 }")]] uint split_index,
  [[Threads("32")]] uint simd_lane,
  [[Threads("NUM_SIMDGROUPS")]] uint simd_group,
  [[Specialize]] bool has_scale,
  [[Specialize]] bool has_bias,
  [[Specialize]] bool has_soft_cap,
  [[Specialize]] GemmBPrologueKind b_prologue,
  [[Specialize]] QuantizationMode quantization_mode,
  [[Specialize]] bool signed_codes,
  [[Specialize]] uint group_size,
  [[Specialize]] uint ROW_BLOCK,
  [[Specialize]] uint shared_words,
  [[Specialize]] uint split_phase,
  [[Specialize]] uint splits,
  [[Specialize]] GemmBLayout b_layout,
  [[Specialize]] uint NUM_SIMDGROUPS,
  [[Specialize]] bool has_rht,
) {
  gemm_int8<16, GemmInt8K16Shared, B, D, false>(
    b, scales, biases, zero_points, a, a_scales, a_group_sums, d, bias, partials, k, n, m, scale_output_stride,
    scale_group_stride, zero_point_output_stride, zero_point_group_stride, ab_scale, soft_cap, uint2(column_tile, row_tile),
    simd_group * 32 + simd_lane, has_scale, has_bias, has_soft_cap, b_prologue, quantization_mode, signed_codes, group_size,
    ROW_BLOCK, 64, NUM_SIMDGROUPS, split_phase, splits, split_index, b_layout, rht_factors, has_rht
  );
}

// GemmInt8 folding each weight group in registers by VK_NV_cooperative_matrix2 per-element operations; the 16-row tile
// stages `stage_depth` 64 or 128 values deep, 128 needing weight groups of whole 64-value halves.
[[Kernel]]
[[RequiredSubgroupSize("32")]]
[[Precondition("new", "b_prologue != GemmBPrologueKind::FullPrecision && [32, 64, 128].contains(&group_size)")]]
[[Precondition("new", "quantization_mode == QuantizationMode::U4 || (quantization_mode == QuantizationMode::U8 && b_prologue == GemmBPrologueKind::ScaleSymmetricDequant)")]]
[[Precondition("new", "[16, 32, 64].contains(&ROW_BLOCK)")]]
[[Precondition("new", "if ROW_BLOCK == 32 { [4, 8].contains(&NUM_SIMDGROUPS) } else { NUM_SIMDGROUPS == if ROW_BLOCK == 64 { 8 } else { 4 } }")]]
[[Precondition("new", "context.physical_device().capabilities.contains(VkDeviceCapabilities::INT8_ELEMENT_FOLD | VkDeviceCapabilities::INT8_TENSOR_CORES_K32 | VkDeviceCapabilities::SUBGROUP_SIZE_32)")]]
[[Precondition("new", "stage_depth == 64 || (stage_depth == 128 && ROW_BLOCK == 16 && group_size.is_multiple_of(64))")]]
[[Precondition("new", "shared_words == 2 * (ROW_BLOCK + 64) * (stage_depth / 16 + 1)")]]
[[Precondition("new", "split_phase < 3 && [1, 2, 4, 8].contains(&splits) && (split_phase == 0) == (splits == 1)")]]
[[Precondition("encode", "k.is_multiple_of(128 * splits)")]]
[[Precondition("encode", "b_layout == GemmBLayout::RowMajor || n.is_multiple_of(32)")]]
[[Precondition("encode", "!has_rht || n.is_multiple_of(32)")]]
public func GemmInt8Nv<B: IModelFloat, D: IModelFloat>(
  Ptr<uint, Access.Read> b,
  Ptr<B, Access.Read> scales,
  [[Optional("b_prologue == GemmBPrologueKind::ScaleBiasDequant")]] Ptr<B, Access.Read> biases,
  [[Optional("b_prologue == GemmBPrologueKind::ScaleZeroPointDequant")]] Ptr<uint8_t, Access.Read> zero_points,
  Ptr<int8_t, Access.Read> a,
  Ptr<float, Access.Read> a_scales,
  [[Optional("b_prologue == GemmBPrologueKind::ScaleBiasDequant")]] Ptr<int, Access.Read> a_group_sums,
  Ptr<D, Access.ReadWrite> d,
  [[Optional("has_bias")]] Ptr<B, Access.Read> bias,
  [[Optional("split_phase != 0")]] Ptr<uint, Access.ReadWrite> partials,
  [[Optional("has_rht")]] Ptr<int, Access.Read> rht_factors,
  uint k,
  uint n,
  uint m,
  uint scale_output_stride,
  uint scale_group_stride,
  [[Optional("b_prologue == GemmBPrologueKind::ScaleZeroPointDequant")]] uint zero_point_output_stride,
  [[Optional("b_prologue == GemmBPrologueKind::ScaleZeroPointDequant")]] uint zero_point_group_stride,
  [[Optional("has_scale")]] float ab_scale,
  [[Optional("has_soft_cap")]] float soft_cap,
  [[Groups("n.div_ceil(64)")]] uint column_tile,
  [[Groups("m.div_ceil(ROW_BLOCK)")]] uint row_tile,
  [[Groups("if split_phase == 1 { splits } else { 1 }")]] uint split_index,
  [[Threads("32")]] uint simd_lane,
  [[Threads("NUM_SIMDGROUPS")]] uint simd_group,
  [[Specialize]] bool has_scale,
  [[Specialize]] bool has_bias,
  [[Specialize]] bool has_soft_cap,
  [[Specialize]] GemmBPrologueKind b_prologue,
  [[Specialize]] QuantizationMode quantization_mode,
  [[Specialize]] bool signed_codes,
  [[Specialize]] uint group_size,
  [[Specialize]] uint ROW_BLOCK,
  [[Specialize]] uint shared_words,
  [[Specialize]] uint split_phase,
  [[Specialize]] uint splits,
  [[Specialize]] GemmBLayout b_layout,
  [[Specialize]] uint NUM_SIMDGROUPS,
  [[Specialize]] bool has_rht,
  [[Specialize]] uint stage_depth,
) {
  let column_tiles = (n - 1) / 64 + 1;
  let row_tiles = (m - 1) / ROW_BLOCK + 1;
  let linear_tile = column_tile + row_tile * column_tiles;
  gemm_int8<32, GemmInt8Shared<1, 32>, B, D, true>(
    b, scales, biases, zero_points, a, a_scales, a_group_sums, d, bias, partials, k, n, m, scale_output_stride,
    scale_group_stride, zero_point_output_stride, zero_point_group_stride, ab_scale, soft_cap, uint2(linear_tile / row_tiles, linear_tile % row_tiles),
    simd_group * 32 + simd_lane, has_scale, has_bias, has_soft_cap, b_prologue, quantization_mode, signed_codes, group_size,
    ROW_BLOCK, stage_depth, NUM_SIMDGROUPS, split_phase, splits, split_index, b_layout, rht_factors, has_rht
  );
}

// RowMajor B codes `rows` copied in GemmBLayout.Tiled order to `tiles`. A workgroup moves the 8 x 64-code tile at
// (`block`, `tile`), 16 bytes per invocation; a 64-code row run is `run_words` 16-byte words, 2 for W4 and 4 for W8.
[[Kernel]]
[[Precondition("encode", "n.is_multiple_of(32) && k.is_multiple_of(64) && [2, 4].contains(&run_words)")]]
public func GemmPrepareBLayout(
  Ptr<uint, Access.Read> rows,
  Ptr<uint, Access.ReadWrite> tiles,
  uint k,
  uint n,
  uint run_words,
  [[Groups("k / 64")]] uint tile,
  [[Groups("n / 8")]] uint block,
  [[Threads("32")]] uint thread,
) {
  if (thread >= 8 * run_words) {
    return;
  }
  let row_words = k / 64 * run_words;
  let source = (block * 8 + thread / run_words) * row_words + tile * run_words + thread % run_words;
  let destination = block * (k / 64) * 8 * run_words + tile * 8 * run_words + thread;
  let words = load_aligned<uint4, 16>(rows, 4 * source);
  store_aligned<16>(tiles, 4 * destination, words);
}
