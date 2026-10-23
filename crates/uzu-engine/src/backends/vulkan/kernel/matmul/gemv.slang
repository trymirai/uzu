import definitions;
import common.matmul;
import common.workgroup_reduction;
import generated.gemm;
import generated.gemm_b_layout;
import generated.quantization;
import generated.quantization_method;
import matmul.common.int8_operands;
import matmul.gemm.common.simdgroup_mma_core;
import matmul.gemv.common.arguments;
import matmul.gemv.common.epilogue;
import matmul.gemv.common.full_precision_b_source;
import matmul.gemv.common.quant_b_source;
import matmul.gemv.common.reduce;
import matmul.gemv.common.tile;

// Rows of A against n rows of B, full precision or U4/U8 codes (the B prologue), with gathered B rows. METAL_LAYOUT reads
// contiguous B rows through FullPrecisionBSource or QuantBSource; other shapes use gemv_body, where GEMV_LANES
// invocations in subgroup_invocation order stride K for each of GEMV_OUTPUTS outputs of one A row.

// Full-precision B element (row, inner) at row row_stride + inner inner_stride, the CPU MatmulKernel's index for
// transposed B of any leading dimension (stride ld, 1) and for B that is not transposed (1, ld).
struct StridedOperand<T : IModelFloat> : IMatmulOperand {
  Ptr<T, Access.Read> values;
  uint row_stride;
  uint inner_stride;

  float value(uint row, uint inner, uint base) {
    return values[row * row_stride + inner * inner_stride].toFloat();
  }
}

static const uint GEMV_THREADS = 256;
// Output rows per workgroup, as the binding's group count n.div_ceil(8) assumes.
static const uint GEMV_OUTPUTS = 8;
static const uint GEMV_LANES = GEMV_THREADS / GEMV_OUTPUTS;

void gemv_body<X : IMatmulOperand, B : IModelFloat, D : IModelFloat, W : IMatmulOperand>(
  W weights,
  X activations,
  Ptr<D, Access.ReadWrite> d,
  Ptr<B, Access.Read> bias,
  Ptr<uint, Access.Read> gather,
  uint k,
  uint n,
  uint m,
  float ab_scale,
  float soft_cap,
  uint tile,
  uint row,
  bool has_scale,
  bool has_bias,
  bool has_soft_cap,
  bool gathered,
) {
  let invocation = subgroup_invocation();
  let output = invocation / GEMV_LANES;
  let lane = invocation % GEMV_LANES;
  let column = tile * GEMV_OUTPUTS + output;
  let active = column < n;
  let a_base = row * k;
  var partial = 0.0;
  if (active) {
    let b_row = gathered ? gather[row * n + column] : column;
    for (uint j = lane; j < k; j += GEMV_LANES) {
      partial = fma(activations.value(row, j, a_base), weights.value(b_row, j, b_row * k), partial);
    }
  }
  let dot = segment_reduce<Sum<float>, GEMV_THREADS>(partial, GEMV_LANES);
  if (lane == 0 && active) {
    let bias_value = has_bias ? bias[column].toFloat() : 0.0;
    let value = matmul_epilogue(dot, has_scale, ab_scale, has_bias, bias_value, has_soft_cap, soft_cap);
    d[row * n + column] = D.fromFloat(value);
  }
}

[[Kernel]]
[[Precondition("new", "b_prologue == GemmBPrologueKind::FullPrecision || (group_size > 0 && quantization_mode != QuantizationMode::I8)")]]
[[Precondition("new", "METAL_LAYOUT || (OUTPUT_ROW_TILE == 8 && NUM_SIMDGROUPS == 8 && INPUT_ROW_TILE == 1)")]]
[[Precondition("new", "!METAL_LAYOUT || b_prologue != GemmBPrologueKind::FullPrecision || (NUM_SIMDGROUPS == 8 && INPUT_ROW_TILE == 1 && REDUCTION_LANES == 32 && GROUP_LANES == 1 && [1, 2, 4, 8].contains(&K_SPLIT) && (OUTPUT_ROW_TILE == 8 / K_SPLIT || OUTPUT_ROW_TILE == 32 / K_SPLIT) && (INPUT_ALIGNED || K_SPLIT == 1))")]]
[[Precondition("new", "!METAL_LAYOUT || b_prologue == GemmBPrologueKind::FullPrecision || (K_SPLIT == 1 && [16, 32, 64, 128].contains(&group_size))")]]
[[Precondition("new", "!METAL_LAYOUT || b_prologue == GemmBPrologueKind::FullPrecision || ((1..=16).contains(&INPUT_ROW_TILE) && (INPUT_ROW_TILE != 1 || REDUCTION_LANES == 32))")]]
[[Precondition("new", "!METAL_LAYOUT || b_prologue == GemmBPrologueKind::FullPrecision || ([8, 16, 32].contains(&REDUCTION_LANES) && [1, 2, 4, 8, 16].contains(&GROUP_LANES) && REDUCTION_LANES.is_multiple_of(GROUP_LANES))")]]
[[Precondition("new", "!METAL_LAYOUT || b_prologue == GemmBPrologueKind::FullPrecision || (group_size.is_multiple_of(GROUP_LANES) && (group_size / GROUP_LANES).is_multiple_of(if quantization_mode == QuantizationMode::U4 { 16 } else { 8 }))")]]
[[Precondition("new", "!METAL_LAYOUT || b_prologue == GemmBPrologueKind::FullPrecision || ([2, 4, 8].contains(&NUM_SIMDGROUPS) && OUTPUT_ROW_TILE.is_multiple_of(NUM_SIMDGROUPS) && [1, 2, 4, 8].contains(&(OUTPUT_ROW_TILE / NUM_SIMDGROUPS)) && (OUTPUT_ROW_TILE / NUM_SIMDGROUPS).is_multiple_of(32 / REDUCTION_LANES))")]]
[[Precondition("new", "!METAL_LAYOUT || !gathered || INPUT_ROW_TILE == 1")]]
[[Precondition("encode", "!METAL_LAYOUT || b_prologue != GemmBPrologueKind::FullPrecision || (b_row_stride == k.into() && b_inner_stride == 1u32.into())")]]
[[Precondition("encode", "!METAL_LAYOUT || b_prologue == GemmBPrologueKind::FullPrecision || quantization_mode != QuantizationMode::U4 || k.is_multiple_of(2)")]]
[[Precondition("encode", "!METAL_LAYOUT || !INPUT_ALIGNED || k.is_multiple_of(if b_prologue == GemmBPrologueKind::FullPrecision { 128 } else { group_size })")]]
[[Precondition("encode", "!METAL_LAYOUT || !FULL_TILE || (n.is_multiple_of(OUTPUT_ROW_TILE) && m.is_multiple_of(INPUT_ROW_TILE))")]]
[[Precondition("encode", "b_layout == GemmBLayout::RowMajor || (b_prologue != GemmBPrologueKind::FullPrecision && n.is_multiple_of(32) && k.is_multiple_of(128))")]]
func Gemv<A: IModelFloat, B: IModelFloat, D: IModelFloat>(
  Ptr<uint, Access.Read> b,
  [[Optional("b_prologue != GemmBPrologueKind::FullPrecision")]] Ptr<B, Access.Read> scales,
  [[Optional("b_prologue == GemmBPrologueKind::ScaleBiasDequant")]] Ptr<B, Access.Read> biases,
  [[Optional("b_prologue == GemmBPrologueKind::ScaleZeroPointDequant")]] Ptr<uint8_t, Access.Read> zero_points,
  Ptr<A, Access.Read> a,
  Ptr<D, Access.ReadWrite> d,
  [[Optional("has_bias")]] Ptr<B, Access.Read> bias,
  [[Optional("gathered")]] Ptr<uint, Access.Read> gather,
  uint k,
  uint n,
  uint m,
  [[Optional("b_prologue == GemmBPrologueKind::FullPrecision")]] uint b_row_stride,
  [[Optional("b_prologue == GemmBPrologueKind::FullPrecision")]] uint b_inner_stride,
  [[Optional("b_prologue != GemmBPrologueKind::FullPrecision")]] uint scale_output_stride,
  [[Optional("b_prologue != GemmBPrologueKind::FullPrecision")]] uint scale_group_stride,
  [[Optional("b_prologue == GemmBPrologueKind::ScaleZeroPointDequant")]] uint zero_point_output_stride,
  [[Optional("b_prologue == GemmBPrologueKind::ScaleZeroPointDequant")]] uint zero_point_group_stride,
  [[Optional("has_scale")]] float ab_scale,
  [[Optional("has_soft_cap")]] float soft_cap,
  [[Groups("n.div_ceil(OUTPUT_ROW_TILE)")]] uint output_tile_idx,
  [[Groups("m.div_ceil(INPUT_ROW_TILE)")]] uint input_tile_idx,
  [[Threads("32")]] uint simd_lane,
  [[Threads("NUM_SIMDGROUPS")]] uint simd_group,
  [[Specialize]] bool has_scale,
  [[Specialize]] bool has_bias,
  [[Specialize]] bool has_soft_cap,
  [[Specialize]] bool gathered,
  [[Specialize]] GemmBPrologueKind b_prologue,
  [[Specialize]] QuantizationMode quantization_mode,
  [[Specialize]] bool signed_codes,
  [[Specialize]] uint group_size,
  [[Specialize]] uint OUTPUT_ROW_TILE,
  [[Specialize]] uint NUM_SIMDGROUPS,
  [[Specialize]] uint INPUT_ROW_TILE,
  [[Specialize]] uint REDUCTION_LANES,
  [[Specialize]] uint GROUP_LANES,
  [[Specialize]] uint K_SPLIT,
  [[Specialize]] bool INPUT_ALIGNED,
  [[Specialize]] bool FULL_TILE,
  [[Specialize]] bool METAL_LAYOUT,
  [[Specialize]] GemmBLayout b_layout,
) {
  let quantized = b_prologue != GemmBPrologueKind.FullPrecision;
  let method = gemm_quantization_method(b_prologue);
  let BITS = quantization_mode == QuantizationMode.U4 ? 4u : 8u;
  if (!METAL_LAYOUT) {
    DenseOperand<A> activations = { a };
    if (quantized) {
      let weights = gemm_weights<B>(b, scales, biases, zero_points, group_size, scale_output_stride, scale_group_stride,
        zero_point_output_stride, zero_point_group_stride, quantization_mode, signed_codes, b_prologue, k, b_layout);
      gemv_body<DenseOperand<A>, B, D>(weights, activations, d, bias, gather, k, n, m, ab_scale, soft_cap,
        output_tile_idx, input_tile_idx, has_scale, has_bias, has_soft_cap, gathered);
    } else {
      StridedOperand<B> strided = { Ptr<B, Access.Read>(b), b_row_stride, b_inner_stride };
      gemv_body<DenseOperand<A>, B, D>(strided, activations, d, bias, gather, k, n, m, ab_scale, soft_cap,
        output_tile_idx, input_tile_idx, has_scale, has_bias, has_soft_cap, gathered);
    }
    return;
  }
  GemvOperands<A, B, D> ops = { b, scales, zero_points, biases, a, d, bias, gather };
  GemvParams params = {
    k, n, scale_output_stride, scale_group_stride, zero_point_output_stride, zero_point_group_stride, m, ab_scale,
    soft_cap, has_scale, has_bias, has_soft_cap, gathered, signed_codes, b_layout
  };
  let Tile = GemvTile(INPUT_ROW_TILE, OUTPUT_ROW_TILE, REDUCTION_LANES, GROUP_LANES, NUM_SIMDGROUPS, K_SPLIT);
  let tile = OutputTile.make(Tile, FULL_TILE, output_tile_idx, input_tile_idx, n);
  float result[GEMV_MAX_INPUT_ROWS][GEMV_MAX_ROWS_PER_LANE];
  [unroll]
  for (uint I = 0; I < GEMV_MAX_INPUT_ROWS; I++) {
    [unroll]
    for (uint R = 0; R < GEMV_MAX_ROWS_PER_LANE; R++) {
      result[I][R] = 0.0;
    }
  }
  if (quantized) {
    QuantBSource.accumulate(result, ops, params, Tile, tile, method, group_size, BITS, INPUT_ALIGNED, FULL_TILE);
  } else {
    FullPrecisionBSource.accumulate(result, ops, params, Tile, tile, INPUT_ALIGNED, FULL_TILE);
  }
  Reduce.run(result, Tile, tile);
  Epilogue.store(result, ops, params, Tile, tile, FULL_TILE);
}

// A8 activations (MatmulA::Int8Symmetric) against U4 codes or U8 symmetric codes, its own entry point so the integer
// dot product capability stays out of the float modules. Each of 8 simdgroups computes ROWS rows for INPUT_ROWS inputs;
// a lane takes every 32nd 32-value run, one weight group's, and folds its exact integer dots as int8_fold does.
static const uint GEMV_INT8_SIMDGROUPS = 8;
static const uint GEMV_INT8_RUN = 32;
// Outputs a simdgroup reduces through shared memory at a time, on native 128-lane subgroups.
static const uint GEMV_INT8_CHUNK = 4;

groupshared float gemv_int8_partials[GEMV_INT8_CHUNK * GEMV_THREADS];

// A subgroup-scope barrier over workgroup memory (execution and memory scope Subgroup = 3, semantics AcquireRelease |
// WorkgroupMemory = 0x108): a simdgroup's cluster lies in one subgroup of the native path, so its partials need no
// workgroup barrier.
void gemv_int8_subgroup_barrier() {
  spirv_asm {
    OpControlBarrier $(3u) $(3u) $(264u);
  };
}

// Eight words of the `count` bytes (16 or 32) at byte `start` of `source`, one 16-byte load per 4, +0 past `count`.
void gemv_int8_words(Ptr<uint8_t, Access.Read> source, uint start, uint count, out uint words[8]) {
  [unroll]
  for (uint half = 0; half < 2; half++) {
    if (half * 16 >= count) {
      words[4 * half] = 0;
      words[4 * half + 1] = 0;
      words[4 * half + 2] = 0;
      words[4 * half + 3] = 0;
      continue;
    }
    let word = load_aligned<uint4, 16>(source, start + 16 * half);
    [unroll]
    for (uint component = 0; component < 4; component++) {
      words[4 * half + component] = word[component];
    }
  }
}

// The GemvInt8 entry points' algorithm for simdgroup `simd_group` of 8, lane `simd_lane` of 32: segment_reduce sums its
// lanes, or through shared memory where CLUSTERED: shuffles on wide subgroups cost more than its stores and reads.
void gemv_int8<B : IModelFloat, D : IModelFloat, let CLUSTERED : bool>(
  Ptr<uint, Access.Read> b,
  Ptr<B, Access.Read> scales,
  Ptr<B, Access.Read> biases,
  Ptr<uint8_t, Access.Read> zero_points,
  Ptr<int8_t, Access.Read> a,
  Ptr<float, Access.Read> a_scales,
  Ptr<int, Access.Read> a_group_sums,
  Ptr<D, Access.ReadWrite> d,
  Ptr<B, Access.Read> bias,
  Ptr<uint, Access.Read> gather,
  uint k,
  uint n,
  uint m,
  uint scale_output_stride,
  uint scale_group_stride,
  uint zero_point_output_stride,
  uint zero_point_group_stride,
  float ab_scale,
  float soft_cap,
  uint output_tile,
  uint input_tile,
  uint simd_lane,
  uint simd_group,
  bool has_scale,
  bool has_bias,
  bool has_soft_cap,
  GemmBPrologueKind b_prologue,
  QuantizationMode quantization_mode,
  bool signed_codes,
  uint group_size,
  uint ROWS,
  uint INPUT_ROWS,
  GemmBLayout b_layout,
  bool unsigned_dot,
  bool gathered,
) {
  let u4 = quantization_mode == QuantizationMode.U4;
  let method = gemm_quantization_method(b_prologue);
  let scale_bias = method == QuantizationMethod.ScaleBias;
  let weights = gemm_weights<B>(b, scales, biases, zero_points, group_size, scale_output_stride, scale_group_stride,
    zero_point_output_stride, zero_point_group_stride, quantization_mode, signed_codes, b_prologue, k, b_layout);
  GemmParams params = { m, n, k, k, n, ab_scale, soft_cap, has_scale, has_bias, has_soft_cap, 0 };
  let first_row = (output_tile * GEMV_INT8_SIMDGROUPS + simd_group) * ROWS;
  let first_input = input_tile * INPUT_ROWS;
  let run_bytes = u4 ? GEMV_INT8_RUN / 2 : GEMV_INT8_RUN;
  let codes = Ptr<uint8_t, Access.Read>(a);

  float sums[GEMV_MAX_INPUT_ROWS][4];
  [unroll]
  for (uint I = 0; I < GEMV_MAX_INPUT_ROWS; I++) {
    [unroll]
    for (uint R = 0; R < 4; R++) {
      sums[I][R] = 0.0;
    }
  }
  for (uint run = simd_lane; run < k / GEMV_INT8_RUN; run += SIMD_SIZE) {
    let column = run * GEMV_INT8_RUN;
    let group = column / group_size;
    // A group's first run adds its ScaleBias correction.
    let head = scale_bias && column % group_size == 0;
    uint words[4][8];
    WeightGroup metadata[4];
    uint code_sums[4];
    [unroll]
    for (uint R = 0; R < ROWS; R++) {
      // Gathered outputs of the one input row read the B rows its indices name.
      let output_row = min(first_row + R, n - 1);
      let row = gathered ? gather[first_input * n + output_row] : output_row;
      metadata[R] = weights.metadata(row, group);
      gemv_int8_words(weights.codes, weights.code_byte(row, column), run_bytes, words[R]);
      if (unsigned_dot) {
        code_sums[R] = unsigned_codes(words[R], u4, signed_codes);
      }
    }
    [unroll]
    for (uint I = 0; I < INPUT_ROWS; I++) {
      let input = first_input + I;
      if (input < m) {
        uint activation[8];
        gemv_int8_words(codes, input * k + column, GEMV_INT8_RUN, activation);
        let scale = a_scales[input * (k / INT8_SCALE_GROUP) + column / INT8_SCALE_GROUP];
        let group_sum = head ? a_group_sums[input * (k / group_size) + group] : 0;
        let activation_sum = unsigned_dot ? int(biased_activations(activation)) - 128 * int(GEMV_INT8_RUN) : 0;
        [unroll]
        for (uint R = 0; R < ROWS; R++) {
          var dot = 0;
          if (unsigned_dot) {
            var products = 0u;
            [unroll]
            for (uint word = 0; word < 8; word++) {
              products = udot_u8x4(activation[word], words[R][word], products);
            }
            dot = int(products) - 128 * int(code_sums[R]) - int(metadata[R].zero_point) * activation_sum;
          } else {
            [unroll]
            for (uint word = 0; word < 4; word++) {
              if (u4) {
                let centered = centered_u4(words[R][word], metadata[R].zero_point, signed_codes);
                dot += dot_i8x4(activation[2 * word], centered.x) + dot_i8x4(activation[2 * word + 1], centered.y);
              } else {
                dot += dot_i8x4(activation[word], centered_u8(words[R][word], signed_codes))
                  + dot_i8x4(activation[4 + word], centered_u8(words[R][4 + word], signed_codes));
              }
            }
          }
          sums[I][R] = int8_fold(sums[I][R], scale, metadata[R], dot, head, group_sum);
        }
      }
    }
  }
  // Each lane finishes one of its simdgroup's INPUT_ROWS x ROWS outputs, output I ROWS + R in lane I ROWS + R.
  var value = 0.0;
  if (CLUSTERED) {
    // Each chunk: every lane stores its partials of the chunk's outputs, then the chunk's lanes sum their output's 32.
    let invocation = simd_group * SIMD_SIZE + simd_lane;
    [unroll]
    for (uint chunk = 0; chunk < SIMD_SIZE; chunk += GEMV_INT8_CHUNK) {
      if (chunk < INPUT_ROWS * ROWS) {
        [unroll]
        for (uint c = 0; c < GEMV_INT8_CHUNK; c++) {
          let j = chunk + c;
          gemv_int8_partials[c * GEMV_THREADS + invocation] = j < INPUT_ROWS * ROWS ? sums[j / ROWS][j % ROWS] : 0.0;
        }
        gemv_int8_subgroup_barrier();
        if (simd_lane >= chunk && simd_lane < chunk + GEMV_INT8_CHUNK) {
          let first = (simd_lane - chunk) * GEMV_THREADS + simd_group * SIMD_SIZE;
          var total = 0.0;
          for (uint lane = 0; lane < SIMD_SIZE; lane++) {
            total += gemv_int8_partials[first + lane];
          }
          value = total;
        }
        gemv_int8_subgroup_barrier();
      }
    }
  } else {
    [unroll]
    for (uint I = 0; I < INPUT_ROWS; I++) {
      [unroll]
      for (uint R = 0; R < ROWS; R++) {
        let total = segment_reduce<Sum<float>, GEMV_THREADS>(sums[I][R], SIMD_SIZE);
        if (simd_lane == I * ROWS + R) {
          value = total;
        }
      }
    }
  }
  let input = first_input + simd_lane / ROWS;
  let row = first_row + simd_lane % ROWS;
  if (simd_lane < INPUT_ROWS * ROWS && input < m && row < n) {
    let bias_value = has_bias ? bias[row].toFloat() : 0.0;
    let output = gemm_output(value, bias_value, params);
    d[input * n + row] = D.fromFloat(output);
  }
}

[[Kernel]]
[[Precondition("new", "b_prologue != GemmBPrologueKind::FullPrecision && [32, 64, 128].contains(&group_size)")]]
[[Precondition("new", "quantization_mode == QuantizationMode::U4 || (quantization_mode == QuantizationMode::U8 && b_prologue == GemmBPrologueKind::ScaleSymmetricDequant)")]]
[[Precondition("new", "[1, 2, 4].contains(&ROWS) && (1..=8).contains(&INPUT_ROWS) && (!gathered || INPUT_ROWS == 1)")]]
[[Precondition("encode", "k.is_multiple_of(128) && (b_layout == GemmBLayout::RowMajor || n.is_multiple_of(32))")]]
func GemvInt8<B: IModelFloat, D: IModelFloat>(
  Ptr<uint, Access.Read> b,
  Ptr<B, Access.Read> scales,
  [[Optional("b_prologue == GemmBPrologueKind::ScaleBiasDequant")]] Ptr<B, Access.Read> biases,
  [[Optional("b_prologue == GemmBPrologueKind::ScaleZeroPointDequant")]] Ptr<uint8_t, Access.Read> zero_points,
  Ptr<int8_t, Access.Read> a,
  Ptr<float, Access.Read> a_scales,
  [[Optional("b_prologue == GemmBPrologueKind::ScaleBiasDequant")]] Ptr<int, Access.Read> a_group_sums,
  Ptr<D, Access.ReadWrite> d,
  [[Optional("has_bias")]] Ptr<B, Access.Read> bias,
  [[Optional("gathered")]] Ptr<uint, Access.Read> gather,
  uint k,
  uint n,
  uint m,
  uint scale_output_stride,
  uint scale_group_stride,
  [[Optional("b_prologue == GemmBPrologueKind::ScaleZeroPointDequant")]] uint zero_point_output_stride,
  [[Optional("b_prologue == GemmBPrologueKind::ScaleZeroPointDequant")]] uint zero_point_group_stride,
  [[Optional("has_scale")]] float ab_scale,
  [[Optional("has_soft_cap")]] float soft_cap,
  [[Groups("n.div_ceil(8 * ROWS)")]] uint output_tile,
  [[Groups("m.div_ceil(INPUT_ROWS)")]] uint input_tile,
  [[Threads("256")]] uint tid,
  [[Specialize]] bool has_scale,
  [[Specialize]] bool has_bias,
  [[Specialize]] bool has_soft_cap,
  [[Specialize]] GemmBPrologueKind b_prologue,
  [[Specialize]] QuantizationMode quantization_mode,
  [[Specialize]] bool signed_codes,
  [[Specialize]] uint group_size,
  [[Specialize]] uint ROWS,
  [[Specialize]] uint INPUT_ROWS,
  [[Specialize]] GemmBLayout b_layout,
  [[Specialize]] bool unsigned_dot,
  [[Specialize]] bool gathered,
) {
  let invocation = subgroup_invocation();
  gemv_int8<B, D, false>(
    b, scales, biases, zero_points, a, a_scales, a_group_sums, d, bias, gather, k, n, m, scale_output_stride,
    scale_group_stride, zero_point_output_stride, zero_point_group_stride, ab_scale, soft_cap, output_tile, input_tile,
    invocation % 32, invocation / 32, has_scale, has_bias, has_soft_cap, b_prologue, quantization_mode, signed_codes,
    group_size, ROWS, INPUT_ROWS, b_layout, unsigned_dot, gathered);
}

// GemvInt8 on native subgroups of 128 lanes, where the device requires them for this pipeline in full subgroups:
// each subgroup holds four of the workgroup's simdgroups as 32-lane clusters, in subgroup order.
[[Kernel]]
[[Precondition("new", "context.physical_device().capabilities.contains(VkDeviceCapabilities::GEMV_INT8_NATIVE_128)")]]
[[RequiredSubgroupSize("128")]]
[[Precondition("new", "b_prologue != GemmBPrologueKind::FullPrecision && [32, 64, 128].contains(&group_size)")]]
[[Precondition("new", "quantization_mode == QuantizationMode::U4 || (quantization_mode == QuantizationMode::U8 && b_prologue == GemmBPrologueKind::ScaleSymmetricDequant)")]]
[[Precondition("new", "[1, 2, 4].contains(&ROWS) && (1..=8).contains(&INPUT_ROWS) && (!gathered || INPUT_ROWS == 1)")]]
[[Precondition("encode", "k.is_multiple_of(128) && (b_layout == GemmBLayout::RowMajor || n.is_multiple_of(32))")]]
func GemvInt8Native128<B: IModelFloat, D: IModelFloat>(
  Ptr<uint, Access.Read> b,
  Ptr<B, Access.Read> scales,
  [[Optional("b_prologue == GemmBPrologueKind::ScaleBiasDequant")]] Ptr<B, Access.Read> biases,
  [[Optional("b_prologue == GemmBPrologueKind::ScaleZeroPointDequant")]] Ptr<uint8_t, Access.Read> zero_points,
  Ptr<int8_t, Access.Read> a,
  Ptr<float, Access.Read> a_scales,
  [[Optional("b_prologue == GemmBPrologueKind::ScaleBiasDequant")]] Ptr<int, Access.Read> a_group_sums,
  Ptr<D, Access.ReadWrite> d,
  [[Optional("has_bias")]] Ptr<B, Access.Read> bias,
  [[Optional("gathered")]] Ptr<uint, Access.Read> gather,
  uint k,
  uint n,
  uint m,
  uint scale_output_stride,
  uint scale_group_stride,
  [[Optional("b_prologue == GemmBPrologueKind::ScaleZeroPointDequant")]] uint zero_point_output_stride,
  [[Optional("b_prologue == GemmBPrologueKind::ScaleZeroPointDequant")]] uint zero_point_group_stride,
  [[Optional("has_scale")]] float ab_scale,
  [[Optional("has_soft_cap")]] float soft_cap,
  [[Groups("n.div_ceil(8 * ROWS)")]] uint output_tile,
  [[Groups("m.div_ceil(INPUT_ROWS)")]] uint input_tile,
  [[Threads("256")]] uint tid,
  [[Specialize]] bool has_scale,
  [[Specialize]] bool has_bias,
  [[Specialize]] bool has_soft_cap,
  [[Specialize]] GemmBPrologueKind b_prologue,
  [[Specialize]] QuantizationMode quantization_mode,
  [[Specialize]] bool signed_codes,
  [[Specialize]] uint group_size,
  [[Specialize]] uint ROWS,
  [[Specialize]] uint INPUT_ROWS,
  [[Specialize]] GemmBLayout b_layout,
  [[Specialize]] bool unsigned_dot,
  [[Specialize]] bool gathered,
) {
  let invocation = subgroup_invocation();
  gemv_int8<B, D, true>(
    b, scales, biases, zero_points, a, a_scales, a_group_sums, d, bias, gather, k, n, m, scale_output_stride,
    scale_group_stride, zero_point_output_stride, zero_point_group_stride, ab_scale, soft_cap, output_tile, input_tile,
    invocation % 32, invocation / 32, has_scale, has_bias, has_soft_cap, b_prologue, quantization_mode, signed_codes,
    group_size, ROWS, INPUT_ROWS, b_layout, unsigned_dot, gathered);
}
