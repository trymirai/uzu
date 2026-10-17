import definitions;
import common.activation;
import common.matmul;
import common.storage_address;
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

// The Gemv logical kernel, Metal's gemv.metal: rows of A against n rows of B, full precision or U4/U8 codes (the B
// prologue), one or more input rows per workgroup and gathered B rows. Metal's layout (METAL_LAYOUT) reads contiguous
// B rows through FullPrecisionBSource or QuantBSource, reduces and finishes in the Epilogue. Shapes outside its
// contract, B of any strides and quantized groups or K its packed loads do not take, use gemv_body: the CPU
// MatmulKernel's product of one A row and B rows read through common.matmul's operand readers. A workgroup computes
// GEMV_OUTPUTS outputs of one A row, GEMV_LANES invocations each striding K; the partial sums meet in shared memory in
// invocation order, assuming nothing about subgroups. Outputs common.matmul classes careful are summed again from exact
// products, those below 2^-101 summed scaled up by 2^24, where no partial flushes either, and added back with one
// rounding; replayed outputs take the CPU's order.

// Full-precision B element (row, inner) at row row_stride + inner inner_stride, the CPU MatmulKernel's index for
// transposed B of any leading dimension (stride ld, 1) and for B that is not transposed (1, ld).
struct StridedOperand<T : IModelFloat> : IMatmulOperand {
  Ptr<T, Access.Read> values;
  uint row_stride;
  uint inner_stride;

  float value(uint row, uint inner, uint base) {
    return element_pointer(values, add_address(element_index(row, row_stride, 0), element_index(inner, inner_stride, 0)))[0].toFloat();
  }
}

static const uint GEMV_THREADS = 256;
// Output rows per workgroup, as the binding's group count n.div_ceil(8) assumes.
static const uint GEMV_OUTPUTS = 8;
static const uint GEMV_LANES = GEMV_THREADS / GEMV_OUTPUTS;

groupshared float gemv_main[GEMV_THREADS];
groupshared float gemv_tiny[GEMV_THREADS];
// Per invocation: the matmul_fields of its operands merged.
groupshared uint gemv_fields[GEMV_THREADS];
groupshared uint gemv_mode[GEMV_OUTPUTS];

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
  uint tid,
  bool has_scale,
  bool has_bias,
  bool has_soft_cap,
  bool gathered,
) {
  let output = tid / GEMV_LANES;
  let lane = tid % GEMV_LANES;
  let column = tile * GEMV_OUTPUTS + output;
  let active = column < n;
  let a_base = row * k;
  uint b_row = 0;
  var partial = 0.0;
  var fields = 0x0000ffffu;
  if (active) {
    b_row = gathered ? gather[row * n + column] : column;
    for (uint j = lane; j < k; j += GEMV_LANES) {
      let x = activations.value(row, j, a_base);
      let w = weights.value(b_row, j, b_row * k);
      fields = matmul_merge_fields(fields, matmul_fields(x, w));
      partial = ordered_sum(partial, x * w);
    }
  }
  gemv_main[tid] = partial;
  gemv_fields[tid] = fields;
  GroupMemoryBarrierWithGroupSync();

  var dot = 0.0;
  var bias_value = 0.0;
  if (lane == 0) {
    var mode = MATMUL_FAST;
    if (active) {
      for (uint other = tid; other < tid + GEMV_LANES; other++) {
        dot = ordered_sum(dot, gemv_main[other]);
        fields = matmul_merge_fields(fields, gemv_fields[other]);
      }
      if (has_bias) {
        bias_value = bias[column].toFloat();
      }
      mode = matmul_mode(fields, k, has_scale, ab_scale, has_bias, bias_value);
    }
    gemv_mode[output] = mode;
  }
  GroupMemoryBarrierWithGroupSync();

  let mode = gemv_mode[output];
  var careful = false;
  for (uint other = 0; other < GEMV_OUTPUTS; other++) {
    careful = careful || gemv_mode[other] == MATMUL_CAREFUL;
  }
  if (careful) {
    var main = 0.0;
    var tiny = 0.0;
    if (mode == MATMUL_CAREFUL) {
      for (uint j = lane; j < k; j += GEMV_LANES) {
        let product = exact_product(activations.value(row, j, a_base), weights.value(b_row, j, b_row * k));
        if ((asuint(product) & 0x7fffffffu) >= 0x0d000000u) {
          main = ordered_sum(main, product);
        } else {
          tiny = ordered_sum(tiny, exact_scale_up_24(product));
        }
      }
    }
    gemv_main[tid] = main;
    gemv_tiny[tid] = tiny;
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0 && mode == MATMUL_CAREFUL) {
      main = 0.0;
      tiny = 0.0;
      for (uint other = tid; other < tid + GEMV_LANES; other++) {
        main = ordered_sum(main, gemv_main[other]);
        tiny = ordered_sum(tiny, gemv_tiny[other]);
      }
      dot = exact_sum(main, exact_scale_down(tiny, 24), false);
    }
  }

  if (lane == 0 && active) {
    if (mode == MATMUL_REPLAY) {
      dot = matmul_replay(activations, row, weights, b_row, k);
    }
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
  QuantizedWeights<B> weights = {
    gemv_cast<uint8_t, uint>(b), scales, biases, zero_points, group_size, scale_output_stride, scale_group_stride,
    zero_point_output_stride, zero_point_group_stride, BITS == 4, signed_codes, method, k, b_layout
  };
  DenseOperand<A> activations = { a };
  if (!METAL_LAYOUT) {
    let tid = simd_group * 32 + simd_lane;
    if (quantized) {
      gemv_body<DenseOperand<A>, B, D>(weights, activations, d, bias, gather, k, n, m, ab_scale, soft_cap,
        output_tile_idx, input_tile_idx, tid, has_scale, has_bias, has_soft_cap, gathered);
    } else {
      StridedOperand<B> strided = { gemv_cast<B, uint>(b), b_row_stride, b_inner_stride };
      gemv_body<DenseOperand<A>, B, D>(strided, activations, d, bias, gather, k, n, m, ab_scale, soft_cap,
        output_tile_idx, input_tile_idx, tid, has_scale, has_bias, has_soft_cap, gathered);
    }
    return;
  }
  GemvOperands<A, B, D> ops = { b, scales, zero_points, biases, a, d, bias, gather };
  GemvParams params = {
    k, n, scale_output_stride, scale_group_stride, zero_point_output_stride, zero_point_group_stride, m, ab_scale,
    soft_cap, has_scale, has_bias, has_soft_cap, gathered, signed_codes, b_layout
  };
  let Tile = GemvTile(INPUT_ROW_TILE, OUTPUT_ROW_TILE, REDUCTION_LANES, GROUP_LANES, NUM_SIMDGROUPS, K_SPLIT);
  let tile = OutputTile.make(Tile, FULL_TILE, output_tile_idx, input_tile_idx, simd_group, simd_lane, n);
  uint fields = 0x0000ffffu;
  float result[GEMV_MAX_INPUT_ROWS][GEMV_MAX_ROWS_PER_LANE];
  [unroll]
  for (uint I = 0; I < GEMV_MAX_INPUT_ROWS; I++) {
    [unroll]
    for (uint R = 0; R < GEMV_MAX_ROWS_PER_LANE; R++) {
      result[I][R] = 0.0;
    }
  }
  if (quantized) {
    QuantBSource.accumulate(
      fields, result, ops, params, Tile, tile, method, group_size, BITS, INPUT_ALIGNED, FULL_TILE);
    Reduce.run(result, Tile, tile);
    fields = matmul_workgroup_fields(fields);
    Epilogue.store(result, ops, params, Tile, tile, fields, FULL_TILE, activations, weights);
  } else {
    FullPrecisionBSource.accumulate(fields, result, ops, params, Tile, tile, INPUT_ALIGNED, FULL_TILE);
    Reduce.run(result, Tile, tile);
    fields = matmul_workgroup_fields(fields);
    DenseOperand<B> dense = { gemv_cast<B, uint>(b) };
    Epilogue.store(result, ops, params, Tile, tile, fields, FULL_TILE, activations, dense);
  }
}

// GemvInt8: Gemv for A8 activations, symmetric INT8 codes and FP32 scales per 128 values (MatmulA::Int8Symmetric),
// against U4 codes under any scheme or U8 symmetric codes, its own entry point so the integer dot product capability
// stays out of the float modules. A workgroup computes ROWS output rows in each of its 8 simdgroups for INPUT_ROWS input
// rows; a lane takes every 32nd run of 32 K values, one weight group's, loads each row's codes once for every input
// row, takes their exact integer dots, of unsigned bytes where `unsigned_dot`, and folds them in Metal's integer
// schedule (common/int8_operands). Lanes' sums are added by the subgroup, then every output is classified by its
// operands' fields, replayed in the CPU's order where they cannot bound it, and finished by the CPU's epilogue.
static const uint GEMV_INT8_SIMDGROUPS = 8;
static const uint GEMV_INT8_RUN = 32;

// The sum of `value` over its 32-lane cluster of a wider subgroup, GemvInt8Native128's simdgroup.
float gemv_int8_row_sum(float value) {
  return spirv_asm {
    OpCapability GroupNonUniformClustered;
    result:$$float = OpGroupNonUniformFAdd Subgroup ClusteredReduce $value $(32u);
  };
}

// Eight words of `bytes` from `source`, one 16-byte load per 4 where `vector`.
void gemv_int8_words(Ptr<uint8_t, Access.Read> source, bool vector, uint count, out uint words[8]) {
  [unroll]
  for (uint half = 0; half < 2; half++) {
    if (half * 16 >= count) {
      words[4 * half] = 0;
      words[4 * half + 1] = 0;
      words[4 * half + 2] = 0;
      words[4 * half + 3] = 0;
      continue;
    }
    let base = element_pointer(source, uint2(half * 16, 0));
    var word = uint4(0);
    if (vector) {
      word = loadAligned<16>(Ptr<uint4>(base));
    } else {
      [unroll]
      for (uint byte = 0; byte < 16; byte++) {
        word[byte / 4] |= uint(base[byte]) << (8 * (byte % 4));
      }
    }
    [unroll]
    for (uint component = 0; component < 4; component++) {
      words[4 * half + component] = word[component];
    }
  }
}

// The GemvInt8 entry points' algorithm for simdgroup `simd_group` of 8, lane `simd_lane` of 32: subgroups of 32, or
// 32-lane clusters of wider subgroups where CLUSTERED.
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
) {
  let u4 = quantization_mode == QuantizationMode.U4;
  let method = gemm_quantization_method(b_prologue);
  let scale_bias = method == QuantizationMethod.ScaleBias;
  QuantizedWeights<B> weights = {
    gemv_cast<uint8_t, uint>(b), scales, biases, zero_points, group_size, scale_output_stride, scale_group_stride,
    zero_point_output_stride, zero_point_group_stride, u4, signed_codes, method, k, b_layout
  };
  Int8Activations activations = { a, a_scales, INT8_SCALE_GROUP, u4 };
  GemmParams params = { m, n, k, k, n, ab_scale, soft_cap, has_scale, has_bias, has_soft_cap, 0 };
  let first_row = (output_tile * GEMV_INT8_SIMDGROUPS + simd_group) * ROWS;
  let first_input = input_tile * INPUT_ROWS;
  let run_bytes = u4 ? GEMV_INT8_RUN / 2 : GEMV_INT8_RUN;
  let codes_vector = (address_words(weights.codes).x & 15u) == 0;
  let a_vector = (address_words(a).x & 15u) == 0;
  let codes = gemv_cast<uint8_t, int8_t>(a);

  float sums[GEMV_MAX_INPUT_ROWS][4];
  [unroll]
  for (uint I = 0; I < GEMV_MAX_INPUT_ROWS; I++) {
    [unroll]
    for (uint R = 0; R < 4; R++) {
      sums[I][R] = 0.0;
    }
  }
  var fields = 0x0000ffffu;
  for (uint run = simd_lane; run < k / GEMV_INT8_RUN; run += GEMV_SIMD_SIZE) {
    let column = run * GEMV_INT8_RUN;
    let group = column / group_size;
    // A group's first run adds its ScaleBias correction.
    let head = scale_bias && column % group_size == 0;
    uint words[4][8];
    Int8WeightGroup metadata[4];
    uint code_sums[4];
    [unroll]
    for (uint R = 0; R < ROWS; R++) {
      let row = min(first_row + R, n - 1);
      metadata[R] = Int8WeightGroup.load<B>(weights, row, group);
      fields = matmul_merge_fields(fields, metadata[R].fields<B>(u4));
      gemv_int8_words(element_pointer(weights.codes, weights.code_byte(row, column)), codes_vector, run_bytes, words[R]);
      if (unsigned_dot) {
        code_sums[R] = unsigned_codes(words[R], u4, signed_codes);
      }
    }
    [unroll]
    for (uint I = 0; I < INPUT_ROWS; I++) {
      let input = first_input + I;
      if (input < m) {
        uint activation[8];
        gemv_int8_words(element_pointer(codes, element_index(input, k, column)), a_vector, GEMV_INT8_RUN, activation);
        let scale = a_scales[input * (k / INT8_SCALE_GROUP) + column / INT8_SCALE_GROUP];
        fields = matmul_merge_fields(fields, int8_activation_fields(scale));
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
  fields = matmul_workgroup_fields(fields);

  // Each lane finishes one of its simdgroup's INPUT_ROWS x ROWS outputs.
  var value = 0.0;
  [unroll]
  for (uint I = 0; I < INPUT_ROWS; I++) {
    [unroll]
    for (uint R = 0; R < ROWS; R++) {
      let total = CLUSTERED ? gemv_int8_row_sum(sums[I][R]) : WaveActiveSum(sums[I][R]);
      if (simd_lane == I * ROWS + R) {
        value = total;
      }
    }
  }
  let input = first_input + simd_lane / ROWS;
  let row = first_row + simd_lane % ROWS;
  if (simd_lane < INPUT_ROWS * ROWS && input < m && row < n) {
    let bias_value = has_bias ? bias[row].toFloat() : 0.0;
    let output = gemm_output(value, fields, bias_value, activations, weights, input, row, params);
    element_pointer(d, element_index(input, n, row))[0] = D.fromFloat(output);
  }
}

[[Kernel]]
[[Precondition("new", "context.physical_device().capabilities.contains(VkDeviceCapabilities::INTEGER_DOT_PRODUCT)")]]
[[Precondition("new", "b_prologue != GemmBPrologueKind::FullPrecision && [32, 64, 128].contains(&group_size)")]]
[[Precondition("new", "quantization_mode == QuantizationMode::U4 || (quantization_mode == QuantizationMode::U8 && b_prologue == GemmBPrologueKind::ScaleSymmetricDequant)")]]
[[Precondition("new", "[1, 2, 4].contains(&ROWS) && (1..=8).contains(&INPUT_ROWS)")]]
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
  [[Threads("32")]] uint simd_lane,
  [[Threads("8")]] uint simd_group,
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
) {
  gemv_int8<B, D, false>(
    b, scales, biases, zero_points, a, a_scales, a_group_sums, d, bias, k, n, m, scale_output_stride,
    scale_group_stride, zero_point_output_stride, zero_point_group_stride, ab_scale, soft_cap, output_tile, input_tile,
    simd_lane, simd_group, has_scale, has_bias, has_soft_cap, b_prologue, quantization_mode, signed_codes, group_size,
    ROWS, INPUT_ROWS, b_layout, unsigned_dot);
}

// GemvInt8 on native subgroups of 128 lanes, where the device requires them for this pipeline in full subgroups:
// each subgroup holds four of the workgroup's simdgroups as 32-lane clusters, in subgroup order.
[[Kernel]]
[[Precondition("new", "context.physical_device().capabilities.contains(VkDeviceCapabilities::GEMV_INT8_NATIVE_128)")]]
[[RequiredSubgroupSize("128")]]
[[Precondition("new", "b_prologue != GemmBPrologueKind::FullPrecision && [32, 64, 128].contains(&group_size)")]]
[[Precondition("new", "quantization_mode == QuantizationMode::U4 || (quantization_mode == QuantizationMode::U8 && b_prologue == GemmBPrologueKind::ScaleSymmetricDequant)")]]
[[Precondition("new", "[1, 2, 4].contains(&ROWS) && (1..=8).contains(&INPUT_ROWS)")]]
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
) {
  let invocation = subgroup_id() * WaveGetLaneCount() + WaveGetLaneIndex();
  gemv_int8<B, D, true>(
    b, scales, biases, zero_points, a, a_scales, a_group_sums, d, bias, k, n, m, scale_output_stride,
    scale_group_stride, zero_point_output_stride, zero_point_group_stride, ab_scale, soft_cap, output_tile, input_tile,
    invocation % 32, invocation / 32, has_scale, has_bias, has_soft_cap, b_prologue, quantization_mode, signed_codes,
    group_size, ROWS, INPUT_ROWS, b_layout, unsigned_dot);
}
