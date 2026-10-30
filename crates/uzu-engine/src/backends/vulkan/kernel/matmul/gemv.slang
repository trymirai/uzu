import definitions;
import matmul.common.matmul;
import common.activation_quantization;
import common.hadamard;
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

  float value(uint row, uint inner) {
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
  var partial = 0.0;
  if (active) {
    let b_row = gathered ? gather[row * n + column] : column;
    for (uint j = lane; j < k; j += GEMV_LANES) {
      partial = fma(activations.value(row, j), weights.value(b_row, j), partial);
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
[[Precondition("new", "!METAL_LAYOUT || b_prologue != GemmBPrologueKind::FullPrecision || (NUM_SIMDGROUPS == 8 && INPUT_ROW_TILE == 1 && GROUP_LANES == 1 && [8, 32].contains(&OUTPUT_ROW_TILE))")]]
[[Precondition("new", "!METAL_LAYOUT || b_prologue == GemmBPrologueKind::FullPrecision || [16, 32, 64, 128].contains(&group_size)")]]
[[Precondition("new", "!METAL_LAYOUT || b_prologue == GemmBPrologueKind::FullPrecision || (1..=16).contains(&INPUT_ROW_TILE)")]]
[[Precondition("new", "!METAL_LAYOUT || b_prologue == GemmBPrologueKind::FullPrecision || [1, 2, 4, 8, 16].contains(&GROUP_LANES)")]]
[[Precondition("new", "!METAL_LAYOUT || b_prologue == GemmBPrologueKind::FullPrecision || (group_size.is_multiple_of(GROUP_LANES) && (group_size / GROUP_LANES).is_multiple_of(if quantization_mode == QuantizationMode::U4 { 16 } else { 8 }))")]]
[[Precondition("new", "!METAL_LAYOUT || b_prologue == GemmBPrologueKind::FullPrecision || ([2, 4, 8].contains(&NUM_SIMDGROUPS) && OUTPUT_ROW_TILE.is_multiple_of(NUM_SIMDGROUPS) && [1, 2, 4, 8].contains(&(OUTPUT_ROW_TILE / NUM_SIMDGROUPS)))")]]
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
  [[Specialize]] uint GROUP_LANES,
  [[Specialize]] bool INPUT_ALIGNED,
  [[Specialize]] bool FULL_TILE,
  [[Specialize]] bool METAL_LAYOUT,
  [[Specialize]] GemmBLayout b_layout,
) {
  let quantized = b_prologue != GemmBPrologueKind.FullPrecision;
  let weights = gemm_weights<B>(b, scales, biases, zero_points, group_size, scale_output_stride, scale_group_stride,
    zero_point_output_stride, zero_point_group_stride, quantization_mode, signed_codes, b_prologue, k, b_layout);
  if (!METAL_LAYOUT) {
    DenseOperand<A> activations = { a, k };
    if (quantized) {
      gemv_body<DenseOperand<A>, B, D>(weights, activations, d, bias, gather, k, n, ab_scale, soft_cap,
        output_tile_idx, input_tile_idx, has_scale, has_bias, has_soft_cap, gathered);
    } else {
      StridedOperand<B> strided = { Ptr<B, Access.Read>(b), b_row_stride, b_inner_stride };
      gemv_body<DenseOperand<A>, B, D>(strided, activations, d, bias, gather, k, n, ab_scale, soft_cap,
        output_tile_idx, input_tile_idx, has_scale, has_bias, has_soft_cap, gathered);
    }
    return;
  }
  GemvOperands<A, B, D> ops = { b, a, d, bias, gather };
  GemvParams params = { k, n, m, ab_scale, soft_cap, has_scale, has_bias, has_soft_cap, gathered };
  let Tile = GemvTile(INPUT_ROW_TILE, OUTPUT_ROW_TILE, GROUP_LANES, NUM_SIMDGROUPS);
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
    QuantBSource.accumulate(result, weights, ops, params, Tile, tile, INPUT_ALIGNED, FULL_TILE);
  } else {
    FullPrecisionBSource.accumulate(result, ops, params, Tile, tile, INPUT_ALIGNED, FULL_TILE);
  }
  Reduce.run(result, Tile);
  Epilogue.store(result, ops, params, Tile, tile, FULL_TILE);
}

// A8 activations (MatmulA::Int8Symmetric) against U4 codes or U8 symmetric codes, its own entry point so the integer
// dot product capability stays out of the float modules. A workgroup computes 8 ROWS rows for INPUT_ROWS inputs; a lane
// folds the exact integer dots of its 32-value runs, each in one weight group, as int8_fold does.
static const uint GEMV_INT8_SIMDGROUPS = 8;
static const uint GEMV_INT8_RUN = 32;

// Split rows' partials: 256 for each row an invocation holds.
groupshared float gemv_int8_partials[2 * GEMV_THREADS];

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

// GemvInt8, invocation `invocation` of 256 in subgroup order: each output row of the workgroup's 8 ROWS lies in a slot
// of `held` rows taken by `parts` consecutive invocations, which split its runs and fold them for each of the
// INPUT_ROWS inputs; each invocation reads its rows' codes once for every input and each input's activations once for
// its rows. Split rows, of one input or 8 or more a simdgroup, hold one row, or two under several inputs, and sum their
// partials through shared memory an input at a time; consecutive invocations so read consecutive rows' group metadata.
// Otherwise a simdgroup holds its ROWS rows, its lanes every 32nd run, which segment_reduce sums. `has_rht`, of one
// input, finishes with ActivationTransform's output transform of the workgroup's whole 32-row blocks, then the bias:
// the outputs, rounded as stored, pass through shared memory to the transform's invocations.
[[Kernel]]
[[Precondition("new", "b_prologue != GemmBPrologueKind::FullPrecision && [32, 64, 128].contains(&group_size)")]]
[[Precondition("new", "quantization_mode == QuantizationMode::U4 || (quantization_mode == QuantizationMode::U8 && b_prologue == GemmBPrologueKind::ScaleSymmetricDequant)")]]
[[Precondition("new", "[1, 2, 4, 8, 16].contains(&ROWS) && (1..=8).contains(&INPUT_ROWS) && (!gathered || INPUT_ROWS == 1)")]]
[[Precondition("new", "!has_rht || INPUT_ROWS == 1 && ROWS >= 4")]]
[[Precondition("encode", "k.is_multiple_of(128) && (b_layout == GemmBLayout::RowMajor || n.is_multiple_of(32))")]]
[[Precondition("encode", "!has_rht || n.is_multiple_of(32)")]]
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
  [[Groups("m.div_ceil(INPUT_ROWS)")]] uint input_tile,
  [[Groups("n.div_ceil(8 * ROWS)")]] uint output_tile,
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
  [[Specialize]] bool has_rht,
) {
  let invocation = subgroup_invocation();
  let u4 = quantization_mode == QuantizationMode.U4;
  let method = gemm_quantization_method(b_prologue);
  let scale_bias = method == QuantizationMethod.ScaleBias;
  let weights = gemm_weights<B>(b, scales, biases, zero_points, group_size, scale_output_stride, scale_group_stride,
    zero_point_output_stride, zero_point_group_stride, quantization_mode, signed_codes, b_prologue, k, b_layout);
  // Under the output transform the bias adds after it.
  GemmParams params = { m, n, k, ab_scale, soft_cap, has_scale, has_bias && !has_rht, has_soft_cap, 0 };
  let split = INPUT_ROWS == 1 || ROWS >= 8;
  let held = split ? (INPUT_ROWS > 1 ? 2u : 1u) : ROWS;
  let parts = held * SIMD_SIZE / ROWS;
  let slot = invocation / parts;
  let part = invocation % parts;
  let first_row = output_tile * GEMV_INT8_SIMDGROUPS * ROWS + slot * held;
  let first_input = input_tile * INPUT_ROWS;
  uint rows[4];
  [ForceUnroll]
  for (uint R = 0; R < 4; R++) {
    // Gathered outputs of the one input row read the B rows its indices name.
    let clamped_row = min(first_row + R, n - 1);
    rows[R] = gathered && R < held ? gather[first_input * n + clamped_row] : clamped_row;
  }
  let run_bytes = u4 ? GEMV_INT8_RUN / 2 : GEMV_INT8_RUN;
  let codes = Ptr<uint8_t, Access.Read>(a);
  float sums[4][GEMV_MAX_INPUT_ROWS];
  [ForceUnroll]
  for (uint R = 0; R < 4; R++) {
    [ForceUnroll]
    for (uint I = 0; I < GEMV_MAX_INPUT_ROWS; I++) {
      sums[R][I] = 0.0;
    }
  }
  for (uint run = part; run < k / GEMV_INT8_RUN; run += parts) {
    let column = run * GEMV_INT8_RUN;
    let group = column / group_size;
    // A group's first run adds its ScaleBias correction.
    let head = scale_bias && column % group_size == 0;
    WeightGroup metadata[4];
    uint words[4][8];
    uint code_sums[4];
    [ForceUnroll]
    for (uint R = 0; R < 4; R++) {
      if (R < held) {
        metadata[R] = weights.metadata(rows[R], group);
        gemv_int8_words(weights.codes, weights.code_byte(rows[R], column), run_bytes, words[R]);
        code_sums[R] = unsigned_dot ? unsigned_codes(words[R], u4, signed_codes) : 0u;
      }
    }
    [ForceUnroll]
    for (uint I = 0; I < GEMV_MAX_INPUT_ROWS; I++) {
      let input = first_input + I;
      // A tile's first input is below m.
      if (I < INPUT_ROWS && (I == 0 || input < m)) {
        uint activation[8];
        gemv_int8_words(codes, input * k + column, GEMV_INT8_RUN, activation);
        let scale = a_scales[input * (k / ACTIVATION_QUANTIZATION_TILE) + column / ACTIVATION_QUANTIZATION_TILE];
        let group_sum = head ? a_group_sums[input * (k / group_size) + group] : 0;
        let activation_sum = unsigned_dot ? int(biased_activations(activation)) - 128 * int(GEMV_INT8_RUN) : 0;
        [ForceUnroll]
        for (uint R = 0; R < 4; R++) {
          if (R < held) {
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
            sums[R][I] = int8_fold(sums[R][I], scale, metadata[R], dot, head, group_sum);
          }
        }
      }
    }
  }
  if (!split) {
    // Lane I ROWS + R of each simdgroup finishes output row R of input I.
    var value = 0.0;
    [unroll]
    for (uint I = 0; I < INPUT_ROWS; I++) {
      [unroll]
      for (uint R = 0; R < ROWS; R++) {
        let total = segment_reduce<Sum<float>, GEMV_THREADS>(sums[R][I], SIMD_SIZE);
        if (part == I * ROWS + R) {
          value = total;
        }
      }
    }
    let input = first_input + part / ROWS;
    let row = first_row + part % ROWS;
    if (part < INPUT_ROWS * ROWS && input < m && row < n) {
      let bias_value = has_bias ? bias[row].toFloat() : 0.0;
      d[input * n + row] = D.fromFloat(gemm_output(value, bias_value, params));
    }
    return;
  }
  // Invocation `part` < held of a slot finishes the slot's row `part`.
  let row_local = slot * held + part;
  let output_row = first_row + part;
  [ForceUnroll]
  for (uint I = 0; I < GEMV_MAX_INPUT_ROWS; I++) {
    if (I < INPUT_ROWS) {
      let input = first_input + I;
      [ForceUnroll]
      for (uint R = 0; R < 4; R++) {
        if (R < held) {
          gemv_int8_partials[(slot * held + R) * parts + part] = sums[R][I];
        }
      }
      GroupMemoryBarrierWithGroupSync();
      var stored = 0.0;
      if (part < held && output_row < n && (I == 0 || input < m)) {
        var total = 0.0;
        for (uint other = 0; other < parts; other++) {
          total += gemv_int8_partials[row_local * parts + other];
        }
        let bias_value = params.has_bias ? bias[output_row].toFloat() : 0.0;
        let output = D.fromFloat(gemm_output(total, bias_value, params));
        if (!has_rht) {
          d[input * n + output_row] = output;
        }
        stored = output.toFloat();
      }
      if (has_rht) {
        // Every invocation has read its row's partials before the stored outputs overwrite them.
        GroupMemoryBarrierWithGroupSync();
        if (part < held) {
          gemv_int8_partials[row_local] = stored;
        }
        GroupMemoryBarrierWithGroupSync();
        let rows = GEMV_INT8_SIMDGROUPS * ROWS;
        let column = output_tile * rows + invocation;
        let value = hadamard_transform<GEMV_THREADS>(invocation < rows ? gemv_int8_partials[invocation] : 0.0);
        // n is whole blocks, so a block lies wholly in or past it.
        if (invocation < rows && column < n) {
          var output = hadamard_factor(value, rht_factors[column]);
          if (has_bias) {
            output = D.fromFloat(output).toFloat() + bias[column].toFloat();
          }
          d[input * n + column] = D.fromFloat(output);
        }
      } else if (I + 1 < INPUT_ROWS) {
        // Every invocation has read its row's partials before the next input's overwrite them.
        GroupMemoryBarrierWithGroupSync();
      }
    }
  }
}
