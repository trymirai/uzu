import definitions;
import common.activation;
import common.int8_code_layout;
import common.quantized_value;
import generated.quantization_method;

// The CPU MatmulKernel's per-output numerics shared by Gemv and Gemm: which outputs a native evaluation keeps, the CPU's
// order where it does not, and the epilogue.
//
// Native products and NoContraction sums agree with the CPU within rounding and reordering where no product can be
// subnormal: every nonzero operand normal and the smallest exponent fields of A's and B's nonzero operands summing to at
// least 153. Every nonzero product is then at least 2^-101, a multiple of 2^-124 like every partial sum, so none is below
// the smallest normal and nothing flushes; zeros multiply exactly. Where the CPU's order may overflow or meets an
// infinity or NaN, whose classes and overflow midpoints reordering does not keep, the owner replays the CPU's order: an
// operand field of 255, 2^23 products or more, a bound 2^(largest fields - 252 + ceil(log2 K)) on every partial sum past
// 2^127, or a bound on the later intermediates before the soft cap, that times |scale| plus |D| and |bias|, of 2^126 or
// more. The epilogue follows the CPU's order, each step rounded with subnormals kept.

public static const uint MATMUL_FAST = 0;
public static const uint MATMUL_CAREFUL = 1;
public static const uint MATMUL_REPLAY = 2;

// The exponent fields of a product's operands: the smallest of nonzero A and B operands (255 for zeros) in the low
// bytes, the largest in the high bytes. A zero operand contributes nothing, so fields(x, w) merges fields(x, 0) and
// fields(0, w).
public uint matmul_fields(float x, float w) {
  let x_bits = asuint(x) & 0x7fffffffu;
  let w_bits = asuint(w) & 0x7fffffffu;
  let x_field = x_bits >> 23;
  let w_field = w_bits >> 23;
  return (x_bits == 0 ? 0xffu : x_field) | (w_bits == 0 ? 0xffu : w_field) << 8 | x_field << 16 | w_field << 24;
}

// Fields combined: smallest of the low bytes, largest of the high bytes; 0x0000ffff is the identity.
public uint matmul_merge_fields(uint left, uint right) {
  return min(left & 0xffu, right & 0xffu) | min(left & 0xff00u, right & 0xff00u)
       | max(left & 0xff0000u, right & 0xff0000u) | max(left & 0xff000000u, right & 0xff000000u);
}

// The merged matmul_fields(x, 0.0) of A operands from the least of their |x| bits minus one, wrapping so that zeros
// give 0xffffffff, and the greatest |x| bits: the fields are those bits' exponents.
public uint matmul_a_fields(uint smallest_minus_one, uint largest) {
  let smallest = smallest_minus_one == 0xffffffffu ? 0xffu : (smallest_minus_one + 1u) >> 23;
  return smallest | 0xff00u | ((largest >> 23) << 16);
}

// Integer-code products and FP32 rounding only coarsen the metadata's power-of-two grid. A nonzero decoded weight
// therefore has exponent field at least min(metadata fields) minus its fraction bits. The sum's largest field is
// at most max(scale field + code bits, bias field) + 1. Zero-point corrections share the scale's grid, including when
// their staged bias is rounded to B. Actual staged fields are also merged by the loader.
public uint matmul_quantized_fields<B : IModelFloat>(float scale, float bias, uint bits) {
  let s = asuint(scale) & 0x7fffffffu;
  let b = asuint(bias) & 0x7fffffffu;
  let s_field = s >> 23;
  let b_field = b >> 23;
  let smallest = min(s == 0 ? 255u : s_field, b == 0 ? 255u : b_field);
  let fraction_bits = sizeof(B) == 2 ? 7u : 23u;
  let small = smallest > fraction_bits ? smallest - fraction_bits : 0u;
  let large = min(255u, max(s_field + bits, b_field) + 1u);
  return 0xffu | (small << 8) | (large << 24);
}

groupshared uint matmul_subgroup_fields[256];

// Every workgroup invocation participates. Merging a whole tile conservatively classifies each of its outputs.
public uint matmul_workgroup_fields(uint fields) {
  let merged = WaveActiveMin(fields & 0xffu) | WaveActiveMin(fields & 0xff00u)
    | WaveActiveMax(fields & 0xff0000u) | WaveActiveMax(fields & 0xff000000u);
  if (WaveGetLaneIndex() == 0) {
    matmul_subgroup_fields[subgroup_id()] = merged;
  }
  GroupMemoryBarrierWithGroupSync();
  uint result = 0x0000ffffu;
  for (uint group = 0; group < subgroup_count(); group++) {
    result = matmul_merge_fields(result, matmul_subgroup_fields[group]);
  }
  return result;
}

// An output's class from its operands' fields and its epilogue operands.
public uint matmul_mode(
  uint fields,
  uint k,
  bool has_scale,
  float ab_scale,
  bool accumulate,
  float prior,
  bool has_bias,
  float bias
) {
  let smallest_a = fields & 0xffu;
  let smallest_b = (fields >> 8) & 0xffu;
  let largest_a = (fields >> 16) & 0xffu;
  let largest_b = fields >> 24;
  // Every partial sum, in any order, stays below 2^exponent (1 + 2^-24)^(K + 2) < 2^127.72, under FP32 overflow, while
  // exponent <= 127 and K < 2^23, whatever the scale. Each later intermediate is below the bound scaled and summed, a
  // monotone chain whose native roundings and flushes lose under 2^-24 relative or 2^-126 per step (a flushed scale's
  // product below 2), far inside the factor 4 between 2^126 and overflow.
  let exponent = int(largest_a + largest_b + (k > 1 ? uint(firstbithigh(k - 1)) + 1 : 0u)) - 252;
  var bound = exponent < -126 ? 0.0 : asfloat(uint(min(exponent, 127) + 127) << 23);
  if (has_scale) {
    bound = abs(ab_scale) * bound;
  }
  if (accumulate) {
    bound += abs(prior);
  }
  if (has_bias) {
    bound += abs(bias);
  }
  if (largest_a == 0xffu || largest_b == 0xffu || k >= (1u << 23) || exponent > 127 || !(bound < asfloat(0x7e800000u))) {
    return MATMUL_REPLAY;
  }
  if (smallest_a == 0 || smallest_b == 0 || smallest_a + smallest_b < 153) {
    return MATMUL_CAREFUL;
  }
  return MATMUL_FAST;
}

// The FP32 value of an A or B element `inner` of row `row`, which starts at element `base`, as the CPU reads it.
public interface IMatmulOperand {
  public float value(uint row, uint inner, uint base);
}

// Full-precision rows of k elements.
public struct DenseOperand<T : IModelFloat> : IMatmulOperand {
  public Ptr<T, Access.Read> values;

  public float value(uint row, uint inner, uint base) {
    return values[base + inner].toFloat();
  }
}

// Symmetric INT8 A rows of k codes at int8_code_index, each times the FP32 scale of its group (base + inner) / group_size
// in one rounding, as the CPU decodes them; base is a multiple of 8 and of the group.
public struct Int8Activations : IMatmulOperand {
  public Ptr<int8_t, Access.Read> codes;
  public Ptr<float, Access.Read> scales;
  public uint group_size;
  public bool grouped_by_weight_nibble;

  public float value(uint row, uint inner, uint base) {
    let code = int(codes[base + int8_code_index(inner, grouped_by_weight_nibble)]);
    return code_product(scales[(base + inner) / group_size], code);
  }
}

// Packed U4 or U8 codes of B rows of k elements, code i = row k + inner, a U4 code the low nibble of byte i / 2 first,
// stored signed codes with their top bit flipped. Scales, biases and zero points (nibbles for U4) of group
// inner / group_size sit at row output_stride + group group_stride, each plane with its own strides; without zero points
// the offset is the midpoint, 8 or 128. The value is quantized_value's, in FP32, never rounded to B.
public struct QuantizedWeights<B : IModelFloat> : IMatmulOperand {
  public Ptr<uint8_t, Access.Read> codes;
  public Ptr<B, Access.Read> scales;
  public Ptr<B, Access.Read> biases;
  public Ptr<uint8_t, Access.Read> zero_points;
  public uint group_size;
  public uint scale_output_stride;
  public uint scale_group_stride;
  public uint zero_point_output_stride;
  public uint zero_point_group_stride;
  public bool u4;
  public bool signed_codes;
  public QuantizationMethod method;

  public float value(uint row, uint inner, uint base) {
    let index = base + inner;
    var code = u4 ? read_nibble(codes, index) : uint(codes[index]);
    if (signed_codes) {
      code ^= u4 ? 8u : 128u;
    }
    let group = inner / group_size;
    let metadata = row * scale_output_stride + group * scale_group_stride;
    let scale = scales[metadata].toFloat();
    let scale_bias = method == QuantizationMethod.ScaleBias;
    float correction;
    if (scale_bias) {
      correction = biases[metadata].toFloat();
    } else if (method == QuantizationMethod.ScaleZeroPoint) {
      let offset = row * zero_point_output_stride + group * zero_point_group_stride;
      correction = float(u4 ? read_nibble(zero_points, offset) : uint(zero_points[offset]));
    } else {
      correction = u4 ? 8.0 : 128.0;
    }
    return quantized_value(scale, int(code), scale_bias, correction);
  }
}

// The CPU's dot of A row `a_row` and B row `b_row`, in its order with every product and sum rounded as it rounds them.
[noinline]
public float matmul_replay<X : IMatmulOperand, W : IMatmulOperand>(
  X activations,
  uint a_row,
  W weights,
  uint b_row,
  uint k
) {
  var dot = 0.0;
  let a_base = a_row * k;
  let b_base = b_row * k;
  for (uint j = 0; j < k; j++) {
    dot = ordered_sum(dot, exact_product(activations.value(a_row, j, a_base), weights.value(b_row, j, b_base)));
  }
  return dot;
}

// v / cap as the CPU divides. Classes are decided on the bits, as IEEE gives them: a NaN operand, 0 / 0 and inf / inf
// are NaN, x / 0 and inf / x infinite, x / inf zero. Finite nonzero operands divide with the device seeing no operand or
// quotient it may flush and only denominators where Vulkan bounds the division, [2^-126, 2^126]: subnormal operands
// scaled up by 2^24 exactly, a quotient that may be subnormal formed scaled up and rounded back down, and larger
// denominators through quarter_quotient.
float matmul_quotient(float v, float cap) {
  let v_magnitude = asuint(v) & 0x7fffffffu;
  let cap_magnitude = asuint(cap) & 0x7fffffffu;
  let sign = (asuint(v) ^ asuint(cap)) & 0x80000000u;
  if (max(v_magnitude, cap_magnitude) > 0x7f800000u || (v_magnitude == cap_magnitude && (v_magnitude == 0 || v_magnitude == 0x7f800000u))) {
    return asfloat(0x7fc00000u);
  }
  if (v_magnitude == 0x7f800000u || cap_magnitude == 0) {
    return asfloat(sign | 0x7f800000u);
  }
  if (v_magnitude == 0 || cap_magnitude == 0x7f800000u) {
    return asfloat(sign);
  }
  let v_subnormal = v_magnitude < 0x00800000u;
  if (cap_magnitude < 0x00800000u) {
    let scaled_cap = exact_scale_up_24(cap);
    return v_subnormal ? exact_scale_up_24(v) / scaled_cap : v / scaled_cap * 16777216.0;
  }
  let large = cap_magnitude > 0x7e800000u;
  if (v_subnormal || (v_magnitude >> 23) + 126 <= cap_magnitude >> 23) {
    let numerator = exact_scale_up_24(v);
    return exact_scale_down(large ? quarter_quotient(numerator, cap) : numerator / cap, 24);
  }
  return large ? quarter_quotient(v, cap) : v / cap;
}

// The CPU's cap tanh(v / cap). Below 2^-12 tanh rounds to its argument in FP32, which is kept bit for bit.
float matmul_soft_cap(float v, float cap) {
  let q = matmul_quotient(v, cap);
  return exact_product(cap, (asuint(q) & 0x7fffffffu) < 0x39800000u ? q : accurate_tanh(q));
}

// The CPU's epilogue on a dot: scale, the prior D, bias and soft cap, each step rounded with subnormals kept.
[noinline]
public float matmul_epilogue(
  float dot,
  bool has_scale,
  float ab_scale,
  bool accumulate,
  float prior,
  bool has_bias,
  float bias,
  bool has_soft_cap,
  float soft_cap
) {
  var value = has_scale ? exact_product(ab_scale, dot) : dot;
  if (accumulate) {
    value = ordered_sum(value, prior);
  }
  if (has_bias) {
    value = ordered_sum(value, bias);
  }
  if (has_soft_cap) {
    value = matmul_soft_cap(value, soft_cap);
  }
  return value;
}

// A threadgroup array of T, element `index` from its start.
public interface IThreadgroupStorage<T> {
  public static T read(uint index);
  public static void write(uint index, T value);
}


// Existing reader-based GEMM for A8, arbitrary strides and shapes outside the Metal quantized layout.
static const uint GEMM_THREADS = 256;
// Rows and columns of D per workgroup, as the binding's group counts m.div_ceil(64) and n.div_ceil(64) assume.
static const uint GEMM_TILE = 64;
static const uint GEMM_DEPTH = 16;
static const uint GEMM_STRIDE = 16;
static const uint GEMM_SPAN = GEMM_TILE / GEMM_STRIDE;

// A's and B's tile rows of one slice, depth-major: [depth * GEMM_TILE + tile row].
public struct CanonicalGemmStorage : IThreadgroupStorage<float> {
  static groupshared float data[2 * GEMM_DEPTH * GEMM_TILE];
  public static float read(uint index) { return data[index]; }
  public static void write(uint index, float value) { data[index] = value; }
}

// sum + x w, the product and the sum each rounded natively, never fused.
float gemm_step(float sum, float x, float w) {
  return spirv_asm {
    %product:$$float = OpFMul $x $w;
    OpDecorate %product NoContraction;
    %sum:$$float = OpFAdd $sum %product;
    OpDecorate %sum NoContraction;
    result:$$float = OpCopyObject %sum;
  };
}

public void gemm_body<X : IMatmulOperand, B : IModelFloat, D : IModelFloat, W : IMatmulOperand, S : IThreadgroupStorage<float> = CanonicalGemmStorage>(
  W weights,
  X activations,
  Ptr<D, Access.ReadWrite> d,
  Ptr<B, Access.Read> bias,
  uint k,
  uint n,
  uint m,
  float ab_scale,
  float soft_cap,
  uint column_tile,
  uint row_tile,
  uint tid,
  bool has_scale,
  bool accumulate,
  bool has_bias,
  bool has_soft_cap,
) {
  let first_row = row_tile * GEMM_TILE;
  let first_column = column_tile * GEMM_TILE;
  // Each slice, this invocation loads GEMM_SPAN consecutive K elements from `slot` of tile row `loaded` of A and of B.
  let loaded = tid / GEMM_SPAN;
  let slot = tid % GEMM_SPAN * GEMM_SPAN;
  let a_row = first_row + loaded;
  let b_row = first_column + loaded;
  // It owns tile rows owner_row + GEMM_STRIDE i and tile columns owner_column + GEMM_STRIDE j.
  let owner_row = tid / GEMM_STRIDE;
  let owner_column = tid % GEMM_STRIDE;
  // Row groups from here hold no row of A in every invocation alike.
  let live_rows = m - first_row;
  float sums[GEMM_SPAN][GEMM_SPAN];
  [unroll]
  for (uint i = 0; i < GEMM_SPAN; i++) {
    [unroll]
    for (uint j = 0; j < GEMM_SPAN; j++) {
      sums[i][j] = 0.0;
    }
  }
  var a_fields = 0x0000ffffu;
  var b_fields = 0x0000ffffu;
  for (uint base = 0; base < k; base += GEMM_DEPTH) {
    [unroll]
    for (uint q = 0; q < GEMM_SPAN; q++) {
      let j = base + slot + q;
      var x = 0.0;
      var w = 0.0;
      if (a_row < m && j < k) {
        x = activations.value(a_row, j, a_row * k);
      }
      if (b_row < n && j < k) {
        w = weights.value(b_row, j, b_row * k);
      }
      a_fields = matmul_merge_fields(a_fields, matmul_fields(x, 0.0));
      b_fields = matmul_merge_fields(b_fields, matmul_fields(0.0, w));
      S.write((slot + q) * GEMM_TILE + loaded, x);
      S.write(GEMM_DEPTH * GEMM_TILE + (slot + q) * GEMM_TILE + loaded, w);
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint depth = 0; depth < GEMM_DEPTH; depth++) {
      float w[GEMM_SPAN];
      [unroll]
      for (uint j = 0; j < GEMM_SPAN; j++) {
        w[j] = S.read(GEMM_DEPTH * GEMM_TILE + depth * GEMM_TILE + owner_column + GEMM_STRIDE * j);
      }
      [unroll]
      for (uint i = 0; i < GEMM_SPAN; i++) {
        if (GEMM_STRIDE * i < live_rows) {
          let x = S.read(depth * GEMM_TILE + owner_row + GEMM_STRIDE * i);
          [unroll]
          for (uint j = 0; j < GEMM_SPAN; j++) {
            sums[i][j] = gemm_step(sums[i][j], x, w[j]);
          }
        }
      }
    }
    GroupMemoryBarrierWithGroupSync();
  }
  let tile_fields = matmul_workgroup_fields(matmul_merge_fields(a_fields, b_fields));

  [unroll]
  for (uint i = 0; i < GEMM_SPAN; i++) {
    [unroll]
    for (uint j = 0; j < GEMM_SPAN; j++) {
      let tile_row = owner_row + GEMM_STRIDE * i;
      let tile_column = owner_column + GEMM_STRIDE * j;
      let row = first_row + tile_row;
      let column = first_column + tile_column;
      if (row < m && column < n) {
        let fields = tile_fields;
        var prior = 0.0;
        if (accumulate) {
          prior = d[row * n + column].toFloat();
        }
        var bias_value = 0.0;
        if (has_bias) {
          bias_value = bias[column].toFloat();
        }
        var dot = sums[i][j];
        if (matmul_mode(fields, k, has_scale, ab_scale, accumulate, prior, has_bias, bias_value) != MATMUL_FAST) {
          dot = matmul_replay(activations, row, weights, column, k);
        }
        let value =
          matmul_epilogue(dot, has_scale, ab_scale, accumulate, prior, has_bias, bias_value, has_soft_cap, soft_cap);
        d[row * n + column] = D.fromFloat(value);
      }
    }
  }
}
