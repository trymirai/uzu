import definitions;
import common.activation;

// The CPU MatmulKernel's full-precision product of one A row and transposed B rows, D = epilogue(A Bᵀ), for every
// BF16/F32 triple. A workgroup computes GEMV_OUTPUTS outputs of one A row, GEMV_LANES invocations each striding K; the
// partial sums meet in shared memory in invocation order, assuming nothing about subgroups.
//
// The native products and NoContraction sums of the parallel path agree with the CPU within rounding and reordering
// where no product can be subnormal: every nonzero operand normal and the smallest exponent fields of A's and B's
// nonzero operands summing to at least 153. Every nonzero product is then at least 2^-101, a multiple of 2^-124 like
// every partial sum, so none is below the smallest normal and nothing flushes; zeros multiply exactly. Otherwise the
// outputs' products are formed exactly with subnormals kept, those below 2^-101 summed scaled up by 2^24, where no
// partial flushes either, and added back with one rounding. Where the CPU's order may overflow or meets an infinity or
// NaN, whose classes and overflow midpoints reordering does not keep, the owner replays the CPU's order: an operand
// field of 255, 2^23 products or more, a bound 2^(largest fields - 252 + ceil(log2 K)) on every partial sum past
// 2^127, or a bound on the later intermediates before the soft cap, that times |scale| plus |D| and |bias|, of 2^126 or
// more. The epilogue follows the CPU's order, each step rounded with subnormals kept.

static const uint GEMV_THREADS = 256;
// Output rows per workgroup, as the binding's group count n.div_ceil(8) assumes.
static const uint GEMV_OUTPUTS = 8;
static const uint GEMV_LANES = GEMV_THREADS / GEMV_OUTPUTS;
static const uint GEMV_FAST = 0;
static const uint GEMV_CAREFUL = 1;
static const uint GEMV_REPLAY = 2;

groupshared float gemv_main[GEMV_THREADS];
groupshared float gemv_tiny[GEMV_THREADS];
// Per invocation: the smallest exponent fields of its nonzero A and B operands (255 if none) and the largest ones.
groupshared uint gemv_fields[GEMV_THREADS];
groupshared uint gemv_mode[GEMV_OUTPUTS];

uint gemv_fields_of(float x, float w) {
  let x_bits = asuint(x) & 0x7fffffffu;
  let w_bits = asuint(w) & 0x7fffffffu;
  let x_field = x_bits >> 23;
  let w_field = w_bits >> 23;
  return (x_bits == 0 ? 0xffu : x_field) | (w_bits == 0 ? 0xffu : w_field) << 8 | x_field << 16 | w_field << 24;
}

// Fields of two invocations combined: smallest of the low bytes, largest of the high bytes.
uint gemv_merge_fields(uint left, uint right) {
  return min(left & 0xffu, right & 0xffu) | min(left & 0xff00u, right & 0xff00u)
       | max(left & 0xff0000u, right & 0xff0000u) | max(left & 0xff000000u, right & 0xff000000u);
}

// v / cap as the CPU divides. Classes are decided on the bits, as IEEE gives them: a NaN operand, 0 / 0 and inf / inf
// are NaN, x / 0 and inf / x infinite, x / inf zero. Finite nonzero operands divide with the device seeing no operand or
// quotient it may flush and only denominators where Vulkan bounds the division, [2^-126, 2^126]: subnormal operands
// scaled up by 2^24 exactly, a quotient that may be subnormal formed scaled up and rounded back down, and larger
// denominators through quarter_quotient.
float gemv_quotient(float v, float cap) {
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
float gemv_soft_cap(float v, float cap) {
  let q = gemv_quotient(v, cap);
  return exact_product(cap, (asuint(q) & 0x7fffffffu) < 0x39800000u ? q : accurate_tanh(q));
}

[[Kernel]] func Gemv<A: IModelFloat, B: IModelFloat, D: IModelFloat>(
  Ptr<B, Access.Read> b,
  Ptr<A, Access.Read> a,
  Ptr<D, Access.ReadWrite> d,
  [[Optional("has_bias")]] Ptr<B, Access.Read> bias,
  [[Optional("gathered")]] Ptr<uint, Access.Read> gather,
  uint k,
  uint n,
  uint m,
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
  let output = tid / GEMV_LANES;
  let lane = tid % GEMV_LANES;
  let column = tile * GEMV_OUTPUTS + output;
  let active = column < n;
  let a_base = row * k;
  uint b_base = 0;
  var partial = 0.0;
  var fields = 0x0000ffffu;
  if (active) {
    b_base = (gathered ? gather[row * n + column] : column) * k;
    for (uint j = lane; j < k; j += GEMV_LANES) {
      let x = a[a_base + j].toFloat();
      let w = b[b_base + j].toFloat();
      fields = gemv_merge_fields(fields, gemv_fields_of(x, w));
      partial = ordered_sum(partial, x * w);
    }
  }
  gemv_main[tid] = partial;
  gemv_fields[tid] = fields;
  GroupMemoryBarrierWithGroupSync();

  var dot = 0.0;
  if (lane == 0) {
    var mode = GEMV_FAST;
    if (active) {
      for (uint other = tid; other < tid + GEMV_LANES; other++) {
        dot = ordered_sum(dot, gemv_main[other]);
        fields = gemv_merge_fields(fields, gemv_fields[other]);
      }
      let smallest_a = fields & 0xffu;
      let smallest_b = (fields >> 8) & 0xffu;
      let largest_a = (fields >> 16) & 0xffu;
      let largest_b = fields >> 24;
      // Every partial sum, in any order, stays below 2^exponent (1 + 2^-24)^(K + 2) < 2^127.72, under FP32 overflow, while
      // exponent <= 127 and K < 2^23, whatever the scale. Each later intermediate is below the bound scaled and
      // summed, a monotone chain whose native roundings and flushes lose under 2^-24 relative or 2^-126 per step (a
      // flushed scale's product below 2), far inside the factor 4 between 2^126 and overflow.
      let exponent = int(largest_a + largest_b + (k > 1 ? uint(firstbithigh(k - 1)) + 1 : 0u)) - 252;
      var bound = exponent < -126 ? 0.0 : asfloat(uint(min(exponent, 127) + 127) << 23);
      if (has_scale) {
        bound = abs(ab_scale) * bound;
      }
      if (accumulate) {
        bound += abs(d[row * n + column].toFloat());
      }
      if (has_bias) {
        bound += abs(bias[column].toFloat());
      }
      if (largest_a == 0xffu || largest_b == 0xffu || k >= (1u << 23) || exponent > 127 || !(bound < asfloat(0x7e800000u))) {
        mode = GEMV_REPLAY;
      } else if (smallest_a == 0 || smallest_b == 0 || smallest_a + smallest_b < 153) {
        mode = GEMV_CAREFUL;
      }
    }
    gemv_mode[output] = mode;
  }
  GroupMemoryBarrierWithGroupSync();

  let mode = gemv_mode[output];
  var careful = false;
  for (uint other = 0; other < GEMV_OUTPUTS; other++) {
    careful = careful || gemv_mode[other] == GEMV_CAREFUL;
  }
  if (careful) {
    var main = 0.0;
    var tiny = 0.0;
    if (mode == GEMV_CAREFUL) {
      for (uint j = lane; j < k; j += GEMV_LANES) {
        let product = exact_product(a[a_base + j].toFloat(), b[b_base + j].toFloat());
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
    if (lane == 0 && mode == GEMV_CAREFUL) {
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
    if (mode == GEMV_REPLAY) {
      dot = 0.0;
      for (uint j = 0; j < k; j++) {
        dot = ordered_sum(dot, exact_product(a[a_base + j].toFloat(), b[b_base + j].toFloat()));
      }
    }
    var value = has_scale ? exact_product(ab_scale, dot) : dot;
    if (accumulate) {
      value = ordered_sum(value, d[row * n + column].toFloat());
    }
    if (has_bias) {
      value = ordered_sum(value, bias[column].toFloat());
    }
    if (has_soft_cap) {
      value = gemv_soft_cap(value, soft_cap);
    }
    d[row * n + column] = D.fromFloat(value);
  }
}
