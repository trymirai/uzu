import generated.activation_type;

// FP32 activation math in the operation order of the CPU ActivationType::activate, which every kernel applying an
// activation or a tanh soft cap shares.

// Below |x| = 0.1 the builtin tanh loses relative accuracy, so it is replaced there by its degree-5 Taylor polynomial
// x (1 - x²/3 + 2x⁴/15), whose omitted 17x⁷/315 term is under 5.5e-8 relative; the factored form keeps the sign of zero.
// From log(3) / 2 on, where the builtin is off by ULPs that 1 + tanh would keep and saturates early, it is
// 1 - 2 / (e^2|x| + 1): the quotient is at most 1/2, so the subtraction cancels nothing, and an infinite exponential gives
// exactly ±1.
public float accurate_tanh(float x) {
  let a = abs(x);
  if (a < 0.1) {
    let s = x * x;
    return x * fma(s, fma(s, 2.0 / 15.0, -1.0 / 3.0), 1.0);
  }
  if (a < 0.54930615) {
    return tanh(x);
  }
  return copysign(1.0 - 2.0 / (exp(2.0 * a) + 1.0), x);
}

// Faithfully rounded erff polynomial from Norbert Juffa, ported from the Metal erf_precise, also used by MLX. From
// |a| = 4 on it returns exactly ±1, the correctly rounded value there (erfc(4) = 1.5e-8 is below half the FP32 spacing
// under 1).
public float erf_precise(float a) {
  let t = abs(a);
  let s = a * a;
  if (t >= 4.0) {
    return copysign(1.0, a);
  }
  if (t > 0.927734375) {
    var r = fma(-1.72853470e-5, t, 3.83197126e-4);
    let u = fma(-3.88396438e-3, t, 2.42546219e-2);
    r = fma(r, s, u);
    r = fma(r, t, -1.06777877e-1);
    r = fma(r, t, -6.34846687e-1);
    r = fma(r, t, -1.28717512e-1);
    r = fma(r, t, -t);
    return copysign(1.0 - exp(r), a);
  }
  var r = -5.96761703e-4;
  r = fma(r, s, 4.99119423e-3);
  r = fma(r, s, -2.67681349e-2);
  r = fma(r, s, 1.12819925e-1);
  r = fma(r, s, -3.76125336e-1);
  r = fma(r, s, 1.28379166e-1);
  return fma(r, a, a);
}

// Exact FP32 arithmetic on bits for the cases Vulkan FP32 arithmetic may flush: subnormal operands and results, which
// this device class flushes without a denormal-preserving mode. Normal operands keep the native fast paths.

// The 64-bit product of two 32-bit integers. The pinned Slang 2026.2 SDK has no umulExtended; the SPIR-V instruction
// needs no Int64.
public uint2 multiply_extended(uint a, uint b) {
  return spirv_asm {
    %pair_type = OpTypeStruct $$uint $$uint;
    %pair:%pair_type = OpUMulExtended $a $b;
    %low:$$uint = OpCompositeExtract %pair 0;
    %high:$$uint = OpCompositeExtract %pair 1;
    result:$$uint2 = OpCompositeConstruct %low %high;
  };
}

// The significand of positive finite nonzero FP32 bits with its implicit bit, in [2^23, 2^24), and the exponent e of
// x = significand 2^(e - 23); subnormals are normalized through their bits.
public uint significand(uint bits, out int exponent) {
  if (bits >= 0x00800000u) {
    exponent = int(bits >> 23) - 127;
    return (bits & 0x007fffffu) | 0x00800000u;
  }
  let shift = 23 - int(firstbithigh(bits));
  exponent = -126 - shift;
  return bits << uint(shift);
}

// The integer (high, low) / 2^shift rounded to nearest even, for values below 2^48 and shifts from 1.
uint round_shift(uint high, uint low, uint shift) {
  if (shift > 49) {
    return 0;
  }
  var sticky = false;
  if (shift > 32) {
    sticky = low != 0;
    low = high;
    high = 0;
    shift -= 32;
  }
  let kept = shift == 32 ? high : high << (32 - shift) | low >> shift;
  let half = (shift == 32 ? low >> 31 : low >> (shift - 1)) & 1;
  sticky = sticky || (low & ((1u << (shift - 1)) - 1)) != 0;
  return kept + (half != 0 && (sticky || (kept & 1) != 0) ? 1u : 0u);
}

// x / 2^shift rounded to nearest even through the bits, for shifts 1 to 24: an exponent decrement while the result stays
// normal, otherwise the significand shifted onto the subnormal grid, a carry rounding up into the smallest normal. Zeros
// keep their sign; from shift 2 on infinities and NaNs return unchanged, while halving takes finite magnitudes only.
public float exact_scale_down(float x, uint shift) {
  let bits = asuint(x);
  let magnitude = bits & 0x7fffffffu;
  let field = magnitude >> 23;
  if (shift == 1) {
    // Halving, whose callers pass only finite x (the GELUs' tiny x, SiLU's x under a tiny exponent -alpha x, which a
    // zero or tiny alpha gives any finite x): from 2^-125 on an exponent decrement, below it the bits halved with the
    // dropped bit rounded to even, also into the smallest normal.
    if (magnitude >= 0x01000000u) {
      return asfloat(bits - 0x00800000u);
    }
    return asfloat((bits & 0x80000000u) | ((magnitude >> 1) + (magnitude & (magnitude >> 1) & 1u)));
  }
  if (field == 0xffu) {
    return x;
  }
  if (field > shift) {
    return asfloat(bits - (shift << 23));
  }
  // The significand on the 2^-149 grid, at most 24 bits, shifted right by at most 25.
  let units = field == 0 ? magnitude : (magnitude & 0x007fffffu) | 0x00800000u;
  let right = shift + 1 - max(field, 1);
  let kept = units >> right;
  let rest = units & ((1u << right) - 1);
  let half = 1u << (right - 1);
  let round_up = rest > half || (rest == half && (kept & 1) != 0);
  return asfloat((bits & 0x80000000u) | (kept + (round_up ? 1u : 0u)));
}

// x 2^24 exactly, for finite x below 2^103 in magnitude: an exponent increment for normals, a subnormal's fraction
// converted to FP32 exactly and scaled by 2^-125. The sign is restored through the bits.
public float exact_scale_up_24(float x) {
  let bits = asuint(x);
  if ((bits & 0x7f800000u) != 0) {
    return asfloat(bits + (24u << 23));
  }
  return asfloat(asuint(float(bits & 0x007fffffu) * asfloat(0x01000000u)) | (bits & 0x80000000u));
}

// a + b or a - b rounded to nearest even, subnormals kept. When both magnitudes are below 2^-101 both are scaled up by
// 2^24 exactly, so the native operation sees normals: a sum below the smallest normal is exact on the 2^-149 grid and a
// larger one rounds as the scaled one, so scaling back down is exact. Otherwise any subnormal operand is below half the
// other's ULP (at least 2^-125 below a power of two), leaving the rounded sum unchanged, and a nonzero cancellation of
// operands from 2^-101 on is at least the smallest normal.
public float exact_sum(float a, float b, bool subtract) {
  // A branch, not flattened: the scaled path is rare and would otherwise run for every sum.
  [branch]
  if (max(asuint(a) & 0x7fffffffu, asuint(b) & 0x7fffffffu) < 0x0d000000u) {
    let scaled_a = exact_scale_up_24(a);
    let scaled_b = exact_scale_up_24(b);
    return exact_scale_down(subtract ? scaled_a - scaled_b : scaled_a + scaled_b, 24);
  }
  return subtract ? a - b : a + b;
}

// a + b as exact_sum, its native sum decorated NoContraction so it is never fused with the product feeding it: ordered
// paths round each product and each sum as the CPU does.
public float ordered_sum(float a, float b) {
  [branch]
  if (max(asuint(a) & 0x7fffffffu, asuint(b) & 0x7fffffffu) < 0x0d000000u) {
    return exact_sum(a, b, false);
  }
  return spirv_asm {
    %sum:$$float = OpFAdd $a $b;
    OpDecorate %sum NoContraction;
    result:$$float = OpCopyObject %sum;
  };
}

// a * b rounded to nearest even as IEEE FP32, subnormals kept. Natively unless an operand is subnormal or the product
// may be (exponent fields summing below 128); then the significands' 48-bit product is rounded onto the result's grid.
// Classes are decided on the bits: a nonzero subnormal times infinity is infinite, zero times infinity NaN.
public float exact_product(float a, float b) {
  let a_bits = asuint(a);
  let b_bits = asuint(b);
  let a_field = (a_bits >> 23) & 0xffu;
  let b_field = (b_bits >> 23) & 0xffu;
  // A branch, not flattened, as for every rare exact path here: otherwise the exact product runs for every product.
  [branch]
  if (a_field != 0 && b_field != 0 && a_field + b_field >= 128) {
    return a * b;
  }
  let sign = (a_bits ^ b_bits) & 0x80000000u;
  let a_magnitude = a_bits & 0x7fffffffu;
  let b_magnitude = b_bits & 0x7fffffffu;
  if (a_magnitude > 0x7f800000u || b_magnitude > 0x7f800000u) {
    return a * b;
  }
  if (a_magnitude == 0x7f800000u || b_magnitude == 0x7f800000u) {
    return asfloat(a_magnitude == 0 || b_magnitude == 0 ? 0x7fc00000u : sign | 0x7f800000u);
  }
  if (a_magnitude == 0 || b_magnitude == 0) {
    return asfloat(sign);
  }
  int a_exponent;
  int b_exponent;
  let product = multiply_extended(significand(a_magnitude, a_exponent), significand(b_magnitude, b_exponent));
  // The product of significands lies in [2^46, 2^48): keep 24 bits, fewer on the subnormal grid.
  let top = product.y >= (1u << 15) ? 1 : 0;
  var biased = a_exponent + b_exponent + top + 127;
  var shift = uint(23 + top);
  if (biased < 1) {
    shift += uint(1 - biased);
    biased = 1;
  }
  return asfloat(sign | (uint(biased - 1) << 23) + round_shift(product.y, product.x, shift));
}

// A 64-bit integer as (low, high) words shifted right by any count, `dropped` also set when a nonzero bit is lost: the
// exact fused multiply-add aligns its smaller term and compresses its sum with it. No shift reaches a word's width.
uint2 shift_right_wide(uint2 value, uint count, inout bool dropped) {
  if (count == 0) {
    return value;
  }
  if (count >= 64) {
    dropped = dropped || value.x != 0 || value.y != 0;
    return uint2(0, 0);
  }
  if (count >= 32) {
    let rest = count - 32;
    dropped = dropped || value.x != 0;
    if (rest != 0) {
      dropped = dropped || (value.y << (32 - rest)) != 0;
    }
    return uint2(value.y >> rest, 0);
  }
  dropped = dropped || (value.x << (32 - count)) != 0;
  return uint2((value.x >> count) | (value.y << (32 - count)), value.y >> count);
}

// a * b + c rounded once to nearest even, the IEEE FP32 fusedMultiplyAdd of the CPU f32::mul_add, on integers only: no
// device fusion, contraction or subnormal flushing reaches it. Each finite term is an exact integer with its top bit at
// bit 62 of a 64-bit window; the smaller is shifted onto the larger's grid, exactly unless it drops nonzero bits, which
// takes more than 15 places for the product or 39 for c and leaves it below 2^47. The exact sum then lies strictly
// between the window sum S >= 2^61 and S + 1. S is compressed below 2^47 with every dropped bit folded into bit 0, under
// the guard bit, and rounded onto the result's grid, 2^-149 for subnormals; a carry rounds into the exponent and results
// from 2^128 on are infinite. Exact cancellation gives +0. Classes follow IEEE: NaN for a NaN operand, infinity times
// zero or opposite infinities; a zero product keeps c, or gives -0 only from -0 + -0.
public float exact_fma(float a, float b, float c) {
  let a_bits = asuint(a);
  let b_bits = asuint(b);
  let c_bits = asuint(c);
  let a_magnitude = a_bits & 0x7fffffffu;
  let b_magnitude = b_bits & 0x7fffffffu;
  let c_magnitude = c_bits & 0x7fffffffu;
  let product_sign = (a_bits ^ b_bits) & 0x80000000u;
  let c_sign = c_bits & 0x80000000u;
  if (a_magnitude > 0x7f800000u || b_magnitude > 0x7f800000u || c_magnitude > 0x7f800000u) {
    return asfloat(0x7fc00000u);
  }
  if (a_magnitude == 0x7f800000u || b_magnitude == 0x7f800000u) {
    let invalid = a_magnitude == 0 || b_magnitude == 0 || (c_magnitude == 0x7f800000u && c_sign != product_sign);
    return asfloat(invalid ? 0x7fc00000u : product_sign | 0x7f800000u);
  }
  if (c_magnitude == 0x7f800000u) {
    return asfloat(c_bits);
  }
  if (a_magnitude == 0 || b_magnitude == 0) {
    return asfloat(c_magnitude != 0 ? c_bits : product_sign & c_bits);
  }
  int a_exponent;
  int b_exponent;
  let product = multiply_extended(significand(a_magnitude, a_exponent), significand(b_magnitude, b_exponent));
  // The significands' product lies in [2^46, 2^48) and weighs 2^(a_exponent + b_exponent - 46); bit 62 weighs 2^top.
  let lift = product.y >= (1u << 15) ? 15u : 16u;
  var larger = uint2(product.x << lift, (product.y << lift) | (product.x >> (32 - lift)));
  var top = a_exponent + b_exponent + 16 - int(lift);
  var sign = product_sign;
  var smaller = uint2(0, 0);
  var smaller_sign = sign;
  var sticky = false;
  if (c_magnitude != 0) {
    int c_exponent;
    let addend = uint2(0, significand(c_magnitude, c_exponent) << 7);
    if (c_exponent > top) {
      smaller = shift_right_wide(larger, uint(c_exponent - top), sticky);
      smaller_sign = sign;
      larger = addend;
      top = c_exponent;
      sign = c_sign;
    } else {
      smaller = shift_right_wide(addend, uint(top - c_exponent), sticky);
      smaller_sign = c_sign;
    }
  }
  uint2 sum;
  if (sign == smaller_sign) {
    let low = larger.x + smaller.x;
    sum = uint2(low, larger.y + smaller.y + (low < smaller.x ? 1u : 0u));
  } else {
    // Only terms sharing a top bit, never sticky, may order either way: the difference takes the larger one's sign.
    if (smaller.y > larger.y || (smaller.y == larger.y && smaller.x > larger.x)) {
      let swapped = larger;
      larger = smaller;
      smaller = swapped;
      sign = smaller_sign;
    }
    sum = uint2(larger.x - smaller.x, larger.y - smaller.y - (larger.x < smaller.x ? 1u : 0u));
    if (sticky) {
      sum = uint2(sum.x - 1, sum.y - (sum.x == 0 ? 1u : 0u));
    }
  }
  if (sum.x == 0 && sum.y == 0) {
    return asfloat(0u);
  }
  // The window's lowest bit weighs 2^(top - 62); the result's lowest weighs 2^grid, 24 bits below its top or 2^-149.
  let msb = sum.y != 0 ? 32 + int(firstbithigh(sum.y)) : int(firstbithigh(sum.x));
  let grid = max(msb + top - 85, -149);
  let compress = max(msb - 46, 0);
  var dropped = sticky;
  let kept = shift_right_wide(sum, uint(compress), dropped);
  let shift = grid - (top - 62) - compress;
  var rounded = 0u;
  if (shift <= 0) {
    rounded = kept.x << uint(-shift);
  } else {
    rounded = round_shift(kept.y, kept.x | (dropped ? 1u : 0u), uint(shift));
  }
  return asfloat(sign | min((uint(grid + 149) << 23) + rounded, 0x7f800000u));
}

// The log of the FP32 sum s = 1 + e^x as the CPU rounds it. Below 2, where the builtin log keeps only an absolute error,
// it is 2 atanh(w) = 2w (1 + w²/3 + ... + w¹²/13) with w = f / (2 + f) <= 1/3, where f = s - 1 is read exactly from the
// mantissa bits; the omitted terms are under 2^-25 relative.
float staged_log(float s) {
  if (!(s < 2.0)) {
    return log(s);
  }
  let f = float(asuint(s) & 0x007fffffu) * asfloat(0x34000000u);
  let w = f / (2.0 + f);
  let w2 = w * w;
  var r = 1.0 / 13.0;
  r = fma(r, w2, 1.0 / 11.0);
  r = fma(r, w2, 1.0 / 9.0);
  r = fma(r, w2, 1.0 / 7.0);
  r = fma(r, w2, 1.0 / 5.0);
  r = fma(r, w2, 1.0 / 3.0);
  r = fma(r, w2, 1.0);
  return 2.0 * w * r;
}

// (x / 4) / (d / 4) evaluated as written (NoContraction): the device otherwise drops the exact quarter scaling of
// denominators past 2^126, whose subnormal reciprocal then flushes the CPU's normal SiLU tail to zero.
public float quarter_quotient(float x, float d) {
  let quarter = 0.25;
  return spirv_asm {
    %quarter_x:$$float = OpFMul $x $quarter;
    %quarter_d:$$float = OpFMul $d $quarter;
    %quotient:$$float = OpFDiv %quarter_x %quarter_d;
    OpDecorate %quarter_x NoContraction;
    OpDecorate %quarter_d NoContraction;
    OpDecorate %quotient NoContraction;
    result:$$float = OpCopyObject %quotient;
  };
}

// SiLU with slope alpha, x / (1 + e^(-alpha x)) staged as the CPU activation_silu_alpha, the exponent -alpha x an exact
// FP32 product also for subnormal operands. Where it is at most 2^-26 in magnitude its exponential rounds to 1, so the
// CPU result is x / 2, formed exactly; small x alone does not suffice for large alpha. Past 2^126 the reciprocal a
// division may use is subnormal and flushes, so both operands are scaled by an exact 1/4 there, keeping the CPU's normal
// results down to the exponential's overflow. Where the quotient may be subnormal (exponent fields of x at least 126
// below the denominator's), the numerator is scaled up by 2^24 exactly and the quotient back down with rounding. Without
// CUSTOM alpha is 1 and specialized away: x * 1 is x exactly, and the division left after the halving branch never
// yields a subnormal quotient, since |x| above 2^-26 keeps d below 2 for positive x, and below 2^12 where negative x is
// small enough.
public float activate_silu_alpha<let CUSTOM : bool>(float x, float alpha) {
  let exponent = -(CUSTOM ? exact_product(alpha, x) : x);
  if ((asuint(exponent) & 0x7fffffffu) <= 0x32800000u) {
    return exact_scale_down(x, 1);
  }
  let d = 1.0 + exp(exponent);
  [branch]
  if (CUSTOM && ((asuint(x) >> 23) & 0xffu) + 126 <= ((asuint(d) >> 23) & 0xffu)) {
    let numerator = exact_scale_up_24(x);
    return exact_scale_down(d > asfloat(0x7e800000u) ? quarter_quotient(numerator, d) : numerator / d, 24);
  }
  return d > asfloat(0x7e800000u) ? quarter_quotient(x, d) : x / d;
}

// SiLU is the alpha 1 case. Up to 2^-26 in magnitude both GELUs round their tanh or erf term to its exact constant, so
// the CPU result is x / 2, formed exactly. Softplus returns its input above 20 and otherwise adds 1 in FP32 before the
// log, like the CPU, not log1p.
public float activate(float x, ActivationType activation) {
  let halves = activation == ActivationType.GELUApprox || activation == ActivationType.GELUExact;
  if (halves && (asuint(x) & 0x7fffffffu) <= 0x32800000u) {
    return exact_scale_down(x, 1);
  }
  switch (activation) {
  case ActivationType.SILU:
    return activate_silu_alpha<false>(x, 1.0);
  case ActivationType.GELUApprox:
    return 0.5 * x * (1.0 + accurate_tanh(GELU_K1 * (x + GELU_K0 * x * x * x)));
  case ActivationType.GELUExact:
    return 0.5 * x * (1.0 + erf_precise(x * 0.70710677));
  case ActivationType.SOFTPLUS:
    return x > 20.0 ? x : staged_log(1.0 + exp(x));
  default:
    return x;
  }
}
