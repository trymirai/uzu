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

// x / 2 rounded to nearest even through the FP32 bits, exact also for subnormal inputs and results, which FP32
// arithmetic may flush to zero.
float exact_half(float x) {
  let bits = asuint(x);
  let magnitude = bits & 0x7fffffffu;
  if (magnitude >= 0x01000000u) {
    return asfloat(bits - 0x00800000u);
  }
  return asfloat((bits & 0x80000000u) | ((magnitude >> 1) + (magnitude & (magnitude >> 1) & 1u)));
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

// Up to 2^-26 in magnitude SiLU and both GELUs round their exponential, tanh or erf term to its exact constant, so the
// CPU result is x / 2, formed exactly. Softplus returns its input above 20 and otherwise adds 1 in FP32 before the log,
// like the CPU, not log1p.
public float activate(float x, ActivationType activation) {
  let halves = activation == ActivationType.SILU || activation == ActivationType.GELUApprox
    || activation == ActivationType.GELUExact;
  if (halves && (asuint(x) & 0x7fffffffu) <= 0x32800000u) {
    return exact_half(x);
  }
  switch (activation) {
  case ActivationType.SILU:
    // Past 2^126 the reciprocal a division may use is subnormal and flushes, so both operands are scaled by an exact
    // 1/4 there, keeping the CPU's normal results down to the exponential's overflow.
    let d = 1.0 + exp(-x);
    return d > asfloat(0x7e800000u) ? x * 0.25 / (d * 0.25) : x / d;
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
