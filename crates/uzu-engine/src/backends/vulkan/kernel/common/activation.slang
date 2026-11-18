import generated.activation_type;

float accurate_tanh(float x) {
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

public float apply_soft_cap(float x, float cap) {
  return cap * tanh(x / cap);
}

float erf_precise(float a) {
  let t = abs(a);
  let s = a * a;
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

public float sigmoid(float x) {
  return 1.0 / (1.0 + exp(-x));
}

float activate_silu_alpha(float x, float alpha) {
  let y = 1.0 / (1.0 + exp(-abs(x) * alpha));
  return x < 0.0 ? (1.0 - y) * x : y * x;
}

// alpha scales the SiLU slope and is ignored by every other activation.
public float activate(float x, ActivationType activation, float alpha = 1.0) {
  switch (activation) {
  case ActivationType.SILU:
    return activate_silu_alpha(x, alpha);
  case ActivationType.GELUApprox:
    return 0.5 * x * (1.0 + accurate_tanh(GELU_K1 * (x + GELU_K0 * x * x * x)));
  case ActivationType.GELUExact:
    return 0.5 * x * (1.0 + erf_precise(x * 0.70710677));
  case ActivationType.SOFTPLUS:
    return x > 20.0 ? x : log(1.0 + exp(x));
  case ActivationType.SIGMOID: {
    let y = 1.0 / (1.0 + exp(-abs(x)));
    return x < 0.0 ? 1.0 - y : y;
  }
  default:
    return x;
  }
}
