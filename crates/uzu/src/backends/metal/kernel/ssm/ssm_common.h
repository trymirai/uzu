#define ACTIVATION_IDENTITY 0
#define ACTIVATION_SILU 1
#define ACTIVATION_GELU 2

template <typename T>
inline T apply_silu(T x) {
  float xf = float(x);
  float y = 1.0f / (1.0f + fast::exp(-fabs(xf)));
  float out = (xf < 0.0f) ? (1.0f - y) * xf : y * xf;
  return static_cast<T>(out);
}

template <typename T>
inline T softplus(T x) {
  float xf = float(x);
  if (xf > 20.0f) {
    return x;
  }
  return static_cast<T>(log(1.0f + fast::exp(xf)));
}

template <typename T>
inline T apply_gelu(T x) {
  float xf = float(x);
  return static_cast<T>(
      0.5f * xf *
      (1.0f + fast::tanh(0.797885f * (xf + 0.044715f * xf * xf * xf)))
  );
}

template <typename T>
inline T apply_activation_fn(T x, int activation_type) {
  if (activation_type == ACTIVATION_SILU) {
    return apply_silu(x);
  } else if (activation_type == ACTIVATION_GELU) {
    return apply_gelu(x);
  } else {
    return x; // Identity
  }
}