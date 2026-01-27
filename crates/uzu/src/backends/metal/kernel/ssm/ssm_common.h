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