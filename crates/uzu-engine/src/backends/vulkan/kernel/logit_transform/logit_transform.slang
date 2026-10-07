import definitions;

// Like the Metal kernel, the scaled logit is rounded to T before the soft cap. Below |x| = 0.1 the builtin tanh loses
// relative accuracy, so it is replaced there by its degree-5 Taylor polynomial x (1 - x²/3 + 2x⁴/15), whose omitted
// 17x⁷/315 term is under 5.5e-8 relative; the factored form keeps the sign of zero.
[[Kernel]] [[Public]] func LogitTransform<T: IModelFloat>(
  Ptr<T, Access.ReadWrite> logits,
  uint length,
  float scale,
  float soft_cap,
  [[Axis("length", "256")]] uint position,
  [[Specialize]] bool has_soft_cap,
) {
  T value = T.fromFloat(logits[position].toFloat() * scale);
  if (has_soft_cap) {
    let x = value.toFloat() / soft_cap;
    float y;
    if (abs(x) < 0.1) {
      let s = x * x;
      y = x * fma(s, fma(s, 2.0 / 15.0, -1.0 / 3.0), 1.0);
    } else {
      y = tanh(x);
    }
    value = T.fromFloat(soft_cap * y);
  }
  logits[position] = value;
}
