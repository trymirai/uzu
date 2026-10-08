import definitions;
import common.activation;

// Like the Metal kernel, the scaled logit is rounded to T before the soft cap.
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
    value = T.fromFloat(soft_cap * accurate_tanh(value.toFloat() / soft_cap));
  }
  logits[position] = value;
}
