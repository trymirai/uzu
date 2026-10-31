import definitions;
import common.activation;

// The scaled logit rounds to T before the soft cap.
[[Variants("T", "float, bf16")]]
[[Kernel]] [[Public]] func LogitTransform<T: IStorageFloat>(
  Ptr<T, Access.ReadWrite> logits,
  uint length,
  float scale,
  float soft_cap,
  [[Axis("length", "256")]] uint position,
  [[Specialize]] bool has_soft_cap,
) {
  T value = T.fromFloat(logits[position].toFloat() * scale);
  if (has_soft_cap) {
    value = T.fromFloat(apply_soft_cap(value.toFloat(), soft_cap));
  }
  logits[position] = value;
}
