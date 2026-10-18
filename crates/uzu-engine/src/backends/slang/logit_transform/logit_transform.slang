import definitions;
import storage;

[[Variants("T", "float, half, BFloat16")]]
[[Public]] [[Kernel]] func LogitTransform<T : ITensorElement>(
    RWStructuredBuffer<Atomic<uint>> logits,
    uint length,
    float scale,
    float soft_cap,
    [[Axis("length", "256")]] uint position,
    [[Specialize]] bool has_soft_cap,
) {
  uint3 gid = uint3(position, 0, 0);

  let i = gid.x;
  if (i >= length) {
    return;
  }
  var v = tensorLoad<T>(logits, i) * scale;
  if (has_soft_cap) {
    v = tanh(v / soft_cap) * soft_cap;
  }
  tensorStore<T>(logits, i, v);
}
