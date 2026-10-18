import definitions;
import storage;

[[Variants("T", "float, half, BFloat16")]]
[[Public]] [[Kernel]] func TensorAddScale<T : ITensorElement>(
    [[Optional("!in_place")]] StructuredBuffer<uint> input,
    StructuredBuffer<uint> bias,
    RWStructuredBuffer<Atomic<uint>> output,
    uint num_cols,
    uint length,
    float scale,
    [[Axis("length", "32")]] uint position,
    [[Specialize]] bool in_place,
) {
  uint3 gid = uint3(position, 0, 0);

  let i = gid.x;
  if (i >= length) {
    return;
  }
  tensorStore<T>(
      output,
      i,
      ((in_place ? tensorLoad<T>(output, i) : tensorLoad<T>(input, i)) + tensorLoad<T>(bias, i % num_cols)) * scale
  );
}
