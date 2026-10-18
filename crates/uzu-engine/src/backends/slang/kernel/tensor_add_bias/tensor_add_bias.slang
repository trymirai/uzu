import definitions;
import storage;

[[Variants("T", "float, half, BFloat16")]]
[[Variants("BiasT", "float, half, BFloat16")]]
[[Public]] [[Kernel]] func TensorAddBias<T : ITensorElement, BiasT : ITensorElement>(
    [[Optional("!in_place")]] StructuredBuffer<uint32_t> input,
    StructuredBuffer<uint32_t> bias,
    RWStructuredBuffer<Atomic<uint32_t>> output,
    uint32_t num_cols,
    uint32_t length,
    [[Axis("length", "32")]] uint32_t position,
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
      (in_place ? tensorLoad<T>(output, i) : tensorLoad<T>(input, i)) + tensorLoad<BiasT>(bias, i % num_cols)
  );
}
