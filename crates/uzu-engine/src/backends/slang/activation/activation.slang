import definitions;
import storage;
import generated.activation_type;

[[Variants("T", "float, half, BFloat16")]]
[[Public]] [[Kernel]] func Activation<T : ITensorElement>(
    [[Optional("!in_place")]] StructuredBuffer<uint> input,
    RWStructuredBuffer<Atomic<uint>> output,
    uint n,
    ActivationType act_type,
    [[Specialize]] bool in_place,
    [[Axis("n", "256")]] uint tid,
) {
  uint3 gid = uint3(tid, 0, 0);

  let i = gid.x;
  if (i >= n) {
    return;
  }
  tensorStore<T>(
      output,
      i,
      activate((in_place ? tensorLoad<T>(output, i) : tensorLoad<T>(input, i)), uint(act_type), 1.0)
  );
}
