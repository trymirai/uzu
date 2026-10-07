import definitions;

// Pure BF16 storage conversions with FP32 bits carried as uint, so no floating-point arithmetic is involved.

[[Kernel]] [[Test]] func TestBf16Load(
  Ptr<bf16, Access.Read> input,
  Ptr<uint, Access.ReadWrite> output,
  uint size,
  [[Axis("size", "64")]] uint idx,
) {
  output[idx] = asuint(input[idx].toFloat());
}

[[Kernel]] [[Test]] func TestBf16Store(
  Ptr<uint, Access.Read> input,
  Ptr<bf16, Access.ReadWrite> output,
  uint size,
  [[Axis("size", "64")]] uint idx,
) {
  output[idx] = bf16.fromFloat(asfloat(input[idx]));
}
