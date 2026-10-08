import definitions;

// Two kernels with a specialization of the same name and distinct effects, next to a plain one, in one linked module.

[[Kernel]] [[Test]] func TestSpecializationA(
  Ptr<uint, Access.Read> input,
  Ptr<uint, Access.ReadWrite> output,
  uint size,
  [[Axis("size", "64")]] uint idx,
  [[Specialize]] bool flag,
) {
  output[idx] = input[idx] + (flag ? 0x10u : 0x20u);
}

[[Kernel]] [[Test]] func TestSpecializationB(
  Ptr<uint, Access.Read> input,
  Ptr<uint, Access.ReadWrite> output,
  uint size,
  [[Axis("size", "64")]] uint idx,
  [[Specialize]] bool flag,
) {
  output[idx] = input[idx] + (flag ? 0x100u : 0x200u);
}

[[Kernel]] [[Test]] func TestSpecializationPlain(
  Ptr<uint, Access.Read> input,
  Ptr<uint, Access.ReadWrite> output,
  uint size,
  [[Axis("size", "64")]] uint idx,
) {
  output[idx] = input[idx] + 1u;
}
