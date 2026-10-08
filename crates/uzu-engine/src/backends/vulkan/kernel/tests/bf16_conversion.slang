import definitions;
import common.activation;

// Pure BF16 and F16 storage conversions with FP32 bits carried as uint, so no floating-point arithmetic is involved.

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

[[Kernel]] [[Test]] func TestF16Load(
  Ptr<half, Access.Read> input,
  Ptr<uint, Access.ReadWrite> output,
  uint size,
  [[Axis("size", "64")]] uint idx,
) {
  output[idx] = asuint(input[idx].toFloat());
}

[[Kernel]] [[Test]] func TestF16Store(
  Ptr<uint, Access.Read> input,
  Ptr<half, Access.ReadWrite> output,
  uint size,
  [[Axis("size", "64")]] uint idx,
) {
  output[idx] = half.fromFloat(asfloat(input[idx]));
}

// Normalization's residual staging as generic kernels reach the IStorageFloat conversions (concrete half methods may
// lower differently): the FP32 sum of an F16 pair narrowed to F16 in registers, widened, times an FP32 scalar and
// narrowed again for the store; then, as its RMS statistics, the stored result's exact square, narrowed after it.
T staged_residual<T: IStorageFloat>(T a, T b, float scalar) {
  let sum = T.fromFloat(a.toFloat() + b.toFloat());
  return T.fromFloat(sum.toFloat() * scalar);
}

[[Kernel]] [[Test]] func TestF16Staged(
  Ptr<half, Access.Read> input,
  Ptr<half, Access.ReadWrite> output,
  uint size,
  float scalar,
  [[Axis("size", "64")]] uint idx,
) {
  let result = staged_residual(input[2 * idx], input[2 * idx + 1], scalar);
  output[idx] = result;
  let x = result.toFloat();
  output[size + idx] = half.fromFloat(exact_product(x, x));
}
