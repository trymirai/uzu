import definitions;
import storage;

[[Variants("T", "float, half, BFloat16")]]
[[Public]] [[Kernel]] func SigmoidGate<T : ITensorElement>(
    StructuredBuffer<uint> gate,
    RWStructuredBuffer<Atomic<uint>> output,
    uint gate_dim,
    uint batch_dim,
    uint gate_row_stride,
    [[Axis("gate_dim", "256")]] uint gate_idx,
    [[Axis("batch_dim", "1")]] uint batch_idx,
) {
  uint3 gid = uint3(batch_idx * gate_dim + gate_idx, 0, 0);

  let i = gid.x;
  if (i >= gate_dim * batch_dim) {
    return;
  }
  let j = (i / gate_dim) * gate_row_stride + i % gate_dim;
  tensorStore<T>(output, i, tensorLoad<T>(output, i) / (1.0 + exp(-tensorLoad<T>(gate, j))));
}
