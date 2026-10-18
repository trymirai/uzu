import definitions;
import storage;

groupshared float sums[256];

// Portable reduction from experiments/demo-w's matmul_decode kernel.
[[Variants("WeightT", "float, half, BFloat16")]]
[[Variants("InputT", "float, half, BFloat16")]]
[[Variants("OutputT", "float, half, BFloat16")]] [[Kernel]] func MatmulSimple<WeightT : ITensorElement, InputT : ITensorElement, OutputT : ITensorElement>(
    StructuredBuffer<uint> a,
    StructuredBuffer<uint> b,
    [[Optional("quantized")]] StructuredBuffer<uint> scales,
    [[Optional("quantized && correction == 1")]] StructuredBuffer<uint> quant_biases,
    [[Optional("quantized && correction == 2")]] StructuredBuffer<uint> zero_points,
    [[Optional("has_bias")]] StructuredBuffer<uint> bias,
    [[Optional("gather")]] StructuredBuffer<uint> gather_indices,
    RWStructuredBuffer<Atomic<uint>> d,
    uint m,
    uint n,
    uint k,
    uint a_offset,
    uint b_stride,
    uint row_stride,
    uint group_size,
    uint scale_output_stride,
    uint scale_group_stride,
    uint zp_output_stride,
    uint zp_group_stride,
    float output_scale,
    float soft_cap,
    [[Specialize]] bool quantized,
    [[Specialize]] uint bits,
    [[Specialize]] bool signed_codes,
    [[Specialize]] uint correction,
    [[Specialize]] bool transpose,
    [[Specialize]] bool has_bias,
    [[Specialize]] bool has_soft_cap,
    [[Specialize]] bool gather,
    [[Groups("n.min(65535)")]] uint column_group,
    [[Groups("m")]] uint row,
    [[Groups("n.div_ceil(65535)")]] uint column_tile,
    [[Threads("256")]] uint lane,
) {
  uint column = column_group + column_tile * 65535;
  if (column >= n)
    return;
  uint weightColumn = gather ? gather_indices[row * n + column] : column;
  float sum = 0;
  for (uint inner = lane; inner < k; inner += 256) {
    float av = tensorLoad<InputT>(a, a_offset + row * k + inner);
    float bv;
    if (quantized) {
      uint packed = 32 / bits;
      uint index = weightColumn * k + inner;
      uint code = (b[index / packed] >> ((index % packed) * bits)) & ((1u << bits) - 1);
      if (signed_codes)
        code ^= 1u << (bits - 1);
      uint group = inner / group_size;
      uint metadata = weightColumn * scale_output_stride + group * scale_group_stride;
      float scale = tensorLoad<WeightT>(scales, metadata);
      float offset = -scale * float(1u << (bits - 1));
      if (correction == 1)
        offset = tensorLoad<WeightT>(quant_biases, metadata);
      if (correction == 2) {
        uint pointIndex = weightColumn * zp_output_stride + group * zp_group_stride;
        uint point = (zero_points[pointIndex / packed] >> ((pointIndex % packed) * bits)) & ((1u << bits) - 1);
        offset = -scale * float(point);
      }
      bv = scale * float(code) + offset;
    } else {
      bv = tensorLoad<WeightT>(b, transpose ? weightColumn * b_stride + inner : inner * b_stride + weightColumn);
    }
    sum += av * bv;
  }
  sums[lane] = sum;
  GroupMemoryBarrierWithGroupSync();
  for (uint stride = 128; stride > 0; stride /= 2) {
    if (lane < stride)
      sums[lane] += sums[lane + stride];
    GroupMemoryBarrierWithGroupSync();
  }
  if (lane == 0) {
    float value = sums[0] * output_scale;
    if (has_bias)
      value += tensorLoad<WeightT>(bias, column);
    if (has_soft_cap)
      value = soft_cap * tanh(value / soft_cap);
    tensorStore<OutputT>(d, row * row_stride + column, value);
  }
}
