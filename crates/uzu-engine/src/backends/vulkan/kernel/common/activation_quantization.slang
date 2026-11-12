import common.workgroup_reduction;

public uint int8_code_index(uint index, bool grouped_by_weight_nibble) {
  let offset = index % 8;
  return grouped_by_weight_nibble ? index - offset + offset % 2 * 4 + offset / 2 : index;
}

public static const uint ACTIVATION_QUANTIZATION_TILE = 128;

public float activation_quantization_scale(uint bits) {
  return bits != 0u && bits < 0x7f800000u ? asfloat(bits) / 127.0 : 1.0;
}

public int activation_quantization_code(float value, float scale) {
  return int(clamp(round(value / scale), -127.0, 127.0));
}

public void quantize_activation(
  float value,
  uint row,
  uint column,
  uint count,
  Ptr<int8_t, Access.ReadWrite> codes,
  Ptr<float, Access.ReadWrite> scales,
  Ptr<int, Access.ReadWrite> group_sums,
  uint scale_group,
  uint sum_group,
  bool emit_group_sums,
  bool grouped_by_weight_nibble,
) {
  let active = column < count;
  let magnitude = asuint(value) & 0x7fffffffu;
  let largest = segment_reduce<Max<uint>, ACTIVATION_QUANTIZATION_TILE>(magnitude > 0x7f800000u ? 0u : magnitude,
                                                                         scale_group);
  let scale = activation_quantization_scale(largest);
  let code = activation_quantization_code(value, scale);
  if (active) {
    codes[row * count + int8_code_index(column, grouped_by_weight_nibble)] = int8_t(code);
    if (column % scale_group == 0) {
      scales[(row * count + column) / scale_group] = scale;
    }
  }
  if (emit_group_sums) {
    let sum = segment_reduce<Sum<int>, ACTIVATION_QUANTIZATION_TILE>(code, sum_group);
    if (active && column % sum_group == 0) {
      group_sums[(row * count + column) / sum_group] = sum;
    }
  }
}
