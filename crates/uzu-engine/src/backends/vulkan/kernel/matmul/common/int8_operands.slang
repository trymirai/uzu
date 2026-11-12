import definitions;
import matmul.common.matmul;
import common.quantized_value;
import generated.quantization_method;

public uint biased_activations(inout uint words[8]) {
  var sum = 0u;
  [unroll]
  for (uint word = 0; word < 8; word++) {
    words[word] ^= 0x80808080u;
    sum = udot_u8x4(words[word], 0x01010101u, sum);
  }
  return sum;
}

public uint unsigned_codes(inout uint words[8], bool u4, bool signed_codes) {
  if (u4) {
    [unroll]
    for (int word = 3; word >= 0; word--) {
      let codes = words[word] ^ (signed_codes ? 0x88888888u : 0u);
      words[2 * word] = codes & 0x0f0f0f0fu;
      words[2 * word + 1] = (codes >> 4) & 0x0f0f0f0fu;
    }
  }
  var sum = 0u;
  [unroll]
  for (uint word = 0; word < 8; word++) {
    words[word] ^= !u4 && signed_codes ? 0x80808080u : 0u;
    sum = udot_u8x4(words[word], 0x01010101u, sum);
  }
  return sum;
}

public uint centered_nibbles(uint codes, uint zero_point) {
  return (codes + (128u - zero_point) * 0x01010101u) ^ 0x80808080u;
}

public uint2 centered_u4(uint word, uint zero_point, bool signed_codes) {
  let codes = word ^ (signed_codes ? 0x88888888u : 0u);
  return uint2(centered_nibbles(codes & 0x0f0f0f0fu, zero_point), centered_nibbles((codes >> 4) & 0x0f0f0f0fu, zero_point));
}

public uint centered_u8(uint word, bool signed_codes) {
  return signed_codes ? word : word ^ 0x80808080u;
}

public float int8_fold(float sum, float activation_scale, WeightGroup weights, int dot, bool scale_bias, int group_sum) {
  var value = fma(activation_scale * weights.scale, float(dot), sum);
  if (scale_bias) {
    value = fma(weights.bias, activation_scale * float(group_sum), value);
  }
  return value;
}
