import definitions;
import common.matmul;
import common.quantized_value;
import generated.quantization_method;

// A8 operands: INT8 activation codes with an FP32 scale per 128 values against weight codes centered to signed bytes,
// whose dot within a weight group is exact. W4 activations are stored GroupedByNibble (K order 0, 2, 4, 6, 1, 3, 5, 7
// per 8 codes): a weight word's low nibbles then its high ones, so words pair without shuffles.

public static const uint INT8_SCALE_GROUP = 128;

// The integer dot of the 4 signed bytes of `a` and `b`, never saturating.
public int dot_i8x4(uint a, uint b) {
  return spirv_asm {
    OpCapability DotProduct;
    OpCapability DotProductInput4x8BitPacked;
    result:$$int = OpSDot $a $b PackedVectorFormat4x8Bit;
  };
}

// Where only the unsigned packed dot is fast, a run's dot of signed activations a and codes q centered on z is
// dot(a + 128, q) - 128 sum(q) - z sum(a), the same exact integer: a + 128 is a's byte with its sign bit flipped,
// q's bytes are the unsigned codes, and every term is below 2^22 for 32 values. sum(q) is shared by a row's inputs,
// sum(a) by an input's rows.

// `acc` plus the integer dot of the 4 unsigned bytes of `a` and `b`; callers' sums never saturate.
public uint udot_u8x4(uint a, uint b, uint acc) {
  return spirv_asm {
    OpCapability DotProduct;
    OpCapability DotProductInput4x8BitPacked;
    result:$$uint = OpUDotAccSat $a $b $acc PackedVectorFormat4x8Bit;
  };
}

// A run's 8 words of signed activation codes as unsigned bytes code plus 128, returning their sum.
public uint biased_activations(inout uint words[8]) {
  var sum = 0u;
  [unroll]
  for (uint word = 0; word < 8; word++) {
    words[word] ^= 0x80808080u;
    sum = udot_u8x4(words[word], 0x01010101u, sum);
  }
  return sum;
}

// A run's words of weight codes, 4 of U4 or 8 of U8, as 8 words of unsigned codes in its activations' word order,
// returning their sum.
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
    // Stored signed U8 codes are code minus 128.
    words[word] ^= !u4 && signed_codes ? 0x80808080u : 0u;
    sum = udot_u8x4(words[word], 0x01010101u, sum);
  }
  return sum;
}

// The 4 codes of `codes`, each byte 0 to 15, minus `zero_point` (0 to 15) as signed bytes.
public uint centered_nibbles(uint codes, uint zero_point) {
  return (codes + (128u - zero_point) * 0x01010101u) ^ 0x80808080u;
}

// A word of 8 U4 codes, stored signed codes flipped back, as two words of signed bytes code minus `zero_point`: the low
// nibbles (K offsets 0, 2, 4, 6) and the high ones (1, 3, 5, 7).
public uint2 centered_u4(uint word, uint zero_point, bool signed_codes) {
  let codes = word ^ (signed_codes ? 0x88888888u : 0u);
  return uint2(centered_nibbles(codes & 0x0f0f0f0fu, zero_point), centered_nibbles((codes >> 4) & 0x0f0f0f0fu, zero_point));
}

// A word of 4 U8 symmetric codes as signed bytes code minus 128: stored signed codes already are.
public uint centered_u8(uint word, bool signed_codes) {
  return signed_codes ? word : word ^ 0x80808080u;
}

// A group's integer dot folded into `sum`; ScaleBias adds the bias times the activations' scaled group sum.
public float int8_fold(float sum, float activation_scale, WeightGroup weights, int dot, bool scale_bias, int group_sum) {
  var value = fma(activation_scale * weights.scale, float(dot), sum);
  if (scale_bias) {
    value = fma(weights.bias, activation_scale * float(group_sum), value);
  }
  return value;
}
