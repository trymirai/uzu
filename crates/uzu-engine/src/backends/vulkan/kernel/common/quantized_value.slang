import common.activation;

// Quantized weight values as the CPU decodes them, shared by the embedding lookup and the quantized Matmul reader.

// The 4-bit field `nibble` of packed bytes, the low half of a byte first.
public uint read_nibble(Ptr<uint8_t, Access.Read> bytes, uint nibble) {
  return (uint(bytes[nibble / 2]) >> (4 * (nibble % 2))) & 15u;
}

// scale code + correction in the CPU's staging, each step rounded with subnormals kept and never fused: the code
// product, then the bias for ScaleBias, otherwise minus the product of the scale and the offset (zero point or
// midpoint). The CPU adds the negated offset product: IEEE defines x - y as x + (-y), signed zeros, infinities and NaN
// included, and -(s z) is (-s) z under round to nearest even, while the device may flush a negated subnormal, so the
// sign is flipped on the bits.
public float quantized_value(float scale, float code, bool scale_bias, float correction) {
  let product = exact_product(scale, code);
  if (scale_bias) {
    return ordered_sum(product, correction);
  }
  return ordered_sum(product, asfloat(asuint(exact_product(scale, correction)) ^ 0x80000000u));
}
