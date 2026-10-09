import common.activation;

// Quantized values as the CPU decodes them, shared by the embedding lookup and the quantized and A8 Matmul readers.

// The 4-bit field `nibble` of packed bytes, the low half of a byte first.
public uint read_nibble(Ptr<uint8_t, Access.Read> bytes, uint nibble) {
  return (uint(bytes[nibble / 2]) >> (4 * (nibble % 2))) & 15u;
}

// scale code as exact_product rounds it, for an int code (C == int rejects floats): its FP32 value, exact for the
// callers' -128 to 255, is zero or at least 1 in magnitude, so a normal scale's product is never subnormal and keeps a
// zero's sign, and an infinite or NaN scale gives exact_product's class; zero and subnormal scales take exact_product.
public float code_product<C>(float scale, C code) where C == int {
  [branch]
  if ((asuint(scale) & 0x7f800000u) != 0) {
    return scale * float(code);
  }
  return exact_product(scale, float(code));
}

// scale code + correction in the CPU's staging, each step rounded with subnormals kept and never fused: the code
// product, then the bias for ScaleBias, otherwise minus the product of the scale and the offset (zero point or
// midpoint). The CPU adds the negated offset product: IEEE defines x - y as x + (-y), signed zeros, infinities and NaN
// included, and -(s z) is (-s) z under round to nearest even, while the device may flush a negated subnormal, so the
// sign is flipped on the bits.
public float quantized_value<C>(float scale, C code, bool scale_bias, float correction) where C == int {
  let product = code_product(scale, code);
  if (scale_bias) {
    return ordered_sum(product, correction);
  }
  return ordered_sum(product, asfloat(asuint(exact_product(scale, correction)) ^ 0x80000000u));
}
