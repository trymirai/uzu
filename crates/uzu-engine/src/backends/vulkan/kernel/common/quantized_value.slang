// The 4-bit field `nibble` of packed bytes, the low half of a byte first.
public uint read_nibble(Ptr<uint8_t, Access.Read> bytes, uint nibble) {
  return (uint(bytes[nibble / 2]) >> (4 * (nibble % 2))) & 15u;
}

// scale code plus the bias for ScaleBias, otherwise minus scale times the offset (zero point or midpoint).
public float quantized_value(float scale, int code, bool scale_bias, float correction) {
  return scale * float(code) + (scale_bias ? correction : -scale * correction);
}
