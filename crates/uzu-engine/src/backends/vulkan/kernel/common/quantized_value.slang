public uint read_nibble(Ptr<uint8_t, Access.Read> bytes, uint nibble) {
  return (uint(bytes[nibble / 2]) >> (4 * (nibble % 2))) & 15u;
}

public float quantized_value(float scale, int code, bool scale_bias, float correction) {
  return scale * float(code) + (scale_bias ? correction : -scale * correction);
}
