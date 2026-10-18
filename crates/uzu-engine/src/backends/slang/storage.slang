// Floating-point tensor storage over portable WGSL 32-bit words.
module storage;

public interface ITensorElement {
  static uint byteSize();
  static float decode(uint bits);
  static uint encode(float value);
}

public extension float : ITensorElement {
  public static uint byteSize() { return 4; }
  public static float decode(uint bits) { return asfloat(bits); }
  public static uint encode(float value) { return asuint(value); }
}

public extension half : ITensorElement {
  public static uint byteSize() { return 2; }
  public static float decode(uint bits) { return f16tof32(bits); }
  public static uint encode(float value) { return f32tof16(value); }
}

public struct BFloat16 : ITensorElement {
  public static uint byteSize() { return 2; }
  public static float decode(uint bits) { return asfloat(bits << 16); }
  public static uint encode(float value) {
    uint bits = asuint(value);
    // Preserve NaNs, including payloads confined to the discarded bits.
    if ((bits & 0x7fffffff) > 0x7f800000)
      return (bits >> 16) | 0x40;
    return (bits + 0x7fff + ((bits >> 16) & 1)) >> 16;
  }
}

public float tensorRound<T : ITensorElement>(float value) {
  return T.decode(T.encode(value));
}

public float tensorLoad<T : ITensorElement>(StructuredBuffer<uint> buffer, uint index) {
  uint byteOffset = index * T.byteSize();
  uint bits = buffer[byteOffset / 4] >> ((byteOffset % 4) * 8);
  return T.decode(bits);
}

public float tensorLoad<T : ITensorElement>(RWStructuredBuffer<Atomic<uint>> buffer, uint index) {
  uint byteOffset = index * T.byteSize();
  uint bits = buffer[byteOffset / 4].load() >> ((byteOffset % 4) * 8);
  return T.decode(bits);
}

public void tensorStore<T : ITensorElement>(RWStructuredBuffer<Atomic<uint>> buffer, uint index, float value) {
  uint bits = T.encode(value);
  if (T.byteSize() == 4) {
    buffer[index].store(bits);
  } else {
    uint wordIndex = index / 2;
    uint shift = (index % 2) * 16;
    uint mask = 0xffff << shift;
    uint old = buffer[wordIndex].load();
    while (true) {
      uint next = (old & ~mask) | ((bits & 0xffff) << shift);
      uint observed = buffer[wordIndex].compareExchange(old, next);
      if (observed == old)
        break;
      old = observed;
    }
  }
}

public uint byteLoad(StructuredBuffer<uint> buffer, uint index) { return (buffer[index / 4] >> ((index % 4) * 8)) & 0xff; }

public T choose<T>(T a, T b, bool condition) { return condition ? b : a; }

public float activate(float v, uint kind, float alpha) {
  switch (kind) {
  case 0:
    return v / (1.0 + exp(-alpha * v));
  case 1:
    return 0.5 * v * (1.0 + tanh(0.7978846 * (v + 0.044715 * v * v * v)));
  case 2: {
    float x = v * 0.7071067811865476;
    float a = abs(x);
    float t = 1.0 / (1.0 + 0.3275911 * a);
    float e = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t *
                        exp(-a * a);
    return 0.5 * v * (1.0 + sign(x) * e);
  }
  case 4:
    return max(v, 0.0) + log(1.0 + exp(-abs(v)));
  default:
    return v;
  }
}
