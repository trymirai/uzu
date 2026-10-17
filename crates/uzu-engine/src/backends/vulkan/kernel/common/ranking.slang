import definitions;

// Weaver ranks use a descending score followed by ascending tie keys. The last key is the physical index.
public uint4 weaver_rank_better(uint4 a, uint4 b) {
  return b.x > a.x || (b.x == a.x && (b.y < a.y ||
      (b.y == a.y && (b.z < a.z || (b.z == a.z && b.w < a.w))))) ? b : a;
}

public uint4 weaver_rank_wave(uint4 value) {
  uint4 best;
  best.x = WaveActiveMax(value.x);
  best.y = WaveActiveMin(value.x == best.x ? value.y : 0xffffffffu);
  best.z = WaveActiveMin(all(value.xy == best.xy) ? value.z : 0xffffffffu);
  best.w = WaveActiveMin(all(value.xyz == best.xyz) ? value.w : 0xffffffffu);
  return best;
}

// The FP32 bits of a stored 16-bit score as the CPU widens it: NaNs quieted with their payload, whatever the device's
// widening keeps of them.
public uint score_bits<T : IModelFloat16>(T value) {
  // BF16's toFloat widens NaNs so already (7 fraction bits identify it among 16-bit floats).
  let bits = asuint(value.toFloat());
  if (T.FRACTION_BITS == 7 || (bits & 0x7fffffffu) <= 0x7f800000u) {
    return bits;
  }
  let stored = value.storageBits();
  let payload = (stored & ((1u << T.FRACTION_BITS) - 1u)) << (23 - T.FRACTION_BITS);
  return ((stored & 0x8000u) << 16) | 0x7fc00000u | payload;
}

// The monotone key of f32::total_cmp, including signed zeros and signed NaN payloads.
public uint float_order_key(uint bits) {
  return (bits & 0x80000000u) != 0 ? ~bits : bits ^ 0x80000000u;
}

public uint float_order_bits(uint key) {
  return (key & 0x80000000u) != 0 ? key ^ 0x80000000u : ~key;
}
