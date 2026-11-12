import definitions;
import common.workgroup_reduction;

// Weaver ranks use a descending score followed by ascending tie keys. The last key is the physical index.
public struct WeaverRank : IReduction {
  public typealias Value = uint4;
  public static uint4 combine(uint4 a, uint4 b) {
    return b.x > a.x || (b.x == a.x && (b.y < a.y ||
        (b.y == a.y && (b.z < a.z || (b.z == a.z && b.w < a.w))))) ? b : a;
  }
  public static uint4 subgroup(uint4 value) {
    uint4 best;
    best.x = WaveActiveMax(value.x);
    best.y = WaveActiveMin(value.x == best.x ? value.y : 0xffffffffu);
    best.z = WaveActiveMin(all(value.xy == best.xy) ? value.z : 0xffffffffu);
    best.w = WaveActiveMin(all(value.xyz == best.xyz) ? value.w : 0xffffffffu);
    return best;
  }
}

// The monotone key of f32::total_cmp, including signed zeros and signed NaN payloads.
public uint float_order_key(uint bits) {
  return (bits & 0x80000000u) != 0 ? ~bits : bits ^ 0x80000000u;
}

public uint float_order_bits(uint key) {
  return (key & 0x80000000u) != 0 ? key ^ 0x80000000u : ~key;
}
