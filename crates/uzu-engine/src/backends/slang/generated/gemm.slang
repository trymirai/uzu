enum GemmBPrologueKind : uint32_t {
  FullPrecision = 0,
  ScaleBiasDequant = 1,
  ScaleZeroPointDequant = 2,
  ScaleSymmetricDequant = 3,
};

struct GemmDTransform {
  uint raw_value;
  static const uint SCALE = 1 << 0;
  static const uint BIAS = 1 << 1;
  static const uint RHT = 1 << 2;
  static const uint SOFT_CAP = 1 << 3;
  bool contains(uint flag) { return (raw_value & flag) != 0; }
  uint bits() { return raw_value; }
};