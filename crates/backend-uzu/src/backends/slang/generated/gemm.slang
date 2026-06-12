enum GemmBPrologueKind : uint32_t {
  FullPrecision = 0,
  ScaleBiasDequant = 1,
  ScaleZeroPointDequant = 2,
  ScaleSymmetricDequant = 3,
};

enum GemmTiling : uint32_t {
  Tile8x32x32_Simdgroups1x1 = 0,
  Tile64x32x32_Simdgroups2x2 = 1,
  Tile64x64x16_Simdgroups2x2 = 2,
  Tile64x64x32_Simdgroups2x2 = 3,
  Tile32x32x32_Simdgroups2x2 = 4,
  Tile32x64x256_Simdgroups2x2 = 5,
  Tile64x32x256_Simdgroups4x1 = 6,
  Tile64x64x256_Simdgroups2x2 = 7,
  Tile128x128x256_Simdgroups4x4 = 8,
};