enum GemmTiling : uint32_t {
  Tile8x32x32_Simdgroups1x1 = 0,
  Tile64x32x32_Simdgroups2x2 = 1,
  Tile64x64x16_Simdgroups2x2 = 2,
  Tile64x64x32_Simdgroups2x2 = 3,
  Tile32x32x32_Simdgroups2x2 = 4,
  Tile16x64x64_Simdgroups1x4x2 = 5,
  Tile32x64x64_Simdgroups1x4x2 = 6,
  Tile64x128x32_Simdgroups2x4x1 = 7,
  Tile128x128x32_Simdgroups2x4x1 = 8,
};