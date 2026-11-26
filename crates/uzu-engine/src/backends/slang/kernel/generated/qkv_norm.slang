enum QKVL2Op : uint32_t {
  None = 0,
  RmsNorm = 1,
  Clip = 2,
};

struct QKVNormParams {
  uint op[3];
  uint full_layer[3];
  float epsilon[3];
  float scale_offset[3];
  float clip_threshold[3];
};