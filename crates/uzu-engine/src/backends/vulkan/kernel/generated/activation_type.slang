static const float GELU_K0 = 0.044715;

static const float GELU_K1 = 0.7978846;

enum ActivationType : uint32_t {
  SILU = 0,
  GELUApprox = 1,
  GELUExact = 2,
  IDENTITY = 3,
  SOFTPLUS = 4,
};