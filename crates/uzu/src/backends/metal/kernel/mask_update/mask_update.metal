#include <metal_stdlib>
#include "../definitions.metal"

DSL_STRUCT MaskUpdateParams {
  int unmask_col;
  int mask_col;
};

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(MaskUpdate) (
    device T* mask,
    constant MaskUpdateParams* params
) {
  if (params->unmask_col >= 0) {
    mask[params->unmask_col] = T(0);
  }
  if (params->mask_col >= 0) {
    mask[params->mask_col] = -numeric_limits<T>::infinity();
  }
}
