#include <metal_stdlib>
#include "../definitions.metal"

template <typename T>
VARIANTS(T, float, half, bfloat)
KERNEL(MaskUpdate) (
    device T* mask,
    constant int& unmask_col,
    constant int& mask_col
) {
  if (unmask_col >= 0) {
    mask[unmask_col] = T(0);
  }
  if (mask_col >= 0) {
    mask[mask_col] = -numeric_limits<T>::infinity();
  }
}
