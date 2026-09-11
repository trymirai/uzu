#pragma once

#include <metal_stdlib>

#include "integer_formats.h"

using namespace metal;

namespace uzu {
namespace matmul {

template <typename Format>
struct DeviceTensorOperand {
  using Element = typename Format::TensorElement;
  using Pointer = typename Format::DevicePointer;

  Pointer base;
  int row_stride_bytes;
};

} // namespace matmul
} // namespace uzu
