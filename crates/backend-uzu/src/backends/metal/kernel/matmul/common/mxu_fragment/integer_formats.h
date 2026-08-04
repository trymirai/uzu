#pragma once

#include <metal_stdlib>

#include "../../../common/defines.h"

using namespace metal;

namespace uzu {
namespace matmul {

enum class Signedness { Signed, Unsigned };

template <int Bits, Signedness SIGNEDNESS>
struct IntegerFormat;

template <Signedness SIGNEDNESS>
struct IntegerFormat<4, SIGNEDNESS> {
  using StorageElement = uchar;
  using UnpackedElement = metal::conditional_t<SIGNEDNESS == Signedness::Signed, int8_t, uint8_t>;
  using TensorElement =
      metal::conditional_t<SIGNEDNESS == Signedness::Signed, metal::int4b_format, metal::uint4b_format>;
  using DevicePointer = const device uchar*;
  using MutableDevicePointer = device uchar*;
  UZU_CONST int elements_per_byte = 2;
};

template <Signedness SIGNEDNESS>
struct IntegerFormat<8, SIGNEDNESS> {
  using StorageElement = metal::conditional_t<SIGNEDNESS == Signedness::Signed, int8_t, uint8_t>;
  using UnpackedElement = StorageElement;
  using TensorElement = StorageElement;
  using DevicePointer = const device StorageElement*;
  using MutableDevicePointer = device StorageElement*;
  UZU_CONST int elements_per_byte = 1;
};

} // namespace matmul
} // namespace uzu
