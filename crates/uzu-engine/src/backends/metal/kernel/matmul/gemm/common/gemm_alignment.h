#pragma once

#include "../../../common/integral_constant.h"
#include "../../common/defines.h"

using namespace metal;

namespace uzu {
namespace gemm {

template <typename F>
METAL_FUNC void dispatch_gemm_alignment(uint mask, F f) {
  switch (mask & 0b111u) {
  case 0u:
    f(integral_constant<uint, 0u>{});
    break;
  case 1u:
    f(integral_constant<uint, 1u>{});
    break;
  case 2u:
    f(integral_constant<uint, 2u>{});
    break;
  case 3u:
    f(integral_constant<uint, 3u>{});
    break;
  case 4u:
    f(integral_constant<uint, 4u>{});
    break;
  case 5u:
    f(integral_constant<uint, 5u>{});
    break;
  case 6u:
    f(integral_constant<uint, 6u>{});
    break;
  case 7u:
    f(integral_constant<uint, 7u>{});
    break;
  }
}

} // namespace gemm
} // namespace uzu
