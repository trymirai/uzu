#pragma once

#include "mpp_cooperative_matmul.h"
#include "steel/gemm/params.h"
#include "steel/gemm/transforms.h"

using namespace metal;

namespace uzu {
namespace matmul {

template <typename F>
void dispatch_bool(bool v, F f) {
  if (v) {
    f(metal::bool_constant<true>{});
  } else {
    f(metal::bool_constant<false>{});
  }
}

} // namespace matmul
} // namespace uzu
