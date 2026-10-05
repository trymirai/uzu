// MetalPerformancePrimitives declarations, just enough for uzu's MXU headers to parse. The MXU path
// (M5 neural accelerators) is never compiled for AMDGPU: build/amdgpu drops `use_mxu = true` variants.
#pragma once

namespace metal {
struct execution_simdgroup {};
} // namespace metal

namespace mpp {
namespace tensor_ops {

struct matmul2d_descriptor {
  enum class mode { multiply, multiply_accumulate };

  int m;
  int n;
  int k;
  bool transpose_left;
  bool transpose_right;
  bool relaxed_precision;
  mode matmul_mode;

  constexpr matmul2d_descriptor(
      int m,
      int n,
      int k,
      bool transpose_left = false,
      bool transpose_right = false,
      bool relaxed_precision = false,
      mode matmul_mode = mode::multiply
  )
      : m(m), n(n), k(k), transpose_left(transpose_left), transpose_right(transpose_right),
        relaxed_precision(relaxed_precision), matmul_mode(matmul_mode) {}

  // C++ for OpenCL (C++17) has no class-type template parameters; matmul2d takes the descriptor
  // encoded as an int instead.
  constexpr operator int() const {
    return m | (n << 8) | (k << 16) | (int(transpose_left) << 24) | (int(transpose_right) << 25) |
           (int(relaxed_precision) << 26) | (int(matmul_mode) << 27);
  }
};

// Declared, never defined: instantiating it means an MXU variant reached the AMDGPU build.
template <int Descriptor, typename Scope>
struct matmul2d;

} // namespace tensor_ops
} // namespace mpp
