// MSL simdgroup_matrix (8x8) on wave32.
//
// Each lane holds two consecutive elements of a row, in Apple's layout (uzu's SimdgroupFragmentOps
// and Fragment::row_reduce depend on it): lane bits b4 b3 b2 b1 b0 hold row 4*b4 + 2*b2 + b1 and
// columns 2*(2*b3 + b0) .. +1. The multiply gathers the operand row and columns it needs with
// ds_bpermute and accumulates in float. Correct, not fast: WMMA fragments replace it on hot paths.
#pragma once

namespace metal {

template <typename T, int Rows, int Cols>
struct simdgroup_matrix {
  static_assert(Rows == 8 && Cols == 8, "only 8x8 simdgroup matrices are supported");
  vec<T, 2> elements_;

  METAL_FUNC vec<T, 2>& thread_elements() { return elements_; }
  METAL_FUNC const vec<T, 2>& thread_elements() const { return elements_; }
};

// Lane that holds elements (row, 2 * pair) and (row, 2 * pair + 1).
METAL_FUNC uint __simdgroup_matrix_lane(uint row, uint pair) {
  return ((row >> 2) << 4) | ((pair >> 1) << 3) | (((row >> 1) & 1u) << 2) | ((row & 1u) << 1) | (pair & 1u);
}

template <typename TD, typename TA, typename TB, typename TC>
METAL_FUNC void simdgroup_multiply_accumulate(
    simdgroup_matrix<TD, 8, 8>& d,
    const simdgroup_matrix<TA, 8, 8>& a,
    const simdgroup_matrix<TB, 8, 8>& b,
    const simdgroup_matrix<TC, 8, 8>& c
) {
  const uint lane = __simd_lane_id();
  const uint row = ((lane >> 4) << 2) | (((lane >> 2) & 1u) << 1) | ((lane >> 1) & 1u);
  const uint pair = (((lane >> 3) & 1u) << 1) | (lane & 1u);
  float acc0 = float(c.elements_.x);
  float acc1 = float(c.elements_.y);
#pragma unroll
  for (uint k_pair = 0; k_pair < 4; ++k_pair) {
    const vec<TA, 2> a_row = __simd_permute_impl<vec<TA, 2>>::permute(a.elements_, __simdgroup_matrix_lane(row, k_pair));
    const vec<TB, 2> b_even =
        __simd_permute_impl<vec<TB, 2>>::permute(b.elements_, __simdgroup_matrix_lane(2 * k_pair, pair));
    const vec<TB, 2> b_odd =
        __simd_permute_impl<vec<TB, 2>>::permute(b.elements_, __simdgroup_matrix_lane(2 * k_pair + 1, pair));
    acc0 = __builtin_fmaf(float(a_row.x), float(b_even.x), acc0);
    acc0 = __builtin_fmaf(float(a_row.y), float(b_odd.x), acc0);
    acc1 = __builtin_fmaf(float(a_row.x), float(b_even.y), acc1);
    acc1 = __builtin_fmaf(float(a_row.y), float(b_odd.y), acc1);
  }
  d.elements_.x = TD(acc0);
  d.elements_.y = TD(acc1);
}

template <typename TD, typename TA, typename TB>
METAL_FUNC void simdgroup_multiply(
    simdgroup_matrix<TD, 8, 8>& d,
    const simdgroup_matrix<TA, 8, 8>& a,
    const simdgroup_matrix<TB, 8, 8>& b
) {
  simdgroup_matrix<TD, 8, 8> zero;
  zero.elements_.x = TD(0);
  zero.elements_.y = TD(0);
  simdgroup_multiply_accumulate(d, a, b, zero);
}

} // namespace metal
