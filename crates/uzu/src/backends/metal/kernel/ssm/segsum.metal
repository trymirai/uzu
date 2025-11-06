#include <metal_stdlib>
#include "../definitions.metal"

using namespace metal;

// Compute inclusive cumsum along the last dimension for a 2D view (outer_size, length),
// assuming contiguous row-major layout.
template <typename T>
kernel void cumsum_1d_kernel(
    device const T* x [[ buffer(0) ]],
    device T* s [[ buffer(1) ]],
    constant const size_t& length [[ buffer(2) ]],
    constant const size_t& outer_size [[ buffer(3) ]],
    uint row_idx [[ thread_position_in_grid ]]
) {
    if (row_idx >= outer_size) return;
    size_t base = row_idx * length;
    T acc = T(0);
    for (size_t t = 0; t < length; ++t) {
        acc = acc + x[base + t];
        s[base + t] = acc;
    }
}

// Build lower-triangular segmented sum matrix Y from cumsum S.
// For each row (outer index) and target column j, fill Y[i, j] = S[j-1] - S[i-1] for i < j; else 0.
template <typename T>
kernel void segsum_from_cumsum_kernel(
    device const T* s [[ buffer(0) ]],      // (outer_size, length)
    device T* y [[ buffer(1) ]],            // (outer_size, length, length)
    constant const size_t& length [[ buffer(2) ]],
    constant const size_t& outer_size [[ buffer(3) ]],
    uint2 tid [[ thread_position_in_grid ]]
) {
    size_t row_idx = tid.x;
    size_t j = tid.y;
    if (row_idx >= outer_size || j >= length) return;

    size_t s_base = row_idx * length;
    size_t y_row_base = row_idx * (length * length);

    if (j == 0) {
        // Column 0 is all zeros (strictly lower triangle).
        return;
    }

    T s_jm1 = s[s_base + (j - 1)];
    for (size_t i = 0; i < j; ++i) {
        T s_im1 = (i == 0) ? T(0) : s[s_base + (i - 1)];
        T val = s_jm1 - s_im1;
        size_t y_idx = y_row_base + i * length + j;
        y[y_idx] = val;
    }
}

#define instantiate_cumsum_1d_kernel(type_name, type)                              \
  template [[host_name("cumsum_1d_kernel_" #type_name)]]                         \
  kernel void cumsum_1d_kernel<type>(                                              \
    device const type* x [[buffer(0)]],                                            \
    device type* s [[buffer(1)]],                                                  \
    constant const size_t& length [[buffer(2)]],                                   \
    constant const size_t& outer_size [[buffer(3)]],                                \
    uint row_idx [[thread_position_in_grid]]);                                     \

#define instantiate_segsum_from_cumsum_kernel(type_name, type)                     \
  template [[host_name("segsum_from_cumsum_kernel_" #type_name)]]                \
  kernel void segsum_from_cumsum_kernel<type>(                                     \
    device const type* s [[buffer(0)]],                                            \
    device type* y [[buffer(1)]],                                                  \
    constant const size_t& length [[buffer(2)]],                                   \
    constant const size_t& outer_size [[buffer(3)]],                                \
    uint2 tid [[thread_position_in_grid]]);

instantiate_cumsum_1d_kernel(float, float);
instantiate_cumsum_1d_kernel(bfloat, bfloat);
instantiate_cumsum_1d_kernel(half, half);

instantiate_segsum_from_cumsum_kernel(float, float);
instantiate_segsum_from_cumsum_kernel(bfloat, bfloat);
instantiate_segsum_from_cumsum_kernel(half, half);



