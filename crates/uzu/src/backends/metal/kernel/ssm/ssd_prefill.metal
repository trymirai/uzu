#include <metal_stdlib>
#include "../definitions.metal"

using namespace metal;

struct SILU {
  template <typename T>
  T operator()(T x) {
    float xf = float(x);
    float y = 1.0f / (1.0f + fast::exp(-fabs(xf)));
    float out = (xf < 0.0f) ? (1.0f - y) * xf : y * xf;
    return static_cast<T>(out);
  }
};

template <typename T>
kernel void ssd_prefill_kernel(
    device const T* x [[ buffer(0) ]],      // (suffix, h, dh)
    device const T* dt [[ buffer(1) ]],     // (suffix, h)
    device const T* decay [[ buffer(2) ]],  // (suffix, h)
    device const T* B [[ buffer(3) ]],      // (suffix, g, n)
    device const T* C [[ buffer(4) ]],      // (suffix, g, n)
    device const T* D [[ buffer(5) ]],      // (h)
    device const T* z [[ buffer(6) ]],      // (suffix, h, dh)
    device T* state [[ buffer(7) ]],        // (h, dh, n)
    device T* y [[ buffer(8) ]],            // (suffix, h, dh)
    constant const size_t& suffix_len [[ buffer(9) ]],
    constant const int& group_size [[ buffer(10) ]],
    constant const int& state_size [[ buffer(11) ]],
    constant const size_t* x_strides [[ buffer(12) ]],
    constant const size_t* dt_strides [[ buffer(13) ]],
    constant const size_t* cb_strides [[ buffer(14) ]],
    constant const size_t* state_strides [[ buffer(15) ]],
    uint3 tid [[ thread_position_in_grid ]],
    uint3 grid_dim [[ threads_per_grid ]]
) {
    const uint h_idx = tid.x;
    const uint dh_idx = tid.y;
    if (h_idx >= grid_dim.x || dh_idx >= grid_dim.y) {
        return;
    }

    const uint group_idx = h_idx / group_size;
    device T* state_row =
        state + h_idx * state_strides[0] + dh_idx * state_strides[1];

    for (size_t token = 0; token < suffix_len; ++token) {
        const size_t x_idx = token * x_strides[0]
            + size_t(h_idx) * x_strides[1]
            + size_t(dh_idx) * x_strides[2];
        const size_t dt_idx =
            token * dt_strides[0] + size_t(h_idx) * dt_strides[1];
        const size_t cb_base =
            token * cb_strides[0] + group_idx * cb_strides[1];

        const T this_x = x[x_idx];
        const T this_dt = dt[dt_idx];
        const T this_decay = decay[dt_idx];
        const T this_D = D[h_idx];
        const T this_z = SILU{}(z[x_idx]);
        const float dt_f = fmax(float(this_dt), 1e-6f);
        const float mix = 1.0f - float(this_decay);
        const float mix_over_dt = mix / dt_f;
        const T scaled_input = static_cast<T>(mix_over_dt * float(this_x));

        T acc = T(0);
        int s = 0;
        const int vec_bound = (state_size / 4) * 4;
        for (; s < vec_bound; s += 4) {
            const size_t state_idx = s * state_strides[2];
            const size_t cb_idx = cb_base + s * cb_strides[2];
            auto prev_state = *reinterpret_cast<device vec<T, 4>*>(
                state_row + state_idx);
            auto b_vec = *reinterpret_cast<device const vec<T, 4>*>(
                B + cb_idx);
            auto c_vec = *reinterpret_cast<device const vec<T, 4>*>(
                C + cb_idx);
            vec<T, 4> new_state =
                prev_state * this_decay + b_vec * scaled_input;
            *reinterpret_cast<device vec<T, 4>*>(state_row + state_idx) =
                new_state;
            vec<T, 4> prod = new_state * c_vec;
            acc += prod.x + prod.y + prod.z + prod.w;
        }
        for (; s < state_size; ++s) {
            const size_t state_idx = s * state_strides[2];
            const T prev_state = state_row[state_idx];
            const size_t cb_idx = cb_base + s * cb_strides[2];
            const T new_state =
                prev_state * this_decay + B[cb_idx] * scaled_input;
            state_row[state_idx] = new_state;
            acc += new_state * C[cb_idx];
        }

        acc += this_D * this_x;
        acc *= this_z;
        y[x_idx] = acc;
    }
}

#define instantiate_ssd_prefill_kernel(type_name, type)      \
  template [[host_name("ssd_prefill_kernel_" #type_name)]]   \
  kernel void ssd_prefill_kernel<type>(                      \
    device const type* x [[ buffer(0) ]],                    \
    device const type* dt [[ buffer(1) ]],                   \
    device const type* decay [[ buffer(2) ]],                \
    device const type* B [[ buffer(3) ]],                    \
    device const type* C [[ buffer(4) ]],                    \
    device const type* D [[ buffer(5) ]],                    \
    device const type* z [[ buffer(6) ]],                    \
    device type* state [[ buffer(7) ]],                      \
    device type* y [[ buffer(8) ]],                          \
    constant const size_t& suffix_len [[ buffer(9) ]],       \
    constant const int& group_size [[ buffer(10) ]],         \
    constant const int& state_size [[ buffer(11) ]],         \
    constant const size_t* x_strides [[ buffer(12) ]],       \
    constant const size_t* dt_strides [[ buffer(13) ]],      \
    constant const size_t* cb_strides [[ buffer(14) ]],      \
    constant const size_t* state_strides [[ buffer(15) ]],   \
    uint3 tid [[ thread_position_in_grid ]],                 \
    uint3 grid_dim [[ threads_per_grid ]]                    \
  );

instantiate_ssd_prefill_kernel(float, float);
instantiate_ssd_prefill_kernel(bfloat, bfloat);
instantiate_ssd_prefill_kernel(half, half);

#undef instantiate_ssd_prefill_kernel
