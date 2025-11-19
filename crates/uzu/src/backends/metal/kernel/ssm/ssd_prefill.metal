#include <metal_stdlib>
#include <metal_simdgroup>
#include "../definitions.metal"

using namespace metal;

constant ushort SSM_PREFILL_CHUNK = 64;
constant ushort SSM_PREFILL_MAX_STATE = 64;

struct SILU {
  template <typename T>
  T operator()(T x) const {
    float xf = float(x);
    float y = 1.0f / (1.0f + fast::exp(-fabs(xf)));
    float out = (xf < 0.0f) ? (1.0f - y) * xf : y * xf;
    return static_cast<T>(out);
  }
};

template <typename T>
inline T softplus(T x) {
    float xf = float(x);
    if (xf > 20.0f) {
        return x;
    }
    return static_cast<T>(log(1.0f + fast::exp(xf)));
}

template <typename T>
kernel void ssd_prefill_kernel(
    device const T* x [[ buffer(0) ]],      // (suffix, h, dh)
    device const T* dt_raw [[ buffer(1) ]], // (suffix, h) - raw dt values
    device const T* B [[ buffer(2) ]],      // (suffix, g, n)
    device const T* C [[ buffer(3) ]],      // (suffix, g, n)
    device const T* D [[ buffer(4) ]],      // (h)
    device const T* z [[ buffer(5) ]],      // (suffix, h, dh)
    device T* state [[ buffer(6) ]],        // (h, dh, n)
    device T* y [[ buffer(7) ]],            // (suffix, h, dh)
    constant const size_t& suffix_len [[ buffer(8) ]],
    constant const int& group_size [[ buffer(9) ]],
    constant const int& state_size [[ buffer(10) ]],
    constant const size_t* x_strides [[ buffer(11) ]],
    constant const size_t* dt_strides [[ buffer(12) ]],
    constant const size_t* cb_strides [[ buffer(13) ]],
    constant const size_t* state_strides [[ buffer(14) ]],
    constant const uint& num_heads [[ buffer(15) ]],
    constant const uint& head_dim [[ buffer(16) ]],
    uint3 tg_pos [[ threadgroup_position_in_grid ]],
    ushort lane [[ thread_index_in_threadgroup ]]
) {
    const uint simd_width = 32;
    const int state_dim = state_size;
    if (state_dim <= 0 || state_dim > int(2 * simd_width)) {
        return;
    }

    const uint total_pairs = num_heads * head_dim;
    const uint pair_idx = tg_pos.x;
    if (pair_idx >= total_pairs) {
        return;
    }

    const uint lane_idx = uint(lane);
    const uint h_idx = pair_idx / head_dim;
    const uint dh_idx = pair_idx % head_dim;
    const uint safe_group = uint(max(group_size, 1));
    const uint group_idx = h_idx / safe_group;

    const size_t x_token_stride = x_strides[0];
    const size_t x_head_stride = x_strides[1];
    const size_t x_dim_stride = x_strides[2];
    const size_t dt_token_stride = dt_strides[0];
    const size_t dt_head_stride = dt_strides[1];
    const size_t cb_token_stride = cb_strides[0];
    const size_t cb_group_stride = cb_strides[1];
    const size_t cb_state_stride = cb_strides[2];
    const size_t state_head_stride = state_strides[0];
    const size_t state_dim_stride = state_strides[1];
    const size_t state_inner_stride = state_strides[2];

    const size_t x_base = size_t(h_idx) * x_head_stride
        + size_t(dh_idx) * x_dim_stride;
    const size_t dt_base = size_t(h_idx) * dt_head_stride;
    const size_t state_base =
        size_t(h_idx) * state_head_stride + size_t(dh_idx) * state_dim_stride;
    const size_t cb_group_base = size_t(group_idx) * cb_group_stride;
    const float d_scalar = float(D[h_idx]);

    const uint idx0 = lane_idx;
    const uint idx1 = lane_idx + simd_width;
    const bool has0 = idx0 < uint(state_dim);
    const bool has1 = idx1 < uint(state_dim);

    float state0 = 0.0f;
    float state1 = 0.0f;
    size_t state_idx0 = 0;
    size_t state_idx1 = 0;
    if (has0) {
        state_idx0 = state_base + size_t(idx0) * state_inner_stride;
        state0 = float(state[state_idx0]);
    }
    if (has1) {
        state_idx1 = state_base + size_t(idx1) * state_inner_stride;
        state1 = float(state[state_idx1]);
    }

    size_t cb_idx0 = 0;
    size_t cb_idx1 = 0;
    if (has0) {
        cb_idx0 = cb_group_base + size_t(idx0) * cb_state_stride;
    }
    if (has1) {
        cb_idx1 = cb_group_base + size_t(idx1) * cb_state_stride;
    }

    for (size_t token = 0; token < suffix_len; ++token) {
        const size_t x_idx = token * x_token_stride + x_base;
        const size_t dt_idx = token * dt_token_stride + dt_base;

        const float x_val = float(x[x_idx]);
        const float dt_raw_val = float(dt_raw[dt_idx]);
        const float dt_val = float(softplus(dt_raw_val));
        const float decay_val = fast::exp(-dt_val);
        const float gate = float(SILU{}(z[x_idx]));
        const float skip = d_scalar * x_val;
        const float dt_safe = fmax(dt_val, 1e-6f);
        const float normalized_x = x_val / dt_safe;
        const float dt_scaled_input = normalized_x * dt_val;

        float contrib = 0.0f;
        if (has0) {
            const float b0 = float(B[cb_idx0]);
            const float c0 = float(C[cb_idx0]);
            const float new_state0 =
                decay_val * state0 + dt_scaled_input * b0;
            state0 = new_state0;
            contrib += new_state0 * c0;
            cb_idx0 += cb_token_stride;
        }
        if (has1) {
            const float b1 = float(B[cb_idx1]);
            const float c1 = float(C[cb_idx1]);
            const float new_state1 =
                decay_val * state1 + dt_scaled_input * b1;
            state1 = new_state1;
            contrib += new_state1 * c1;
            cb_idx1 += cb_token_stride;
        }

        float dot = simd_sum(contrib);
        if (lane_idx == 0) {
            y[x_idx] = static_cast<T>((skip + dot) * gate);
        }
    }

    if (has0) {
        state[state_idx0] = static_cast<T>(state0);
    }
    if (has1) {
        state[state_idx1] = static_cast<T>(state1);
    }
}

#define instantiate_ssd_prefill_kernel(type_name, type)      \
  template [[host_name("ssd_prefill_kernel_" #type_name)]]   \
  kernel void ssd_prefill_kernel<type>(                      \
    device const type* x [[ buffer(0) ]],                    \
    device const type* dt_raw [[ buffer(1) ]],               \
    device const type* B [[ buffer(2) ]],                    \
    device const type* C [[ buffer(3) ]],                    \
    device const type* D [[ buffer(4) ]],                    \
    device const type* z [[ buffer(5) ]],                    \
    device type* state [[ buffer(6) ]],                      \
    device type* y [[ buffer(7) ]],                          \
    constant const size_t& suffix_len [[ buffer(8) ]],       \
    constant const int& group_size [[ buffer(9) ]],          \
    constant const int& state_size [[ buffer(10) ]],         \
    constant const size_t* x_strides [[ buffer(11) ]],       \
    constant const size_t* dt_strides [[ buffer(12) ]],      \
    constant const size_t* cb_strides [[ buffer(13) ]],      \
    constant const size_t* state_strides [[ buffer(14) ]],   \
    constant const uint& num_heads [[ buffer(15) ]],         \
    constant const uint& head_dim [[ buffer(16) ]],          \
    uint3 tg_pos [[ threadgroup_position_in_grid ]],         \
    ushort lane [[ thread_index_in_threadgroup ]]            \
  );

instantiate_ssd_prefill_kernel(float, float);
instantiate_ssd_prefill_kernel(bfloat, bfloat);
instantiate_ssd_prefill_kernel(half, half);

#undef instantiate_ssd_prefill_kernel

template <typename T>
kernel void ssd_prefill_kernel_sequential(
    device const T* x [[ buffer(0) ]],      // (suffix, h, dh)
    device const T* dt_raw [[ buffer(1) ]], // (suffix, h) - raw dt values
    device const T* B [[ buffer(2) ]],      // (suffix, g, n)
    device const T* C [[ buffer(3) ]],      // (suffix, g, n)
    device const T* D [[ buffer(4) ]],      // (h)
    device const T* z [[ buffer(5) ]],      // (suffix, h, dh)
    device T* state [[ buffer(6) ]],        // (h, dh, n)
    device T* y [[ buffer(7) ]],            // (suffix, h, dh)
    constant const size_t& suffix_len [[ buffer(8) ]],
    constant const int& group_size [[ buffer(9) ]],
    constant const int& state_size [[ buffer(10) ]],
    constant const size_t* x_strides [[ buffer(11) ]],
    constant const size_t* dt_strides [[ buffer(12) ]],
    constant const size_t* cb_strides [[ buffer(13) ]],
    constant const size_t* state_strides [[ buffer(14) ]],
    uint3 tid [[ thread_position_in_grid ]],
    uint3 grid_dim [[ threads_per_grid ]]
) {
    const uint h_idx = tid.x;
    const uint dh_idx = tid.y;
    if (h_idx >= grid_dim.x || dh_idx >= grid_dim.y) {
        return;
    }

    const uint safe_group = uint(max(group_size, 1));
    const uint group_idx = h_idx / safe_group;
    device T* state_row =
        state + size_t(h_idx) * state_strides[0] + size_t(dh_idx) * state_strides[1];

    for (size_t token = 0; token < suffix_len; ++token) {
        const size_t x_idx = token * x_strides[0]
            + size_t(h_idx) * x_strides[1]
            + size_t(dh_idx) * x_strides[2];
        const size_t dt_idx =
            token * dt_strides[0] + size_t(h_idx) * dt_strides[1];
        const size_t cb_base =
            token * cb_strides[0] + group_idx * cb_strides[1];

        const T this_x = x[x_idx];
        const T dt_raw_val = dt_raw[dt_idx];
        const T this_dt = softplus(dt_raw_val);
        const T this_decay = static_cast<T>(fast::exp(-float(this_dt)));
        const T this_D = D[h_idx];
        const T this_z = SILU{}(z[x_idx]);
        const float dt_f = fmax(float(this_dt), 1e-6f);
        const float normalized_x = float(this_x) / dt_f;
        const T dt_scaled_input = static_cast<T>(normalized_x) * this_dt;

        T acc = T(0);
        int s = 0;
        const int vec_bound = (state_size / 4) * 4;
        for (; s < vec_bound; s += 4) {
            const size_t state_idx = s * state_strides[2];
            const size_t cb_idx = cb_base + s * cb_strides[2];
            auto prev_state = *reinterpret_cast<device vec<T, 4>*>(state_row + state_idx);
            auto b_vec = *reinterpret_cast<device const vec<T, 4>*>(B + cb_idx);
            auto c_vec = *reinterpret_cast<device const vec<T, 4>*>(C + cb_idx);
            vec<T, 4> new_state = prev_state * this_decay + b_vec * dt_scaled_input;
            *reinterpret_cast<device vec<T, 4>*>(state_row + state_idx) = new_state;
            vec<T, 4> prod = new_state * c_vec;
            acc += prod.x + prod.y + prod.z + prod.w;
        }
        for (; s < state_size; ++s) {
            const size_t state_idx = s * state_strides[2];
            const T prev_state = state_row[state_idx];
            const size_t cb_idx = cb_base + s * cb_strides[2];
            const T new_state = prev_state * this_decay + B[cb_idx] * dt_scaled_input;
            state_row[state_idx] = new_state;
            acc += new_state * C[cb_idx];
        }

        acc += this_D * this_x;
        acc *= this_z;
        y[x_idx] = acc;
    }
}

#define instantiate_ssd_prefill_kernel_sequential(type_name, type)    \
  template [[host_name("ssd_prefill_kernel_sequential_" #type_name)]] \
  kernel void ssd_prefill_kernel_sequential<type>(                    \
    device const type* x [[ buffer(0) ]],                    \
    device const type* dt_raw [[ buffer(1) ]],               \
    device const type* B [[ buffer(2) ]],                    \
    device const type* C [[ buffer(3) ]],                    \
    device const type* D [[ buffer(4) ]],                    \
    device const type* z [[ buffer(5) ]],                    \
    device type* state [[ buffer(6) ]],                      \
    device type* y [[ buffer(7) ]],                          \
    constant const size_t& suffix_len [[ buffer(8) ]],       \
    constant const int& group_size [[ buffer(9) ]],          \
    constant const int& state_size [[ buffer(10) ]],         \
    constant const size_t* x_strides [[ buffer(11) ]],       \
    constant const size_t* dt_strides [[ buffer(12) ]],      \
    constant const size_t* cb_strides [[ buffer(13) ]],      \
    constant const size_t* state_strides [[ buffer(14) ]],   \
    uint3 tid [[ thread_position_in_grid ]],                 \
    uint3 grid_dim [[ threads_per_grid ]]                    \
  );

instantiate_ssd_prefill_kernel_sequential(float, float);
instantiate_ssd_prefill_kernel_sequential(bfloat, bfloat);
instantiate_ssd_prefill_kernel_sequential(half, half);

#undef instantiate_ssd_prefill_kernel_sequential
