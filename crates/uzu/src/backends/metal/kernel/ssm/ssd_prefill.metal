#include <metal_stdlib>
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
    constant const uint& num_heads [[ buffer(16) ]],
    constant const uint& head_dim [[ buffer(17) ]],
    uint3 tg_pos [[ threadgroup_position_in_grid ]],
    ushort lane [[ thread_index_in_threadgroup ]]
) {
    const int state_dim = state_size;
    if (state_dim <= 0 || state_dim > int(SSM_PREFILL_MAX_STATE)) {
        return;
    }

    const uint total_pairs = num_heads * head_dim;
    const uint pair_idx = tg_pos.x;
    if (pair_idx >= total_pairs || lane >= SSM_PREFILL_CHUNK) {
        return;
    }

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

    const size_t x_base = h_idx * x_head_stride + dh_idx * x_dim_stride;
    const size_t dt_base = h_idx * dt_head_stride;
    const size_t cb_group_base = group_idx * cb_group_stride;
    const size_t state_base =
        h_idx * state_head_stride + dh_idx * state_dim_stride;
    const float d_scalar = float(D[h_idx]);

    threadgroup float scalar_decay;
    threadgroup float scalar_dt_scaled_input;
    threadgroup float scalar_gate;
    threadgroup float scalar_skip;
    threadgroup float reduction_shared[SSM_PREFILL_CHUNK];

    const bool lane_active = lane < state_dim;
    size_t state_idx = 0;
    size_t cb_idx = 0;
    float state_val = 0.0f;

    if (lane_active) {
        state_idx = state_base + size_t(lane) * state_inner_stride;
        state_val = float(state[state_idx]);
        cb_idx = cb_group_base + size_t(lane) * cb_state_stride;
    }

    size_t x_idx = x_base;
    size_t dt_idx = dt_base;

    for (size_t token = 0; token < suffix_len; ++token) {
        if (lane == 0) {
            const float x_val = float(x[x_idx]);
            const float dt_val = float(dt[dt_idx]);
            const float dt_safe = fmax(dt_val, 1e-6f);
            const float normalized_x = x_val / dt_safe;
            scalar_dt_scaled_input = normalized_x * dt_val;
            scalar_decay = float(decay[dt_idx]);
            scalar_gate = SILU{}(float(z[x_idx]));
            scalar_skip = d_scalar * x_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float contrib = 0.0f;
        if (lane_active) {
            const float b_coeff = float(B[cb_idx]);
            const float c_coeff = float(C[cb_idx]);
            const float new_state =
                scalar_decay * state_val + scalar_dt_scaled_input * b_coeff;
            state_val = new_state;
            contrib = new_state * c_coeff;
            cb_idx += cb_token_stride;
        }

        float dot = threadgroup_raking_reduce_sum<SSM_PREFILL_CHUNK>(
            contrib, reduction_shared, lane);
        if (lane == 0) {
            y[x_idx] = static_cast<T>((scalar_skip + dot) * scalar_gate);
            x_idx += x_token_stride;
            dt_idx += dt_token_stride;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane_active) {
        state[state_idx] = static_cast<T>(state_val);
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
    constant const uint& num_heads [[ buffer(16) ]],         \
    constant const uint& head_dim [[ buffer(17) ]],          \
    uint3 tg_pos [[ threadgroup_position_in_grid ]],         \
    ushort lane [[ thread_index_in_threadgroup ]]            \
  );

instantiate_ssd_prefill_kernel(float, float);
instantiate_ssd_prefill_kernel(bfloat, bfloat);
instantiate_ssd_prefill_kernel(half, half);

#undef instantiate_ssd_prefill_kernel

template <typename T>
kernel void ssd_prefill_chunk_transform_kernel(
    device const T* x [[ buffer(0) ]],
    device const T* dt [[ buffer(1) ]],
    device const T* decay [[ buffer(2) ]],
    device const T* B [[ buffer(3) ]],
    device float* chunk_a_out [[ buffer(4) ]],
    device float* chunk_b_out [[ buffer(5) ]],
    constant const size_t& suffix_len [[ buffer(6) ]],
    constant const int& group_size [[ buffer(7) ]],
    constant const int& state_size [[ buffer(8) ]],
    constant const size_t* x_strides [[ buffer(9) ]],
    constant const size_t* dt_strides [[ buffer(10) ]],
    constant const size_t* cb_strides [[ buffer(11) ]],
    constant const size_t* chunk_a_strides [[ buffer(12) ]],
    constant const size_t* chunk_b_strides [[ buffer(13) ]],
    constant const uint& num_heads [[ buffer(14) ]],
    constant const uint& head_dim [[ buffer(15) ]],
    uint2 tg_pos [[ threadgroup_position_in_grid ]],
    ushort lane [[ thread_index_in_threadgroup ]]
) {
    const int state_dim = state_size;
    if (state_dim <= 0 || state_dim > int(SSM_PREFILL_MAX_STATE)) {
        return;
    }

    const uint pair_idx = tg_pos.x;
    const uint chunk_idx = tg_pos.y;
    const uint total_pairs = num_heads * head_dim;
    if (pair_idx >= total_pairs || lane >= SSM_PREFILL_CHUNK) {
        return;
    }

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

    const size_t x_base = h_idx * x_head_stride + dh_idx * x_dim_stride;
    const size_t dt_base = h_idx * dt_head_stride;
    const size_t cb_group_base = group_idx * cb_group_stride;

    const size_t chunk_start = size_t(chunk_idx) * size_t(SSM_PREFILL_CHUNK);
    if (chunk_start >= suffix_len) {
        return;
    }
    const ushort chunk_len = ushort(
        min(size_t(SSM_PREFILL_CHUNK), suffix_len - chunk_start));
    if (chunk_len == 0) {
        return;
    }

    threadgroup float chunk_a_shared[SSM_PREFILL_CHUNK];
    threadgroup float chunk_b_shared[SSM_PREFILL_CHUNK * SSM_PREFILL_MAX_STATE];

    const bool lane_active = lane < chunk_len;
    threadgroup float* b_slot =
        chunk_b_shared + size_t(lane) * SSM_PREFILL_MAX_STATE;
    for (int s = 0; s < state_dim; ++s) {
        b_slot[s] = 0.0f;
    }
    chunk_a_shared[lane] = 1.0f;

    if (lane_active) {
        const size_t token = chunk_start + size_t(lane);
        const size_t x_idx = token * x_token_stride + x_base;
        const size_t dt_idx = token * dt_token_stride + dt_base;
        const size_t cb_idx = token * cb_token_stride + cb_group_base;

        const float x_val = float(x[x_idx]);
        const float dt_val = float(dt[dt_idx]);
        const float decay_val = float(decay[dt_idx]);
        const float dt_safe = fmax(dt_val, 1e-6f);
        const float normalized_x = x_val / dt_safe;
        const float dt_scaled_input = normalized_x * dt_val;

        chunk_a_shared[lane] = decay_val;

        for (int s = 0; s < state_dim; ++s) {
            const size_t offset = cb_idx + size_t(s) * cb_state_stride;
            const float b_coeff = float(B[offset]);
            b_slot[s] = b_coeff * dt_scaled_input;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (ushort stride = 1; stride < chunk_len; stride <<= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane_active && lane >= stride) {
            const float prev_a = chunk_a_shared[lane - stride];
            float curr_a = chunk_a_shared[lane];
            threadgroup const float* prev_b =
                chunk_b_shared + size_t(lane - stride) * SSM_PREFILL_MAX_STATE;
            threadgroup float* curr_b =
                chunk_b_shared + size_t(lane) * SSM_PREFILL_MAX_STATE;
            for (int s = 0; s < state_dim; ++s) {
                curr_b[s] = curr_b[s] + curr_a * prev_b[s];
            }
            chunk_a_shared[lane] = curr_a * prev_a;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane_active && lane == chunk_len - 1) {
        const size_t a_index = size_t(chunk_idx) * chunk_a_strides[0]
            + size_t(h_idx) * chunk_a_strides[1]
            + size_t(dh_idx) * chunk_a_strides[2];
        chunk_a_out[a_index] = chunk_a_shared[lane];

        const size_t b_base = size_t(chunk_idx) * chunk_b_strides[0]
            + size_t(h_idx) * chunk_b_strides[1]
            + size_t(dh_idx) * chunk_b_strides[2];
        threadgroup const float* final_state =
            chunk_b_shared + size_t(lane) * SSM_PREFILL_MAX_STATE;
        for (int s = 0; s < state_dim; ++s) {
            chunk_b_out[b_base + size_t(s) * chunk_b_strides[3]] = final_state[s];
        }
    }
}

template <typename T>
kernel void ssd_prefill_chunk_scan_kernel(
    device const float* chunk_a [[ buffer(0) ]],
    device const float* chunk_b [[ buffer(1) ]],
    device const T* state_in [[ buffer(2) ]],
    device float* prefix_state [[ buffer(3) ]],
    constant const size_t& suffix_len [[ buffer(4) ]],
    constant const int& state_size [[ buffer(5) ]],
    constant const size_t* chunk_a_strides [[ buffer(6) ]],
    constant const size_t* chunk_b_strides [[ buffer(7) ]],
    constant const size_t* state_strides [[ buffer(8) ]],
    constant const size_t* prefix_strides [[ buffer(9) ]],
    constant const uint& num_heads [[ buffer(10) ]],
    constant const uint& head_dim [[ buffer(11) ]],
    uint pair_idx [[ threadgroup_position_in_grid ]],
    ushort lane [[ thread_index_in_threadgroup ]]
) {
    const int state_dim = state_size;
    if (state_dim <= 0 || state_dim > int(SSM_PREFILL_MAX_STATE)) {
        return;
    }
    const uint total_pairs = num_heads * head_dim;
    if (pair_idx >= total_pairs || lane >= SSM_PREFILL_MAX_STATE) {
        return;
    }

    const uint h_idx = pair_idx / head_dim;
    const uint dh_idx = pair_idx % head_dim;
    const size_t state_base = size_t(h_idx) * state_strides[0]
        + size_t(dh_idx) * state_strides[1];

    float state_val = 0.0f;
    if (lane < state_dim) {
        const size_t idx = state_base + size_t(lane) * state_strides[2];
        state_val = float(state_in[idx]);
    }

    const size_t chunk_count =
        (suffix_len + size_t(SSM_PREFILL_CHUNK) - 1) / size_t(SSM_PREFILL_CHUNK);
    for (size_t chunk_idx = 0; chunk_idx < chunk_count; ++chunk_idx) {
        if (lane < state_dim) {
            const size_t prefix_base = size_t(chunk_idx) * prefix_strides[0]
                + size_t(h_idx) * prefix_strides[1]
                + size_t(dh_idx) * prefix_strides[2];
            prefix_state[prefix_base + size_t(lane) * prefix_strides[3]] = state_val;
        }

        const float a_total = chunk_a[
            size_t(chunk_idx) * chunk_a_strides[0]
            + size_t(h_idx) * chunk_a_strides[1]
            + size_t(dh_idx) * chunk_a_strides[2]
        ];
        if (lane < state_dim) {
            const size_t b_base = size_t(chunk_idx) * chunk_b_strides[0]
                + size_t(h_idx) * chunk_b_strides[1]
                + size_t(dh_idx) * chunk_b_strides[2];
            const float b_val = chunk_b[
                b_base + size_t(lane) * chunk_b_strides[3]];
            state_val = a_total * state_val + b_val;
        }
    }
}

template <typename T>
kernel void ssd_prefill_chunk_apply_kernel(
    device const T* x [[ buffer(0) ]],
    device const T* dt [[ buffer(1) ]],
    device const T* decay [[ buffer(2) ]],
    device const T* B [[ buffer(3) ]],
    device const T* C [[ buffer(4) ]],
    device const T* D [[ buffer(5) ]],
    device const T* z [[ buffer(6) ]],
    device T* state [[ buffer(7) ]],
    device T* y [[ buffer(8) ]],
    device const float* prefix_state [[ buffer(9) ]],
    constant const size_t& suffix_len [[ buffer(10) ]],
    constant const int& group_size [[ buffer(11) ]],
    constant const int& state_size [[ buffer(12) ]],
    constant const size_t* x_strides [[ buffer(13) ]],
    constant const size_t* dt_strides [[ buffer(14) ]],
    constant const size_t* cb_strides [[ buffer(15) ]],
    constant const size_t* state_strides [[ buffer(16) ]],
    constant const size_t* prefix_strides [[ buffer(17) ]],
    constant const uint& num_heads [[ buffer(18) ]],
    constant const uint& head_dim [[ buffer(19) ]],
    uint2 tg_pos [[ threadgroup_position_in_grid ]],
    ushort lane [[ thread_index_in_threadgroup ]]
) {
    const int state_dim = state_size;
    if (state_dim <= 0 || state_dim > int(SSM_PREFILL_MAX_STATE)) {
        return;
    }

    const uint pair_idx = tg_pos.x;
    const uint chunk_idx = tg_pos.y;
    const uint total_pairs = num_heads * head_dim;
    if (pair_idx >= total_pairs || lane >= SSM_PREFILL_CHUNK) {
        return;
    }

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

    const size_t x_base = h_idx * x_head_stride + dh_idx * x_dim_stride;
    const size_t dt_base = h_idx * dt_head_stride;
    const size_t cb_group_base = group_idx * cb_group_stride;
    const size_t state_base = h_idx * state_head_stride + dh_idx * state_dim_stride;
    const float d_scalar = float(D[h_idx]);

    const size_t chunk_start = size_t(chunk_idx) * size_t(SSM_PREFILL_CHUNK);
    if (chunk_start >= suffix_len) {
        return;
    }
    const ushort chunk_len = ushort(
        min(size_t(SSM_PREFILL_CHUNK), suffix_len - chunk_start));
    if (chunk_len == 0) {
        return;
    }

    threadgroup float state_block_start[SSM_PREFILL_MAX_STATE];
    threadgroup float chunk_end_state[SSM_PREFILL_MAX_STATE];

    for (int s = lane; s < state_dim; s += SSM_PREFILL_CHUNK) {
        const size_t prefix_base = size_t(chunk_idx) * prefix_strides[0]
            + size_t(h_idx) * prefix_strides[1]
            + size_t(dh_idx) * prefix_strides[2];
        state_block_start[s] =
            prefix_state[prefix_base + size_t(s) * prefix_strides[3]];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const bool lane_active = lane < chunk_len;
    threadgroup float* b_slot =
        chunk_end_state; // reuse for local updates

    threadgroup float chunk_a_shared[SSM_PREFILL_CHUNK];
    threadgroup float chunk_b_shared[SSM_PREFILL_CHUNK * SSM_PREFILL_MAX_STATE];

    if (!lane_active) {
        return;
    }

    threadgroup float* b_local =
        chunk_b_shared + size_t(lane) * SSM_PREFILL_MAX_STATE;
    for (int s = 0; s < state_dim; ++s) {
        b_local[s] = 0.0f;
    }
    chunk_a_shared[lane] = 1.0f;

    const size_t token = chunk_start + size_t(lane);
    const size_t x_idx = token * x_token_stride + x_base;
    const size_t dt_idx = token * dt_token_stride + dt_base;
    const size_t cb_idx = token * cb_token_stride + cb_group_base;

    const float x_val = float(x[x_idx]);
    const float dt_val = float(dt[dt_idx]);
    const float decay_val = float(decay[dt_idx]);
    const float dt_safe = fmax(dt_val, 1e-6f);
    const float normalized_x = x_val / dt_safe;
    const float dt_scaled_input = normalized_x * dt_val;

    chunk_a_shared[lane] = decay_val;

    for (int s = 0; s < state_dim; ++s) {
        const size_t offset = cb_idx + size_t(s) * cb_state_stride;
        const float b_coeff = float(B[offset]);
        b_local[s] = b_coeff * dt_scaled_input;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (ushort stride = 1; stride < chunk_len; stride <<= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane_active && lane >= stride) {
            const float prev_a = chunk_a_shared[lane - stride];
            float curr_a = chunk_a_shared[lane];
            threadgroup const float* prev_b =
                chunk_b_shared + size_t(lane - stride) * SSM_PREFILL_MAX_STATE;
            threadgroup float* curr_b =
                chunk_b_shared + size_t(lane) * SSM_PREFILL_MAX_STATE;
            for (int i = 0; i < state_dim; ++i) {
                curr_b[i] = curr_b[i] + curr_a * prev_b[i];
            }
            chunk_a_shared[lane] = curr_a * prev_a;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane_active) {
        const float prefix_a = chunk_a_shared[lane];
        threadgroup const float* prefix_b =
            chunk_b_shared + size_t(lane) * SSM_PREFILL_MAX_STATE;
        float local_state[SSM_PREFILL_MAX_STATE];
        for (int i = 0; i < state_dim; ++i) {
            local_state[i] = prefix_a * state_block_start[i] + prefix_b[i];
        }
        if (lane == chunk_len - 1) {
            for (int i = 0; i < state_dim; ++i) {
                chunk_end_state[i] = local_state[i];
            }
        }

        float acc = d_scalar * x_val;
        float gate = SILU{}(float(z[x_idx]));

        for (int i = 0; i < state_dim; ++i) {
            const size_t offset = cb_idx + size_t(i) * cb_state_stride;
            float c_coeff = float(C[offset]);
            acc += local_state[i] * c_coeff;
        }
        y[x_idx] = static_cast<T>(acc * gate);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    const size_t chunk_count =
        (suffix_len + size_t(SSM_PREFILL_CHUNK) - 1) / size_t(SSM_PREFILL_CHUNK);
    if (chunk_idx == chunk_count - 1) {
        for (int i = lane; i < state_dim; i += SSM_PREFILL_CHUNK) {
            const size_t idx = state_base + size_t(i) * state_inner_stride;
            state[idx] = static_cast<T>(chunk_end_state[i]);
        }
    }
}

template [[host_name("ssd_prefill_chunk_transform_kernel_float")]]
kernel void ssd_prefill_chunk_transform_kernel<float>(
    device const float* x [[ buffer(0) ]],
    device const float* dt [[ buffer(1) ]],
    device const float* decay [[ buffer(2) ]],
    device const float* B [[ buffer(3) ]],
    device float* chunk_a_out [[ buffer(4) ]],
    device float* chunk_b_out [[ buffer(5) ]],
    constant const size_t& suffix_len [[ buffer(6) ]],
    constant const int& group_size [[ buffer(7) ]],
    constant const int& state_size [[ buffer(8) ]],
    constant const size_t* x_strides [[ buffer(9) ]],
    constant const size_t* dt_strides [[ buffer(10) ]],
    constant const size_t* cb_strides [[ buffer(11) ]],
    constant const size_t* chunk_a_strides [[ buffer(12) ]],
    constant const size_t* chunk_b_strides [[ buffer(13) ]],
    constant const uint& num_heads [[ buffer(14) ]],
    constant const uint& head_dim [[ buffer(15) ]],
    uint2 tg_pos [[ threadgroup_position_in_grid ]],
    ushort lane [[ thread_index_in_threadgroup ]]);
template [[host_name("ssd_prefill_chunk_transform_kernel_bfloat")]]
kernel void ssd_prefill_chunk_transform_kernel<bfloat>(
    device const bfloat* x [[ buffer(0) ]],
    device const bfloat* dt [[ buffer(1) ]],
    device const bfloat* decay [[ buffer(2) ]],
    device const bfloat* B [[ buffer(3) ]],
    device float* chunk_a_out [[ buffer(4) ]],
    device float* chunk_b_out [[ buffer(5) ]],
    constant const size_t& suffix_len [[ buffer(6) ]],
    constant const int& group_size [[ buffer(7) ]],
    constant const int& state_size [[ buffer(8) ]],
    constant const size_t* x_strides [[ buffer(9) ]],
    constant const size_t* dt_strides [[ buffer(10) ]],
    constant const size_t* cb_strides [[ buffer(11) ]],
    constant const size_t* chunk_a_strides [[ buffer(12) ]],
    constant const size_t* chunk_b_strides [[ buffer(13) ]],
    constant const uint& num_heads [[ buffer(14) ]],
    constant const uint& head_dim [[ buffer(15) ]],
    uint2 tg_pos [[ threadgroup_position_in_grid ]],
    ushort lane [[ thread_index_in_threadgroup ]]);
template [[host_name("ssd_prefill_chunk_transform_kernel_half")]]
kernel void ssd_prefill_chunk_transform_kernel<half>(
    device const half* x [[ buffer(0) ]],
    device const half* dt [[ buffer(1) ]],
    device const half* decay [[ buffer(2) ]],
    device const half* B [[ buffer(3) ]],
    device float* chunk_a_out [[ buffer(4) ]],
    device float* chunk_b_out [[ buffer(5) ]],
    constant const size_t& suffix_len [[ buffer(6) ]],
    constant const int& group_size [[ buffer(7) ]],
    constant const int& state_size [[ buffer(8) ]],
    constant const size_t* x_strides [[ buffer(9) ]],
    constant const size_t* dt_strides [[ buffer(10) ]],
    constant const size_t* cb_strides [[ buffer(11) ]],
    constant const size_t* chunk_a_strides [[ buffer(12) ]],
    constant const size_t* chunk_b_strides [[ buffer(13) ]],
    constant const uint& num_heads [[ buffer(14) ]],
    constant const uint& head_dim [[ buffer(15) ]],
    uint2 tg_pos [[ threadgroup_position_in_grid ]],
    ushort lane [[ thread_index_in_threadgroup ]]);

template [[host_name("ssd_prefill_chunk_scan_kernel_float")]]
kernel void ssd_prefill_chunk_scan_kernel<float>(
    device const float* chunk_a [[ buffer(0) ]],
    device const float* chunk_b [[ buffer(1) ]],
    device const float* state_in [[ buffer(2) ]],
    device float* prefix_state [[ buffer(3) ]],
    constant const size_t& suffix_len [[ buffer(4) ]],
    constant const int& state_size [[ buffer(5) ]],
    constant const size_t* chunk_a_strides [[ buffer(6) ]],
    constant const size_t* chunk_b_strides [[ buffer(7) ]],
    constant const size_t* state_strides [[ buffer(8) ]],
    constant const size_t* prefix_strides [[ buffer(9) ]],
    constant const uint& num_heads [[ buffer(10) ]],
    constant const uint& head_dim [[ buffer(11) ]],
    uint pair_idx [[ threadgroup_position_in_grid ]],
    ushort lane [[ thread_index_in_threadgroup ]]);
template [[host_name("ssd_prefill_chunk_scan_kernel_bfloat")]]
kernel void ssd_prefill_chunk_scan_kernel<bfloat>(
    device const float* chunk_a [[ buffer(0) ]],
    device const float* chunk_b [[ buffer(1) ]],
    device const bfloat* state_in [[ buffer(2) ]],
    device float* prefix_state [[ buffer(3) ]],
    constant const size_t& suffix_len [[ buffer(4) ]],
    constant const int& state_size [[ buffer(5) ]],
    constant const size_t* chunk_a_strides [[ buffer(6) ]],
    constant const size_t* chunk_b_strides [[ buffer(7) ]],
    constant const size_t* state_strides [[ buffer(8) ]],
    constant const size_t* prefix_strides [[ buffer(9) ]],
    constant const uint& num_heads [[ buffer(10) ]],
    constant const uint& head_dim [[ buffer(11) ]],
    uint pair_idx [[ threadgroup_position_in_grid ]],
    ushort lane [[ thread_index_in_threadgroup ]]);
template [[host_name("ssd_prefill_chunk_scan_kernel_half")]]
kernel void ssd_prefill_chunk_scan_kernel<half>(
    device const float* chunk_a [[ buffer(0) ]],
    device const float* chunk_b [[ buffer(1) ]],
    device const half* state_in [[ buffer(2) ]],
    device float* prefix_state [[ buffer(3) ]],
    constant const size_t& suffix_len [[ buffer(4) ]],
    constant const int& state_size [[ buffer(5) ]],
    constant const size_t* chunk_a_strides [[ buffer(6) ]],
    constant const size_t* chunk_b_strides [[ buffer(7) ]],
    constant const size_t* state_strides [[ buffer(8) ]],
    constant const size_t* prefix_strides [[ buffer(9) ]],
    constant const uint& num_heads [[ buffer(10) ]],
    constant const uint& head_dim [[ buffer(11) ]],
    uint pair_idx [[ threadgroup_position_in_grid ]],
    ushort lane [[ thread_index_in_threadgroup ]]);

template [[host_name("ssd_prefill_chunk_apply_kernel_float")]]
kernel void ssd_prefill_chunk_apply_kernel<float>(
    device const float* x [[ buffer(0) ]],
    device const float* dt [[ buffer(1) ]],
    device const float* decay [[ buffer(2) ]],
    device const float* B [[ buffer(3) ]],
    device const float* C [[ buffer(4) ]],
    device const float* D [[ buffer(5) ]],
    device const float* z [[ buffer(6) ]],
    device float* state [[ buffer(7) ]],
    device float* y [[ buffer(8) ]],
    device const float* prefix_state [[ buffer(9) ]],
    constant const size_t& suffix_len [[ buffer(10) ]],
    constant const int& group_size [[ buffer(11) ]],
    constant const int& state_size [[ buffer(12) ]],
    constant const size_t* x_strides [[ buffer(13) ]],
    constant const size_t* dt_strides [[ buffer(14) ]],
    constant const size_t* cb_strides [[ buffer(15) ]],
    constant const size_t* state_strides [[ buffer(16) ]],
    constant const size_t* prefix_strides [[ buffer(17) ]],
    constant const uint& num_heads [[ buffer(18) ]],
    constant const uint& head_dim [[ buffer(19) ]],
    uint2 tg_pos [[ threadgroup_position_in_grid ]],
    ushort lane [[ thread_index_in_threadgroup ]]);
template [[host_name("ssd_prefill_chunk_apply_kernel_bfloat")]]
kernel void ssd_prefill_chunk_apply_kernel<bfloat>(
    device const bfloat* x [[ buffer(0) ]],
    device const bfloat* dt [[ buffer(1) ]],
    device const bfloat* decay [[ buffer(2) ]],
    device const bfloat* B [[ buffer(3) ]],
    device const bfloat* C [[ buffer(4) ]],
    device const bfloat* D [[ buffer(5) ]],
    device const bfloat* z [[ buffer(6) ]],
    device bfloat* state [[ buffer(7) ]],
    device bfloat* y [[ buffer(8) ]],
    device const float* prefix_state [[ buffer(9) ]],
    constant const size_t& suffix_len [[ buffer(10) ]],
    constant const int& group_size [[ buffer(11) ]],
    constant const int& state_size [[ buffer(12) ]],
    constant const size_t* x_strides [[ buffer(13) ]],
    constant const size_t* dt_strides [[ buffer(14) ]],
    constant const size_t* cb_strides [[ buffer(15) ]],
    constant const size_t* state_strides [[ buffer(16) ]],
    constant const size_t* prefix_strides [[ buffer(17) ]],
    constant const uint& num_heads [[ buffer(18) ]],
    constant const uint& head_dim [[ buffer(19) ]],
    uint2 tg_pos [[ threadgroup_position_in_grid ]],
    ushort lane [[ thread_index_in_threadgroup ]]);
template [[host_name("ssd_prefill_chunk_apply_kernel_half")]]
kernel void ssd_prefill_chunk_apply_kernel<half>(
    device const half* x [[ buffer(0) ]],
    device const half* dt [[ buffer(1) ]],
    device const half* decay [[ buffer(2) ]],
    device const half* B [[ buffer(3) ]],
    device const half* C [[ buffer(4) ]],
    device const half* D [[ buffer(5) ]],
    device const half* z [[ buffer(6) ]],
    device half* state [[ buffer(7) ]],
    device half* y [[ buffer(8) ]],
    device const float* prefix_state [[ buffer(9) ]],
    constant const size_t& suffix_len [[ buffer(10) ]],
    constant const int& group_size [[ buffer(11) ]],
    constant const int& state_size [[ buffer(12) ]],
    constant const size_t* x_strides [[ buffer(13) ]],
    constant const size_t* dt_strides [[ buffer(14) ]],
    constant const size_t* cb_strides [[ buffer(15) ]],
    constant const size_t* state_strides [[ buffer(16) ]],
    constant const size_t* prefix_strides [[ buffer(17) ]],
    constant const uint& num_heads [[ buffer(18) ]],
    constant const uint& head_dim [[ buffer(19) ]],
    uint2 tg_pos [[ threadgroup_position_in_grid ]],
    ushort lane [[ thread_index_in_threadgroup ]]);



template <typename T>
kernel void ssd_prefill_kernel_sequential(
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
        const T this_dt = dt[dt_idx];
        const T this_decay = decay[dt_idx];
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

instantiate_ssd_prefill_kernel_sequential(float, float);
instantiate_ssd_prefill_kernel_sequential(bfloat, bfloat);
instantiate_ssd_prefill_kernel_sequential(half, half);

#undef instantiate_ssd_prefill_kernel_sequential
