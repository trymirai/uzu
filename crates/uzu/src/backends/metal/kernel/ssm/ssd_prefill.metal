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
    const size_t state_base = h_idx * state_head_stride + dh_idx * state_dim_stride;
    const float d_scalar = float(D[h_idx]);

threadgroup float shared_state[SSM_PREFILL_MAX_STATE];
threadgroup float state_block_start[SSM_PREFILL_MAX_STATE];
threadgroup float chunk_end_state[SSM_PREFILL_MAX_STATE];
threadgroup float chunk_a[SSM_PREFILL_CHUNK];
threadgroup float chunk_b[SSM_PREFILL_CHUNK * SSM_PREFILL_MAX_STATE];

    // Load persistent state into threadgroup memory (strided across lanes)
    for (int s = lane; s < state_dim; s += SSM_PREFILL_CHUNK) {
        const size_t idx = state_base + size_t(s) * state_inner_stride;
        shared_state[s] = float(state[idx]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const size_t chunk_count =
        (suffix_len + size_t(SSM_PREFILL_CHUNK) - 1) / size_t(SSM_PREFILL_CHUNK);

    for (size_t chunk = 0; chunk < chunk_count; ++chunk) {
        const size_t chunk_start = chunk * size_t(SSM_PREFILL_CHUNK);
        if (chunk_start >= suffix_len) {
            break;
        }
        const ushort chunk_len = ushort(min(
            size_t(SSM_PREFILL_CHUNK), suffix_len - chunk_start));
        if (chunk_len == 0) {
            break;
        }

        // Snapshot current state before processing this chunk
        for (int s = lane; s < state_dim; s += SSM_PREFILL_CHUNK) {
            state_block_start[s] = shared_state[s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const bool lane_active = lane < chunk_len;
        float gate = 0.0f;
        float skip = 0.0f;
        size_t y_idx = 0;
        float c_vec[SSM_PREFILL_MAX_STATE];

        threadgroup float* b_slot = chunk_b + size_t(lane) * SSM_PREFILL_MAX_STATE;
        for (int s = 0; s < state_dim; ++s) {
            b_slot[s] = 0.0f;
        }
        chunk_a[lane] = 1.0f;

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

            chunk_a[lane] = decay_val;
            gate = SILU{}(float(z[x_idx]));
            skip = d_scalar * x_val;
            y_idx = x_idx;

            int s = 0;
            for (; s + 4 <= state_dim; s += 4) {
                const size_t offset = cb_idx + size_t(s) * cb_state_stride;
                auto b_vals = *reinterpret_cast<device const vec<T, 4>*>(B + offset);
                auto c_vals = *reinterpret_cast<device const vec<T, 4>*>(C + offset);
                float4 scaled = float4(b_vals) * dt_scaled_input;
                float4 c_pack = float4(c_vals);
                *reinterpret_cast<threadgroup float4*>(b_slot + s) = scaled;
                *reinterpret_cast<thread float4*>(c_vec + s) = c_pack;
            }
            for (; s < state_dim; ++s) {
                const size_t offset = cb_idx + size_t(s) * cb_state_stride;
                const float b_coeff = float(B[offset]);
                const float c_coeff = float(C[offset]);
                b_slot[s] = b_coeff * dt_scaled_input;
                c_vec[s] = c_coeff;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (ushort stride = 1; stride < chunk_len; stride <<= 1) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane_active && lane >= stride) {
                const float prev_a = chunk_a[lane - stride];
                float curr_a = chunk_a[lane];
                threadgroup const float* prev_b =
                    chunk_b + size_t(lane - stride) * SSM_PREFILL_MAX_STATE;
                threadgroup float* curr_b =
                    chunk_b + size_t(lane) * SSM_PREFILL_MAX_STATE;
                for (int s = 0; s < state_dim; ++s) {
                    curr_b[s] = curr_b[s] + curr_a * prev_b[s];
                }
                chunk_a[lane] = curr_a * prev_a;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane_active) {
            const float prefix_a = chunk_a[lane];
            threadgroup const float* prefix_b =
                chunk_b + size_t(lane) * SSM_PREFILL_MAX_STATE;
            float acc = skip;
            for (int s = 0; s < state_dim; ++s) {
                const float new_state =
                    prefix_a * state_block_start[s] + prefix_b[s];
                acc += new_state * c_vec[s];
                if (lane == chunk_len - 1) {
                    chunk_end_state[s] = new_state;
                }
            }
            y[y_idx] = static_cast<T>(acc * gate);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < chunk_len) {
            for (int s = lane; s < state_dim; s += chunk_len) {
                shared_state[s] = chunk_end_state[s];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Persist the updated state back to global memory
    for (int s = lane; s < state_dim; s += SSM_PREFILL_CHUNK) {
        const size_t idx = state_base + size_t(s) * state_inner_stride;
        state[idx] = static_cast<T>(shared_state[s]);
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
