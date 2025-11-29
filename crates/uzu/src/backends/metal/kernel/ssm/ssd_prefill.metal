#include <metal_stdlib>
#include <metal_simdgroup>
#include "../definitions.metal"

using namespace metal;

constant ushort SSM_PREFILL_MAX_STATE = 256;

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
kernel void ssd_prefill_kernel_64(
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

    float state0 = 0.0f;
    float state1 = 0.0f;
    size_t state_idx0 = 0;
    size_t state_idx1 = 0;
    state_idx0 = state_base + size_t(idx0) * state_inner_stride;
    state0 = float(state[state_idx0]);

    state_idx1 = state_base + size_t(idx1) * state_inner_stride;
    state1 = float(state[state_idx1]);

    size_t cb_idx0 = 0;
    size_t cb_idx1 = 0;
    cb_idx0 = cb_group_base + size_t(idx0) * cb_state_stride;
    cb_idx1 = cb_group_base + size_t(idx1) * cb_state_stride;

    for (size_t token = 0; token < suffix_len; ++token) {
        const size_t x_idx = token * x_token_stride + x_base;
        const size_t dt_idx = token * dt_token_stride + dt_base;

        const float x_val = float(x[x_idx]);
        const float decay_val = fast::exp(-float(softplus(float(dt_raw[dt_idx]))));
        const float gate = float(SILU{}(z[x_idx]));
        const float skip = d_scalar * x_val;
        const float dt_scaled_input = x_val;

        float contrib = 0.0f;
        const float new_state0 = decay_val * state0 + dt_scaled_input * float(B[cb_idx0]);
        state0 = new_state0;
        contrib += new_state0 * float(C[cb_idx0]);
        cb_idx0 += cb_token_stride;

        const float new_state1 = decay_val * state1 + dt_scaled_input * float(B[cb_idx1]);
        state1 = new_state1;
        contrib += new_state1 * float(C[cb_idx1]);
        cb_idx1 += cb_token_stride;

        float dot = simd_sum(contrib);
        if (lane_idx == 0) {
            y[x_idx] = static_cast<T>((skip + dot) * gate);
        }
    }

    state[state_idx0] = static_cast<T>(state0);
    state[state_idx1] = static_cast<T>(state1);
}

#define instantiate_ssd_prefill_kernel_64(type_name, type)           \
  template [[host_name("ssd_prefill_kernel_64_" #type_name)]]        \
  kernel void ssd_prefill_kernel_64<type>(                           \
    device const type* x [[ buffer(0) ]],                            \
    device const type* dt_raw [[ buffer(1) ]],                       \
    device const type* B [[ buffer(2) ]],                            \
    device const type* C [[ buffer(3) ]],                            \
    device const type* D [[ buffer(4) ]],                            \
    device const type* z [[ buffer(5) ]],                            \
    device type* state [[ buffer(6) ]],                              \
    device type* y [[ buffer(7) ]],                                  \
    constant const size_t& suffix_len [[ buffer(8) ]],               \
    constant const int& group_size [[ buffer(9) ]],                  \
    constant const int& state_size [[ buffer(10) ]],                 \
    constant const size_t* x_strides [[ buffer(11) ]],               \
    constant const size_t* dt_strides [[ buffer(12) ]],              \
    constant const size_t* cb_strides [[ buffer(13) ]],              \
    constant const size_t* state_strides [[ buffer(14) ]],           \
    constant const uint& num_heads [[ buffer(15) ]],                 \
    constant const uint& head_dim [[ buffer(16) ]],                  \
    uint3 tg_pos [[ threadgroup_position_in_grid ]],                 \
    ushort lane [[ thread_index_in_threadgroup ]]                    \
  );

instantiate_ssd_prefill_kernel_64(float, float);
instantiate_ssd_prefill_kernel_64(bfloat, bfloat);
instantiate_ssd_prefill_kernel_64(half, half);

#undef instantiate_ssd_prefill_kernel_64

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

    const size_t x_base = size_t(h_idx) * x_head_stride + size_t(dh_idx) * x_dim_stride;
    const size_t dt_base = size_t(h_idx) * dt_head_stride;
    const size_t state_base = size_t(h_idx) * state_head_stride + size_t(dh_idx) * state_dim_stride;
    const size_t cb_group_base = size_t(group_idx) * cb_group_stride;
    const float d_scalar = float(D[h_idx]);

    const int max_chunks = SSM_PREFILL_MAX_STATE / simd_width;
    const int chunk_count =
        (state_dim + int(simd_width) - 1) / int(simd_width);
    if (state_dim <= 0 || chunk_count > max_chunks) {
        return;
    }

    thread float lane_states[SSM_PREFILL_MAX_STATE / 32];

    #pragma unroll
    for (int chunk = 0; chunk < chunk_count; ++chunk) {
        const int idx = chunk * int(simd_width) + int(lane_idx);
        const size_t state_idx =
            state_base + size_t(idx) * state_inner_stride;
        lane_states[chunk] = float(state[state_idx]);
    }

    for (size_t token = 0; token < suffix_len; ++token) {
        const size_t x_idx = token * x_token_stride + x_base;
        const size_t dt_idx = token * dt_token_stride + dt_base;

        const float x_val = float(x[x_idx]);
        const float decay_val = fast::exp(-float(softplus(float(dt_raw[dt_idx]))));
        const float gate = float(SILU{}(z[x_idx]));
        const float skip = d_scalar * x_val;
        const float dt_scaled_input = x_val;

        float contrib_sum = 0.0f;
        #pragma unroll
        for (int chunk = 0; chunk < chunk_count; ++chunk) {
            const int idx = chunk * int(simd_width) + int(lane_idx);
            const size_t cb_idx = cb_group_base + size_t(idx) * cb_state_stride + token * cb_token_stride;

            const float new_state = decay_val * lane_states[chunk] + dt_scaled_input * float(B[cb_idx]);
            lane_states[chunk] = new_state;
            contrib_sum += new_state * float(C[cb_idx]);
        }

        float dot = simd_sum(contrib_sum);
        if (lane_idx == 0) {
            y[x_idx] = static_cast<T>((skip + dot) * gate);
        }
    }

    #pragma unroll
    for (int chunk = 0; chunk < chunk_count; ++chunk) {
        const int idx = chunk * int(simd_width) + int(lane_idx);
        const size_t state_idx = state_base + size_t(idx) * state_inner_stride;
        state[state_idx] = static_cast<T>(lane_states[chunk]);
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
        const T dt_scaled_input = this_x;

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

// ============================================================================
// 
// Pass 1: Precompute CB[g,t,s] = B[g,s,:] · C[g,t,:] and decay prefix sums
// Pass 2: Compute y[h,t,dh] = sum_{s<=t} decay[s:t] * CB[g,t,s] * x[h,s,dh]
// ============================================================================

// Pass 1: Compute decay prefix sums per head
// Grid: (num_heads,) - one threadgroup per head
// Output: cum_log_decay[h, t] = sum_{i=0}^{t-1} log_decay[h,i]
template <typename T>
[[max_total_threads_per_threadgroup(256)]]
kernel void ssd_prefill_flash_pass1_decay(
    device const T* dt_raw [[ buffer(0) ]],       // (suffix, h)
    device float* cum_log_decay [[ buffer(1) ]],  // (h, suffix+1)
    constant const size_t& suffix_len [[ buffer(2) ]],
    constant const size_t& dt_token_stride [[ buffer(3) ]],
    constant const size_t& dt_head_stride [[ buffer(4) ]],
    uint h_idx [[ threadgroup_position_in_grid ]],
    ushort lid [[ thread_index_in_threadgroup ]]
) {
    const uint suffix = uint(suffix_len);
    device float* out = cum_log_decay + h_idx * (suffix + 1);
    
    // Use raking approach: each of 256 threads handles suffix/256 elements
    threadgroup float tg_partial[256];
    threadgroup float tg_prefix[256];
    
    const uint elements_per_thread = (suffix + 255) / 256;
    const uint my_start = uint(lid) * elements_per_thread;
    const uint my_end = min(my_start + elements_per_thread, suffix);
    
    // Each thread computes local sum
    float local_sum = 0.0f;
    for (uint i = my_start; i < my_end; i++) {
        float dt_val = softplus(float(dt_raw[i * dt_token_stride + h_idx * dt_head_stride]));
        local_sum -= dt_val;
    }
    tg_partial[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Prefix sum across threads using raking
    float prefix = threadgroup_raking_prefix_exclusive_sum<256>(tg_partial[lid], tg_prefix, lid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float my_prefix = tg_prefix[lid];
    
    // Write output with prefix added
    out[0] = 0.0f;
    float running = my_prefix;
    for (uint i = my_start; i < my_end; i++) {
        float dt_val = softplus(float(dt_raw[i * dt_token_stride + h_idx * dt_head_stride]));
        running -= dt_val;
        out[i + 1] = running;
    }
}

// ============================================================================ 
// Pass 2a: CB = C @ B^T (GEMM)
// Pass 2b: CB *= decay * causal_mask (element-wise)
// Pass 2c: y = CB @ dtx (GEMM) + state contribution
// Pass 2d: y = (y + D*x) * silu(z) (element-wise)
// ============================================================================

#include <metal_simdgroup_matrix>

// GEMM tile sizes (from MoE kernel, proven to work well)
constant constexpr uint SSD_BM = 32;
constant constexpr uint SSD_BN = 32;
constant constexpr uint SSD_BK = 32;
constant constexpr uint SSD_SG_BM = 16;
constant constexpr uint SSD_SG_BN = 16;

// CB GEMM tile configuration
constant uint SSDCB_BM = 16;
constant uint SSDCB_BN = 32;
constant uint SSDCB_BK = 64;
constant uint SSDCB_SG_BM = 8;
constant uint SSDCB_SG_BN = 16;
constant uint SSDCB_SG_TILE = 8;


// Pass 2a+2b FUSED: CB = tril((C @ B^T) * decay_mask) with BM=32
// 4 accumulators per simdgroup for better occupancy
template <typename T>
kernel void ssd_gemm_cb_fused(
    device const T* C_in [[ buffer(0) ]],
    device const T* B_in [[ buffer(1) ]],
    device const float* cum_log_decay [[ buffer(2) ]],
    device float* CB_out [[ buffer(3) ]],
    constant const uint& L [[ buffer(4) ]],
    constant const uint& N [[ buffer(5) ]],
    constant const uint& H [[ buffer(6) ]],
    constant const uint& G [[ buffer(7) ]],
    constant const size_t& lda [[ buffer(8) ]],
    constant const size_t& group_stride [[ buffer(9) ]],
    uint3 tg_pos [[ threadgroup_position_in_grid ]],
    uint sg_id [[ simdgroup_index_in_threadgroup ]],
    uint simd_lid [[ thread_index_in_simdgroup ]]
) {
    constexpr uint BM = 32;  // Doubled from 16
    constexpr uint BN = 32;
    constexpr uint BK = 64;
    constexpr uint TGP_SIZE = 128;
    
    constexpr uint C_VEC = 8;
    constexpr uint C_TCOLS = BK / C_VEC;
    constexpr uint C_TROWS = TGP_SIZE / C_TCOLS;
    constexpr uint B_VEC = 8;
    constexpr uint B_TCOLS = BK / B_VEC;
    constexpr uint B_TROWS = TGP_SIZE / B_TCOLS;
    
    const uint tile_n = tg_pos.x;
    const uint tile_m = tg_pos.y;
    const uint h = tg_pos.z;
    
    const uint row_start = tile_m * BM;
    const uint col_start = tile_n * BN;
    
    const uint m_valid = min(BM, L - row_start);
    const uint n_valid = min(BN, L - col_start);
    
    const uint g = h / (H / G);
    device const T* C_base = C_in + g * group_stride + row_start * lda;
    device const T* B_base = B_in + g * group_stride + col_start * lda;
    device float* CB_h = CB_out + (size_t)h * L * L + row_start * L + col_start;
    device const float* my_decay = cum_log_decay + h * (L + 1);
    
    constexpr uint TG_PAD = 4;
    threadgroup float tg_C[BM * (BK + TG_PAD)];
    threadgroup float tg_B[BN * (BK + TG_PAD)];
    constexpr uint TG_C_LD = BK + TG_PAD;
    constexpr uint TG_B_LD = BK + TG_PAD;
    
    const uint lin = sg_id * 32 + simd_lid;
    const uint c_bi = lin / C_TCOLS;
    const uint c_bj = (lin % C_TCOLS) * C_VEC;
    const uint b_bi = lin / B_TCOLS;
    const uint b_bj = (lin % B_TCOLS) * B_VEC;
    
    // 2×2 simdgroup layout for 32×32 output
    const uint row_sg = sg_id / 2;
    const uint col_sg = sg_id % 2;
    const uint row_sg_off = row_sg * 16;
    const uint col_sg_off = col_sg * 16;
    
    // 4 accumulators per simdgroup
    metal::simdgroup_float8x8 acc[4];
    #pragma clang loop unroll(full)
    for (int i = 0; i < 4; i++) {
        acc[i] = metal::make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    
    {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Load C tile (BM=32 rows)
        {
            threadgroup float* my_dst = tg_C + c_bi * TG_C_LD + c_bj;
            if (c_bi < m_valid) {
                device const T* my_src = C_base + c_bi * lda + c_bj;
                #pragma clang loop unroll(full)
                for (uint j = 0; j < C_VEC; j++) {
                    my_dst[j] = float(my_src[j]);
                }
            } else {
                #pragma clang loop unroll(full)
                for (uint j = 0; j < C_VEC; j++) {
                    my_dst[j] = 0.0f;
                }
            }
            // Second half for BM=32
            if (c_bi + C_TROWS < BM) {
                threadgroup float* my_dst2 = my_dst + C_TROWS * TG_C_LD;
                if (c_bi + C_TROWS < m_valid) {
                    device const T* my_src2 = C_base + (c_bi + C_TROWS) * lda + c_bj;
                    #pragma clang loop unroll(full)
                    for (uint j = 0; j < C_VEC; j++) {
                        my_dst2[j] = float(my_src2[j]);
                    }
                } else {
                    #pragma clang loop unroll(full)
                    for (uint j = 0; j < C_VEC; j++) {
                        my_dst2[j] = 0.0f;
                    }
                }
            }
        }
        
        // Load B tile (BN=32 rows)
        {
            threadgroup float* my_dst = tg_B + b_bi * TG_B_LD + b_bj;
            if (b_bi < n_valid) {
                device const T* my_src = B_base + b_bi * lda + b_bj;
                #pragma clang loop unroll(full)
                for (uint j = 0; j < B_VEC; j++) {
                    my_dst[j] = float(my_src[j]);
                }
            } else {
                #pragma clang loop unroll(full)
                for (uint j = 0; j < B_VEC; j++) {
                    my_dst[j] = 0.0f;
                }
            }
            if (b_bi + B_TROWS < BN) {
                threadgroup float* my_dst2 = my_dst + B_TROWS * TG_B_LD;
                if (b_bi + B_TROWS < n_valid) {
                    device const T* my_src2 = B_base + (b_bi + B_TROWS) * lda + b_bj;
                    #pragma clang loop unroll(full)
                    for (uint j = 0; j < B_VEC; j++) {
                        my_dst2[j] = float(my_src2[j]);
                    }
                } else {
                    #pragma clang loop unroll(full)
                    for (uint j = 0; j < B_VEC; j++) {
                        my_dst2[j] = 0.0f;
                    }
                }
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // MMA with 4 accumulators (2×2 layout)
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < BK; kk += 8) {
            metal::simdgroup_float8x8 c_frag0, c_frag1;
            simdgroup_load(c_frag0, tg_C, TG_C_LD, ulong2(kk, row_sg_off), false);
            simdgroup_load(c_frag1, tg_C, TG_C_LD, ulong2(kk, row_sg_off + 8), false);
            
            metal::simdgroup_float8x8 b_frag0, b_frag1;
            simdgroup_load(b_frag0, tg_B, TG_B_LD, ulong2(kk, col_sg_off), true);
            simdgroup_load(b_frag1, tg_B, TG_B_LD, ulong2(kk, col_sg_off + 8), true);
            
            // Serpentine MMA pattern
            simdgroup_multiply_accumulate(acc[0], c_frag0, b_frag0, acc[0]);
            simdgroup_multiply_accumulate(acc[1], c_frag0, b_frag1, acc[1]);
            simdgroup_multiply_accumulate(acc[3], c_frag1, b_frag1, acc[3]);
            simdgroup_multiply_accumulate(acc[2], c_frag1, b_frag0, acc[2]);
        }
    }
    
    // Store with decay and causal mask - 4 fragments
    const uint qid = simd_lid >> 2;
    const uint lane_row = (qid & 4u) + ((simd_lid >> 1) & 3u);
    const uint lane_col_base = ((qid & 2u) << 1) + ((simd_lid & 1u) << 1);
    
    #pragma clang loop unroll(full)
    for (uint mi = 0; mi < 2; mi++) {
        const uint local_t = row_sg_off + mi * 8 + lane_row;
        if (local_t < m_valid) {
            const uint global_t = row_start + local_t;
            const float decay_t = my_decay[global_t + 1];
            device float* out_row = CB_h + local_t * L;
            
            #pragma clang loop unroll(full)
            for (uint ni = 0; ni < 2; ni++) {
                auto frag = acc[mi * 2 + ni].thread_elements();
                uint local_s = col_sg_off + ni * 8 + lane_col_base;
                uint global_s = col_start + local_s;
                
                if (local_s < n_valid) {
                    if (global_s > global_t) {
                        out_row[local_s] = 0.0f;
                    } else {
                        float decay = fast::exp(decay_t - my_decay[global_s + 1]);
                        out_row[local_s] = frag[0] * decay;
                    }
                }
                if (local_s + 1 < n_valid) {
                    if (global_s + 1 > global_t) {
                        out_row[local_s + 1] = 0.0f;
                    } else {
                        float decay = fast::exp(decay_t - my_decay[global_s + 2]);
                        out_row[local_s + 1] = frag[1] * decay;
                    }
                }
            }
        }
    }
}

// Pass 2c: y = CB @ x (BM=32 for better occupancy)
// Grid: (tiles_n, tiles_m, num_heads) where tiles cover output
// 4 accumulators per simdgroup
template <typename T>
kernel void ssd_gemm_y(
    device const float* CB [[ buffer(0) ]],
    device const T* x [[ buffer(1) ]],
    device float* y_out [[ buffer(2) ]],
    constant const uint& suffix_len [[ buffer(3) ]],
    constant const uint& num_heads [[ buffer(4) ]],
    constant const uint& head_dim [[ buffer(5) ]],
    constant const size_t* x_strides [[ buffer(6) ]],
    uint3 tg_pos [[ threadgroup_position_in_grid ]],
    uint sg_id [[ simdgroup_index_in_threadgroup ]],
    uint simd_lid [[ thread_index_in_simdgroup ]]
) {
    constexpr uint BM = 32;   // Doubled from 16
    constexpr uint BN = 32;
    constexpr uint BK = 64;
    constexpr uint TGP_SIZE = 128;
    
    constexpr uint CB_VEC = 8;
    constexpr uint CB_TCOLS = BK / CB_VEC;
    constexpr uint CB_TROWS = TGP_SIZE / CB_TCOLS;
    constexpr uint X_VEC = 8;
    constexpr uint X_TCOLS = BN / X_VEC;
    constexpr uint X_TROWS = TGP_SIZE / X_TCOLS;
    
    const uint tile_n = tg_pos.x;
    const uint tile_m = tg_pos.y;
    const uint h_idx = tg_pos.z;
    
    const uint L = suffix_len;
    const uint dh = head_dim;
    
    const uint row_start = tile_m * BM;
    const uint col_start = tile_n * BN;
    const uint m_valid = min(BM, L - row_start);
    
    const size_t x_token_stride = x_strides[0];
    const size_t x_head_stride = x_strides[1];
    
    device const float* CB_base = CB + (size_t)h_idx * L * L + row_start * L;
    device const T* x_base = x + h_idx * x_head_stride + col_start;
    device float* y_base = y_out + h_idx * x_head_stride + row_start * x_token_stride + col_start;
    
    constexpr uint TG_PAD = 4;
    threadgroup float tg_CB[BM * (BK + TG_PAD)];
    threadgroup float tg_X[BK * (BN + TG_PAD)];
    constexpr uint TG_CB_LD = BK + TG_PAD;
    constexpr uint TG_X_LD = BN + TG_PAD;
    
    const uint lin = sg_id * 32 + simd_lid;
    const uint cb_bi = lin / CB_TCOLS;
    const uint cb_bj = (lin % CB_TCOLS) * CB_VEC;
    const uint x_bi = lin / X_TCOLS;
    const uint x_bj = (lin % X_TCOLS) * X_VEC;
    
    // 2×2 simdgroup layout for 32×32 output
    const uint row_sg = sg_id / 2;
    const uint col_sg = sg_id % 2;
    const uint row_sg_off = row_sg * 16;  // Each simdgroup handles 16 rows
    const uint col_sg_off = col_sg * 16;  // Each simdgroup handles 16 cols
    
    // 4 accumulators per simdgroup (2×2 fragments of 8×8)
    metal::simdgroup_float8x8 acc[4];
    #pragma clang loop unroll(full)
    for (int i = 0; i < 4; i++) {
        acc[i] = metal::make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    
    for (uint k_off = 0; k_off < L; k_off += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Load CB tile - need to load BM=32 rows
        {
            threadgroup float* my_dst = tg_CB + cb_bi * TG_CB_LD + cb_bj;
            if (cb_bi < m_valid) {
                device const float* my_src = CB_base + cb_bi * L + k_off + cb_bj;
                #pragma clang loop unroll(full)
                for (uint j = 0; j < CB_VEC; j++) {
                    my_dst[j] = my_src[j];
                }
            } else {
                #pragma clang loop unroll(full)
                for (uint j = 0; j < CB_VEC; j++) {
                    my_dst[j] = 0.0f;
                }
            }
            // Second half for BM=32
            if (cb_bi + CB_TROWS < BM) {
                threadgroup float* my_dst2 = my_dst + CB_TROWS * TG_CB_LD;
                if (cb_bi + CB_TROWS < m_valid) {
                    device const float* my_src2 = CB_base + (cb_bi + CB_TROWS) * L + k_off + cb_bj;
                    #pragma clang loop unroll(full)
                    for (uint j = 0; j < CB_VEC; j++) {
                        my_dst2[j] = my_src2[j];
                    }
                } else {
                    #pragma clang loop unroll(full)
                    for (uint j = 0; j < CB_VEC; j++) {
                        my_dst2[j] = 0.0f;
                    }
                }
            }
        }
        
        // Load X tile
        {
            threadgroup float* my_dst = tg_X + x_bi * TG_X_LD + x_bj;
            if (k_off + x_bi < L) {
                device const T* my_src = x_base + (k_off + x_bi) * x_token_stride + x_bj;
                #pragma clang loop unroll(full)
                for (uint j = 0; j < X_VEC; j++) {
                    my_dst[j] = float(my_src[j]);
                }
            } else {
                #pragma clang loop unroll(full)
                for (uint j = 0; j < X_VEC; j++) {
                    my_dst[j] = 0.0f;
                }
            }
            if (x_bi + X_TROWS < BK) {
                threadgroup float* my_dst2 = my_dst + X_TROWS * TG_X_LD;
                if (k_off + x_bi + X_TROWS < L) {
                    device const T* my_src2 = x_base + (k_off + x_bi + X_TROWS) * x_token_stride + x_bj;
                    #pragma clang loop unroll(full)
                    for (uint j = 0; j < X_VEC; j++) {
                        my_dst2[j] = float(my_src2[j]);
                    }
                } else {
                    #pragma clang loop unroll(full)
                    for (uint j = 0; j < X_VEC; j++) {
                        my_dst2[j] = 0.0f;
                    }
                }
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // MMA with 4 accumulators (2×2 layout of 8×8 fragments)
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < BK; kk += 8) {
            // Load 2 CB fragments for rows
            metal::simdgroup_float8x8 cb_frag0, cb_frag1;
            simdgroup_load(cb_frag0, tg_CB, TG_CB_LD, ulong2(kk, row_sg_off), false);
            simdgroup_load(cb_frag1, tg_CB, TG_CB_LD, ulong2(kk, row_sg_off + 8), false);
            
            // Load 2 X fragments for cols
            metal::simdgroup_float8x8 x_frag0, x_frag1;
            simdgroup_load(x_frag0, tg_X, TG_X_LD, ulong2(col_sg_off, kk), false);
            simdgroup_load(x_frag1, tg_X, TG_X_LD, ulong2(col_sg_off + 8, kk), false);
            
            // 2×2 MMA (serpentine for better register reuse)
            simdgroup_multiply_accumulate(acc[0], cb_frag0, x_frag0, acc[0]);
            simdgroup_multiply_accumulate(acc[1], cb_frag0, x_frag1, acc[1]);
            simdgroup_multiply_accumulate(acc[3], cb_frag1, x_frag1, acc[3]);
            simdgroup_multiply_accumulate(acc[2], cb_frag1, x_frag0, acc[2]);
        }
    }
    
    // Store results
    const uint qid = simd_lid >> 2;
    const uint lane_row = (qid & 4u) + ((simd_lid >> 1) & 3u);
    const uint lane_col_base = ((qid & 2u) << 1) + ((simd_lid & 1u) << 1);
    
    // Store all 4 fragments (2×2)
    #pragma clang loop unroll(full)
    for (uint mi = 0; mi < 2; mi++) {
        const uint local_t = row_sg_off + mi * 8 + lane_row;
        if (local_t < m_valid) {
            device float* out_row = y_base + local_t * x_token_stride;
            #pragma clang loop unroll(full)
            for (uint ni = 0; ni < 2; ni++) {
                auto frag = acc[mi * 2 + ni].thread_elements();
                const uint col0 = col_sg_off + ni * 8 + lane_col_base;
                out_row[col0] = frag[0];
                out_row[col0 + 1] = frag[1];
            }
        }
    }
}

// Pass 2c2: state_C = state @ C^T using simdgroup MMA
// Computes state_C[H, dh, L] = state[H, dh, N] @ C[L, G, N]^T (broadcast over groups)
// Grid: (tiles_n, tiles_m, num_heads) where M=dh, N=L, K=state_dim
// Assumes: dh % 16 = 0, N % 64 = 0 (aligned). L can be any value.
template <typename T>
kernel void ssd_gemm_state_c(
    device const T* state [[ buffer(0) ]],         // [H, dh, N]
    device const T* C_in [[ buffer(1) ]],          // [L, groups, N]
    device float* state_C [[ buffer(2) ]],         // [H, dh, L]
    constant const uint& suffix_len [[ buffer(3) ]],
    constant const uint& num_heads [[ buffer(4) ]],
    constant const uint& head_dim [[ buffer(5) ]],
    constant const uint& state_dim [[ buffer(6) ]],
    constant const uint& num_groups [[ buffer(7) ]],
    constant const size_t* cb_strides [[ buffer(8) ]],
    constant const size_t* state_strides [[ buffer(9) ]],
    uint3 tg_pos [[ threadgroup_position_in_grid ]],
    uint sg_id [[ simdgroup_index_in_threadgroup ]],
    uint simd_lid [[ thread_index_in_simdgroup ]]
) {
    constexpr uint BM = 16;   // dh tiles (aligned)
    constexpr uint BN = 32;   // L tiles
    constexpr uint BK = 64;   // N (state_dim) tiles (aligned)
    constexpr uint TGP_SIZE = 128;
    
    constexpr uint S_VEC = 8;
    constexpr uint S_TCOLS = BK / S_VEC;
    constexpr uint C_VEC = 8;
    constexpr uint C_TCOLS = BK / C_VEC;
    constexpr uint C_TROWS = TGP_SIZE / C_TCOLS;
    
    const uint tile_n = tg_pos.x;
    const uint tile_m = tg_pos.y;
    const uint h_idx = tg_pos.z;
    
    const uint L = suffix_len;
    const uint dh = head_dim;
    const uint N = state_dim;
    const uint group_idx = h_idx / (num_heads / num_groups);
    
    const uint row_start = tile_m * BM;
    const uint col_start = tile_n * BN;
    const uint n_valid = min(BN, L - col_start);
    
    const size_t cb_token_stride = cb_strides[0];
    const size_t cb_group_stride = cb_strides[1];
    const size_t state_head_stride = state_strides[0];
    const size_t state_dim_stride = state_strides[1];
    
    device const T* state_base = state + h_idx * state_head_stride + row_start * state_dim_stride;
    device const T* C_base = C_in + group_idx * cb_group_stride + col_start * cb_token_stride;
    device float* out_base = state_C + h_idx * dh * L + row_start * L + col_start;
    
    constexpr uint TG_PAD = 4;
    threadgroup float tg_state[BM * (BK + TG_PAD)];
    threadgroup float tg_C[BN * (BK + TG_PAD)];
    constexpr uint TG_S_LD = BK + TG_PAD;
    constexpr uint TG_C_LD = BK + TG_PAD;
    
    const uint lin = sg_id * 32 + simd_lid;
    const uint s_bi = lin / S_TCOLS;
    const uint s_bj = (lin % S_TCOLS) * S_VEC;
    const uint c_bi = lin / C_TCOLS;
    const uint c_bj = (lin % C_TCOLS) * C_VEC;
    
    const uint row_sg = sg_id / 2;
    const uint col_sg = sg_id % 2;
    const uint row_sg_off = row_sg * 8;
    const uint col_sg_off = col_sg * 16;
    
    metal::simdgroup_float8x8 acc[2];
    acc[0] = metal::make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    acc[1] = metal::make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    
    {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Load state tile: aligned, no bounds check needed
        {
            device const T* my_src = state_base + s_bi * state_dim_stride + s_bj;
            threadgroup float* my_dst = tg_state + s_bi * TG_S_LD + s_bj;
            #pragma clang loop unroll(full)
            for (uint j = 0; j < S_VEC; j++) {
                my_dst[j] = float(my_src[j]);
            }
        }
        
        // Load C tile with L bounds check
        {
            threadgroup float* my_dst = tg_C + c_bi * TG_C_LD + c_bj;
            if (c_bi < n_valid) {
                device const T* my_src = C_base + c_bi * cb_token_stride + c_bj;
                #pragma clang loop unroll(full)
                for (uint j = 0; j < C_VEC; j++) {
                    my_dst[j] = float(my_src[j]);
                }
            } else {
                #pragma clang loop unroll(full)
                for (uint j = 0; j < C_VEC; j++) {
                    my_dst[j] = 0.0f;
                }
            }
            
            // Second half
            if (c_bi + C_TROWS < BN) {
                threadgroup float* my_dst2 = my_dst + C_TROWS * TG_C_LD;
                if (c_bi + C_TROWS < n_valid) {
                    device const T* my_src2 = C_base + (c_bi + C_TROWS) * cb_token_stride + c_bj;
                    #pragma clang loop unroll(full)
                    for (uint j = 0; j < C_VEC; j++) {
                        my_dst2[j] = float(my_src2[j]);
                    }
                } else {
                    #pragma clang loop unroll(full)
                    for (uint j = 0; j < C_VEC; j++) {
                        my_dst2[j] = 0.0f;
                    }
                }
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < BK; kk += 8) {
            metal::simdgroup_float8x8 s_frag;
            simdgroup_load(s_frag, tg_state, TG_S_LD, ulong2(kk, row_sg_off), false);
            
            #pragma clang loop unroll(full)
            for (uint n_sub = 0; n_sub < 16; n_sub += 8) {
                metal::simdgroup_float8x8 c_frag;
                simdgroup_load(c_frag, tg_C, TG_C_LD, ulong2(kk, col_sg_off + n_sub), true);
                simdgroup_multiply_accumulate(acc[n_sub / 8], s_frag, c_frag, acc[n_sub / 8]);
            }
        }
    }
    
    // Store with L bounds check
    const uint qid = simd_lid >> 2;
    const uint lane_row = (qid & 4u) + ((simd_lid >> 1) & 3u);
    const uint lane_col_base = ((qid & 2u) << 1) + ((simd_lid & 1u) << 1);
    
    const uint local_d = row_sg_off + lane_row;
    device float* out_row = out_base + local_d * L;
    
    #pragma clang loop unroll(full)
    for (uint n_sub = 0; n_sub < 16; n_sub += 8) {
        auto frag = acc[n_sub / 8].thread_elements();
        uint local_t = col_sg_off + n_sub + lane_col_base;
        if (local_t < n_valid) out_row[local_t] = frag[0];
        if (local_t + 1 < n_valid) out_row[local_t + 1] = frag[1];
    }
}

// Pass 4 FUSED: add_state + skip_gate in one pass
// Computes: y_out = (y_in + decay * state_C + D * x) * silu(z)
template <typename T>
kernel void ssd_fused_add_skip_gate(
    device const float* y_in [[ buffer(0) ]],      // [L, H, dh] from y GEMM
    device const float* state_C [[ buffer(1) ]],   // [H, dh, L]
    device const float* cum_log_decay [[ buffer(2) ]], // [H, L+1]
    device const T* x [[ buffer(3) ]],             // [L, H, dh]
    device const T* D [[ buffer(4) ]],             // [H]
    device const T* z [[ buffer(5) ]],             // [L, H, dh]
    device T* y_out [[ buffer(6) ]],               // [L, H, dh]
    constant const uint& suffix_len [[ buffer(7) ]],
    constant const uint& num_heads [[ buffer(8) ]],
    constant const uint& head_dim [[ buffer(9) ]],
    constant const size_t* x_strides [[ buffer(10) ]],
    uint tg_pos [[ threadgroup_position_in_grid ]],
    uint tid [[ thread_index_in_threadgroup ]]
) {
    const uint idx = tg_pos * 256 + tid;
    const uint L = suffix_len;
    const uint H = num_heads;
    const uint dh = head_dim;
    const uint total = L * H * dh;
    
    if (idx >= total) return;
    
    // Decode idx to (t, h, d)
    const uint d = idx % dh;
    const uint h = (idx / dh) % H;
    const uint t = idx / (dh * H);
    
    // Compute linear index for x/z/y_out (may have different strides)
    const size_t xz_idx = t * x_strides[0] + h * x_strides[1] + d * x_strides[2];
    
    // Load values
    float y_val = y_in[xz_idx];
    float x_val = float(x[xz_idx]);
    float z_val = float(z[xz_idx]);
    float d_val = float(D[h]);
    
    // Decay for state contribution: exp(cum_log_decay[h, t+1])
    float decay_0_t = fast::exp(cum_log_decay[h * (L + 1) + t + 1]);
    
    // state_C[h, d, t]
    float sc = state_C[h * dh * L + d * L + t];
    
    // Fused computation: (y + decay*state_c + D*x) * silu(z)
    float gate = z_val / (1.0f + fast::exp(-z_val));  // SiLU
    float result = (y_val + decay_0_t * sc + d_val * x_val) * gate;
    
    y_out[xz_idx] = static_cast<T>(result);
}

// =============================================================================
// State Update Kernels
// =============================================================================

// State update GEMM: contribution = (scaled_x)^T @ B
// Computes: contribution[H, dh, N] = Σ_t (decay_t_to_T * x[t, h, d]) * B[t, g, n]
// This is: [dh, L] @ [L, N] = [dh, N] per head
// Grid: (tiles_n, tiles_m, num_heads) where M=dh, N=state_dim, K=L
// Assumes: dh % 16 = 0, N % 32 = 0 (aligned). L can be any value.
template <typename T>
kernel void ssd_gemm_state_update(
    device const T* x [[ buffer(0) ]],                 // [L, H, dh]
    device const T* B_in [[ buffer(1) ]],              // [L, G, N]
    device const float* cum_log_decay [[ buffer(2) ]], // [H, L+1]
    device float* contribution [[ buffer(3) ]],        // [H, dh, N]
    constant const uint& suffix_len [[ buffer(4) ]],
    constant const uint& num_heads [[ buffer(5) ]],
    constant const uint& head_dim [[ buffer(6) ]],
    constant const uint& state_dim [[ buffer(7) ]],
    constant const uint& num_groups [[ buffer(8) ]],
    constant const size_t* x_strides [[ buffer(9) ]],
    constant const size_t* cb_strides [[ buffer(10) ]],
    uint3 tg_pos [[ threadgroup_position_in_grid ]],
    uint sg_id [[ simdgroup_index_in_threadgroup ]],
    uint simd_lid [[ thread_index_in_simdgroup ]]
) {
    constexpr uint BM = 16;   // dh tiles (aligned)
    constexpr uint BN = 32;   // N (state_dim) tiles (aligned)
    constexpr uint BK = 64;   // L (sequence) tiles
    constexpr uint TGP_SIZE = 128;
    
    constexpr uint X_VEC = 8;
    constexpr uint X_TCOLS = BK / X_VEC;
    constexpr uint B_VEC = 8;
    constexpr uint B_TCOLS = BK / B_VEC;
    constexpr uint B_TROWS = TGP_SIZE / B_TCOLS;
    
    const uint tile_n = tg_pos.x;
    const uint tile_m = tg_pos.y;
    const uint h_idx = tg_pos.z;
    
    const uint L = suffix_len;
    const uint dh = head_dim;
    const uint N = state_dim;
    const uint group_idx = h_idx / (num_heads / num_groups);
    
    const uint row_start = tile_m * BM;
    const uint col_start = tile_n * BN;
    
    const size_t x_token_stride = x_strides[0];
    const size_t x_head_stride = x_strides[1];
    const size_t cb_token_stride = cb_strides[0];
    const size_t cb_group_stride = cb_strides[1];
    
    const float total_log_decay = cum_log_decay[h_idx * (L + 1) + L];
    device const float* decay_base = cum_log_decay + h_idx * (L + 1) + 1;
    
    device const T* x_base = x + h_idx * x_head_stride + row_start;
    device const T* B_base = B_in + group_idx * cb_group_stride + col_start;
    device float* out_base = contribution + h_idx * dh * N + row_start * N + col_start;
    
    constexpr uint TG_PAD = 4;
    threadgroup float tg_x[BM * (BK + TG_PAD)];
    threadgroup float tg_B[BN * (BK + TG_PAD)];
    constexpr uint TG_X_LD = BK + TG_PAD;
    constexpr uint TG_B_LD = BK + TG_PAD;
    
    const uint lin = sg_id * 32 + simd_lid;
    const uint x_bi = lin / X_TCOLS;
    const uint x_bj = (lin % X_TCOLS) * X_VEC;
    const uint b_bi = lin / B_TCOLS;
    const uint b_bj = (lin % B_TCOLS) * B_VEC;
    
    const uint row_sg = sg_id / 2;
    const uint col_sg = sg_id % 2;
    const uint row_sg_off = row_sg * 8;
    const uint col_sg_off = col_sg * 16;
    
    metal::simdgroup_float8x8 acc[2];
    acc[0] = metal::make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    acc[1] = metal::make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    
    // K-loop with L bounds
    for (uint k_off = 0; k_off < L; k_off += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Load x tile with decay scaling and L bounds
        {
            const uint t_base = k_off + x_bj;
            threadgroup float* my_dst = tg_x + x_bi * TG_X_LD + x_bj;
            
            #pragma clang loop unroll(full)
            for (uint j = 0; j < X_VEC; j++) {
                if (t_base + j < L) {
                    float x_val = float(x_base[(t_base + j) * x_token_stride + x_bi]);
                    float decay_t_to_T = fast::exp(total_log_decay - decay_base[t_base + j]);
                    my_dst[j] = x_val * decay_t_to_T;
                } else {
                    my_dst[j] = 0.0f;
                }
            }
        }
        
        // Load B tile with L bounds
        {
            const uint t_base = k_off + b_bj;
            threadgroup float* my_dst = tg_B + b_bi * TG_B_LD + b_bj;
            
            #pragma clang loop unroll(full)
            for (uint j = 0; j < B_VEC; j++) {
                if (t_base + j < L) {
                    my_dst[j] = float(B_base[(t_base + j) * cb_token_stride + b_bi]);
                } else {
                    my_dst[j] = 0.0f;
                }
            }
            
            // Second half
            if (b_bi + B_TROWS < BN) {
                threadgroup float* my_dst2 = my_dst + B_TROWS * TG_B_LD;
                #pragma clang loop unroll(full)
                for (uint j = 0; j < B_VEC; j++) {
                    if (t_base + j < L) {
                        my_dst2[j] = float(B_base[(t_base + j) * cb_token_stride + b_bi + B_TROWS]);
                    } else {
                        my_dst2[j] = 0.0f;
                    }
                }
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < BK; kk += 8) {
            metal::simdgroup_float8x8 x_frag;
            simdgroup_load(x_frag, tg_x, TG_X_LD, ulong2(kk, row_sg_off), false);
            
            #pragma clang loop unroll(full)
            for (uint n_sub = 0; n_sub < 16; n_sub += 8) {
                metal::simdgroup_float8x8 b_frag;
                simdgroup_load(b_frag, tg_B, TG_B_LD, ulong2(kk, col_sg_off + n_sub), true);
                simdgroup_multiply_accumulate(acc[n_sub / 8], x_frag, b_frag, acc[n_sub / 8]);
            }
        }
    }
    
    // Store results (dh and N are aligned, no bounds check needed)
    const uint qid = simd_lid >> 2;
    const uint lane_row = (qid & 4u) + ((simd_lid >> 1) & 3u);
    const uint lane_col_base = ((qid & 2u) << 1) + ((simd_lid & 1u) << 1);
    
    const uint local_d = row_sg_off + lane_row;
    device float* out_row = out_base + local_d * N;
    
    #pragma clang loop unroll(full)
    for (uint n_sub = 0; n_sub < 16; n_sub += 8) {
        auto frag = acc[n_sub / 8].thread_elements();
        uint local_n = col_sg_off + n_sub + lane_col_base;
        out_row[local_n] = frag[0];
        out_row[local_n + 1] = frag[1];
    }
}

// Finalize state: state_new = decay_total * state_old + contribution
// Grid: (ceil(H*dh*N / 256),)
template <typename T>
kernel void ssd_finalize_state(
    device T* state [[ buffer(0) ]],                   // [H, dh, N] in-place
    device const float* contribution [[ buffer(1) ]], // [H, dh, N]
    device const float* cum_log_decay [[ buffer(2) ]], // [H, L+1]
    constant const uint& suffix_len [[ buffer(3) ]],
    constant const uint& num_heads [[ buffer(4) ]],
    constant const uint& head_dim [[ buffer(5) ]],
    constant const uint& state_dim [[ buffer(6) ]],
    constant const size_t* state_strides [[ buffer(7) ]],
    uint tg_pos [[ threadgroup_position_in_grid ]],
    uint tid [[ thread_index_in_threadgroup ]]
) {
    const uint idx = tg_pos * 256 + tid;
    const uint H = num_heads;
    const uint dh = head_dim;
    const uint N = state_dim;
    const uint L = suffix_len;
    const uint total = H * dh * N;
    
    if (idx >= total) return;
    
    // Decode idx to (h, d, n)
    const uint n = idx % N;
    const uint d = (idx / N) % dh;
    const uint h = idx / (N * dh);
    
    // Total decay: exp(cum_log_decay[h, L])
    float decay_total = fast::exp(cum_log_decay[h * (L + 1) + L]);
    
    // Read contribution[h, d, n] from contiguous layout
    float contrib = contribution[h * dh * N + d * N + n];
    
    // Read/write state using strides
    const size_t state_head_stride = state_strides[0];
    const size_t state_dim_stride = state_strides[1];
    const size_t state_inner_stride = state_strides[2];
    size_t state_idx = h * state_head_stride + d * state_dim_stride + n * state_inner_stride;
    
    float old_state = float(state[state_idx]);
    float new_state = decay_total * old_state + contrib;
    state[state_idx] = static_cast<T>(new_state);
}

// Template instantiations

#define INSTANTIATE_SSD_GEMM_STATE_UPDATE(tname, T) \
    template [[host_name("ssd_gemm_state_update_" #tname)]] \
    kernel void ssd_gemm_state_update<T>( \
        device const T*, device const T*, device const float*, device float*, \
        constant const uint&, constant const uint&, constant const uint&, \
        constant const uint&, constant const uint&, constant const size_t*, \
        constant const size_t*, uint3, uint, uint);

INSTANTIATE_SSD_GEMM_STATE_UPDATE(float, float)
INSTANTIATE_SSD_GEMM_STATE_UPDATE(bfloat, bfloat)
INSTANTIATE_SSD_GEMM_STATE_UPDATE(half, half)

#define INSTANTIATE_SSD_FINALIZE_STATE(tname, T) \
    template [[host_name("ssd_finalize_state_" #tname)]] \
    kernel void ssd_finalize_state<T>( \
        device T*, device const float*, device const float*, \
        constant const uint&, constant const uint&, constant const uint&, \
        constant const uint&, constant const size_t*, uint, uint);

INSTANTIATE_SSD_FINALIZE_STATE(float, float)
INSTANTIATE_SSD_FINALIZE_STATE(bfloat, bfloat)
INSTANTIATE_SSD_FINALIZE_STATE(half, half)

#define INSTANTIATE_SSD_GEMM_CB_FUSED(tname, T) \
    template [[host_name("ssd_gemm_cb_fused_" #tname)]] \
    kernel void ssd_gemm_cb_fused<T>( \
        device const T*, device const T*, device const float*, device float*, \
        constant const uint&, constant const uint&, constant const uint&, \
        constant const uint&, constant const size_t&, constant const size_t&, \
        uint3, uint, uint);

INSTANTIATE_SSD_GEMM_CB_FUSED(float, float)
INSTANTIATE_SSD_GEMM_CB_FUSED(bfloat, bfloat)
INSTANTIATE_SSD_GEMM_CB_FUSED(half, half)

#define INSTANTIATE_SSD_GEMM_Y(tname, T) \
    template [[host_name("ssd_gemm_y_" #tname)]] \
    kernel void ssd_gemm_y<T>( \
        device const float*, device const T*, device float*, \
        constant const uint&, constant const uint&, constant const uint&, \
        constant const size_t*, uint3, uint, uint);

INSTANTIATE_SSD_GEMM_Y(float, float)
INSTANTIATE_SSD_GEMM_Y(bfloat, bfloat)
INSTANTIATE_SSD_GEMM_Y(half, half)

// =============================================================================
// Template instantiations
// =============================================================================

#define INSTANTIATE_SSD_GEMM_STATE_C(tname, T) \
    template [[host_name("ssd_gemm_state_c_" #tname)]] \
    kernel void ssd_gemm_state_c<T>( \
        device const T*, device const T*, device float*, \
        constant const uint&, constant const uint&, constant const uint&, \
        constant const uint&, constant const uint&, constant const size_t*, \
        constant const size_t*, uint3, uint, uint);

INSTANTIATE_SSD_GEMM_STATE_C(float, float)
INSTANTIATE_SSD_GEMM_STATE_C(bfloat, bfloat)
INSTANTIATE_SSD_GEMM_STATE_C(half, half)

#define INSTANTIATE_SSD_FUSED_ADD_SKIP_GATE(tname, T) \
    template [[host_name("ssd_fused_add_skip_gate_" #tname)]] \
    kernel void ssd_fused_add_skip_gate<T>( \
        device const float*, device const float*, device const float*, \
        device const T*, device const T*, device const T*, device T*, \
        constant const uint&, constant const uint&, constant const uint&, \
        constant const size_t*, uint, uint);

INSTANTIATE_SSD_FUSED_ADD_SKIP_GATE(float, float)
INSTANTIATE_SSD_FUSED_ADD_SKIP_GATE(bfloat, bfloat)
INSTANTIATE_SSD_FUSED_ADD_SKIP_GATE(half, half)

// Template instantiation for decay pass (kept from original)
#define instantiate_ssd_prefill_flash_pass1(type_name, type)                \
  template [[host_name("ssd_prefill_flash_pass1_decay_" #type_name)]]       \
  kernel void ssd_prefill_flash_pass1_decay<type>(                          \
    device const type* dt_raw [[ buffer(0) ]],                              \
    device float* cum_log_decay [[ buffer(1) ]],                            \
    constant const size_t& suffix_len [[ buffer(2) ]],                      \
    constant const size_t& dt_token_stride [[ buffer(3) ]],                 \
    constant const size_t& dt_head_stride [[ buffer(4) ]],                  \
    uint h_idx [[ threadgroup_position_in_grid ]],                          \
    ushort lid [[ thread_index_in_threadgroup ]]                            \
  );

instantiate_ssd_prefill_flash_pass1(float, float);
instantiate_ssd_prefill_flash_pass1(bfloat, bfloat);
instantiate_ssd_prefill_flash_pass1(half, half);

#undef instantiate_ssd_prefill_flash_pass1
