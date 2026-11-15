#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static const char* METAL_SHADERS =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"inline float erf_approx(float x) {\n"
"    if (x > 5.0f) return 1.0f;\n"
"    if (x < -5.0f) return -1.0f;\n"
"    float a1 =  0.254829592f;\n"
"    float a2 = -0.284496736f;\n"
"    float a3 =  1.421413741f;\n"
"    float a4 = -1.453152027f;\n"
"    float a5 =  1.061405429f;\n"
"    float p  =  0.3275911f;\n"
"    float sign = x < 0.0f ? -1.0f : 1.0f;\n"
"    x = metal::abs(x);\n"
"    float t = 1.0f / (1.0f + p * x);\n"
"    float y = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * metal::exp(-x * x);\n"
"    return sign * y;\n"
"}\n"
"\n"
"// ==================== FP16 KERNELS ====================\n"
"kernel void add_bias_kernel_fp16(\n"
"    device half* c [[buffer(0)]],\n"
"    device const half* bias [[buffer(1)]],\n"
"    constant uint& m [[buffer(2)]],\n"
"    constant uint& n [[buffer(3)]],\n"
"    uint2 id [[thread_position_in_grid]]\n"
") {\n"
"    uint col = id.x;\n"
"    uint row = id.y;\n"
"    if (col < n && row < m) {\n"
"        c[row * n + col] += bias[col];\n"
"    }\n"
"}\n"
"\n"
"kernel void add_pos_embed_indexed_kernel_fp16(\n"
"    device half* h [[buffer(0)]],\n"
"    device const half* pos_table [[buffer(1)]],\n"
"    device const uint* pos_indices [[buffer(2)]],\n"
"    constant uint& hidden_size [[buffer(3)]],\n"
"    constant uint& num_patches [[buffer(4)]],\n"
"    uint2 id [[thread_position_in_grid]]\n"
") {\n"
"    uint hid = id.x;\n"
"    uint patch_idx = id.y;\n"
"    if (hid >= hidden_size || patch_idx >= num_patches) return;\n"
"    uint pos_idx = pos_indices[patch_idx];\n"
"    h[patch_idx * hidden_size + hid] += pos_table[pos_idx * hidden_size + hid];\n"
"}\n"
"\n"
"kernel void gelu_tanh_kernel_fp16(\n"
"    device half* x [[buffer(0)]],\n"
"    constant uint& n [[buffer(1)]],\n"
"    uint id [[thread_position_in_grid]]\n"
") {\n"
"    if (id >= n) return;\n"
"    float v = float(x[id]);\n"
"    float sqrt_2_over_pi = 0.7978845608f;\n"
"    float arg = clamp(sqrt_2_over_pi * (v + 0.044715f * v * v * v), -10.0f, 10.0f);\n"
"    x[id] = half(0.5f * v * (1.0f + metal::precise::tanh(arg)));\n"
"}\n"
"\n"
"kernel void gelu_erf_kernel_fp16(\n"
"    device half* x [[buffer(0)]],\n"
"    constant uint& n [[buffer(1)]],\n"
"    uint id [[thread_position_in_grid]]\n"
") {\n"
"    if (id >= n) return;\n"
"    float v = float(x[id]);\n"
"    x[id] = half(0.5f * v * (1.0f + erf_approx(v * 0.7071067811865475f)));\n"
"}\n"
"\n"
"kernel void transpose_qkv_heads_kernel_fp16(\n"
"    device const half* in [[buffer(0)]],\n"
"    device half* out_q [[buffer(1)]],\n"
"    device half* out_k [[buffer(2)]],\n"
"    device half* out_v [[buffer(3)]],\n"
"    constant uint& total_tokens [[buffer(4)]],\n"
"    constant uint& num_heads [[buffer(5)]],\n"
"    constant uint& head_dim [[buffer(6)]],\n"
"    uint id [[thread_position_in_grid]]\n"
") {\n"
"    uint hidden_size = num_heads * head_dim;\n"
"    uint total = total_tokens * hidden_size;\n"
"    if (id >= total) return;\n"
"    uint t = id / hidden_size;\n"
"    uint rem = id % hidden_size;\n"
"    uint h = rem / head_dim;\n"
"    uint d = rem % head_dim;\n"
"    uint out_idx = h * (total_tokens * head_dim) + t * head_dim + d;\n"
"    uint in_base = t * (3 * hidden_size) + rem;\n"
"    out_q[out_idx] = in[in_base];\n"
"    out_k[out_idx] = in[in_base + hidden_size];\n"
"    out_v[out_idx] = in[in_base + 2 * hidden_size];\n"
"}\n"
"\n"
"kernel void untranspose_heads_kernel_fp16(\n"
"    device const half* in [[buffer(0)]],\n"
"    device half* out [[buffer(1)]],\n"
"    constant uint& total_tokens [[buffer(2)]],\n"
"    constant uint& num_heads [[buffer(3)]],\n"
"    constant uint& head_dim [[buffer(4)]],\n"
"    uint id [[thread_position_in_grid]]\n"
") {\n"
"    uint total = total_tokens * num_heads * head_dim;\n"
"    if (id >= total) return;\n"
"    uint hidden_size = num_heads * head_dim;\n"
"    uint t = id / hidden_size;\n"
"    uint rem = id % hidden_size;\n"
"    uint h = rem / head_dim;\n"
"    uint d = rem % head_dim;\n"
"    uint in_idx = h * (total_tokens * head_dim) + t * head_dim + d;\n"
"    out[id] = in[in_idx];\n"
"}\n"
"\n"
"kernel void add_residual_kernel_fp16(\n"
"    device half* x [[buffer(0)]],\n"
"    device const half* res [[buffer(1)]],\n"
"    constant uint& count [[buffer(2)]],\n"
"    uint id [[thread_position_in_grid]]\n"
") {\n"
"    if (id < count) {\n"
"        x[id] += res[id];\n"
"    }\n"
"}\n"
"\n"
"kernel void add_bias_gelu_tanh_kernel_fp16(\n"
"    device half* c [[buffer(0)]],\n"
"    device const half* bias [[buffer(1)]],\n"
"    constant uint& m [[buffer(2)]],\n"
"    constant uint& n [[buffer(3)]],\n"
"    constant uint& has_bias [[buffer(4)]],\n"
"    uint2 id [[thread_position_in_grid]]\n"
") {\n"
"    uint col = id.x;\n"
"    uint row = id.y;\n"
"    if (col < n && row < m) {\n"
"        uint idx = row * n + col;\n"
"        float v = float(c[idx]) + (has_bias ? float(bias[col]) : 0.0f);\n"
"        float sqrt_2_over_pi = 0.7978845608f;\n"
"        float arg = clamp(sqrt_2_over_pi * (v + 0.044715f * v * v * v), -10.0f, 10.0f);\n"
"        c[idx] = half(0.5f * v * (1.0f + metal::precise::tanh(arg)));\n"
"    }\n"
"}\n"
"\n"
"kernel void add_bias_residual_kernel_fp16(\n"
"    device half* res [[buffer(0)]],\n"
"    device const half* c [[buffer(1)]],\n"
"    device const half* bias [[buffer(2)]],\n"
"    constant uint& m [[buffer(3)]],\n"
"    constant uint& n [[buffer(4)]],\n"
"    constant uint& has_bias [[buffer(5)]],\n"
"    uint2 id [[thread_position_in_grid]]\n"
") {\n"
"    uint col = id.x;\n"
"    uint row = id.y;\n"
"    if (col < n && row < m) {\n"
"        uint idx = row * n + col;\n"
"        half val = c[idx];\n"
"        if (has_bias) {\n"
"            val += bias[col];\n"
"        }\n"
"        res[idx] += val;\n"
"    }\n"
"}\n"
"\n"
"kernel void layer_norm_kernel_fp16(\n"
"    device const half* in [[buffer(0)]],\n"
"    device const half* gamma [[buffer(1)]],\n"
"    device const half* beta [[buffer(2)]],\n"
"    device half* out [[buffer(3)]],\n"
"    constant uint& hidden_size [[buffer(4)]],\n"
"    constant float& eps [[buffer(5)]],\n"
"    constant uint& has_beta [[buffer(6)]],\n"
"    uint token_idx [[threadgroup_position_in_grid]],\n"
"    uint thread_idx [[thread_index_in_threadgroup]],\n"
"    uint threads_per_group [[threads_per_threadgroup]]\n"
") {\n"
"    threadgroup float sum_shared[64];\n"
"    threadgroup float var_shared[64];\n"
"    threadgroup float tg_mean;\n"
"    threadgroup float tg_inv_std;\n"
"\n"
"    uint simd_lane = thread_idx % 32;\n"
"    uint simd_id = thread_idx / 32;\n"
"    uint num_simds = (threads_per_group + 31) / 32;\n"
"\n"
"    device const half* tok_in = in + token_idx * hidden_size;\n"
"    device half* tok_out = out + token_idx * hidden_size;\n"
"\n"
"    float local_sum = 0.0f;\n"
"    for (uint i = thread_idx; i < hidden_size; i += threads_per_group) {\n"
"        local_sum += float(tok_in[i]);\n"
"    }\n"
"    local_sum = simd_sum(local_sum);\n"
"    if (simd_lane == 0) sum_shared[simd_id] = local_sum;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"\n"
"    if (thread_idx == 0) {\n"
"        float total_sum = 0.0f;\n"
"        for (uint s = 0; s < num_simds; s++) {\n"
"            total_sum += sum_shared[s];\n"
"        }\n"
"        tg_mean = total_sum / float(hidden_size);\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"\n"
"    float mean = tg_mean;\n"
"\n"
"    float local_var = 0.0f;\n"
"    for (uint i = thread_idx; i < hidden_size; i += threads_per_group) {\n"
"        float diff = float(tok_in[i]) - mean;\n"
"        local_var += diff * diff;\n"
"    }\n"
"    local_var = simd_sum(local_var);\n"
"    if (simd_lane == 0) var_shared[simd_id] = local_var;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"\n"
"    if (thread_idx == 0) {\n"
"        float total_var = 0.0f;\n"
"        for (uint s = 0; s < num_simds; s++) {\n"
"            total_var += var_shared[s];\n"
"        }\n"
"        tg_inv_std = rsqrt(total_var / float(hidden_size) + eps);\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"\n"
"    float inv_std = tg_inv_std;\n"
"\n"
"    for (uint i = thread_idx; i < hidden_size; i += threads_per_group) {\n"
"        float val = (float(tok_in[i]) - mean) * inv_std * float(gamma[i]);\n"
"        if (has_beta) val += float(beta[i]);\n"
"        tok_out[i] = half(val);\n"
"    }\n"
"}\n"
"\n"
"kernel void window_gather_kernel_fp16(\n"
"    device const half* in [[buffer(0)]],\n"
"    device const uint* indices [[buffer(1)]],\n"
"    device half* out [[buffer(2)]],\n"
"    constant uint& total_tokens [[buffer(3)]],\n"
"    constant uint& hidden_size [[buffer(4)]],\n"
"    uint2 id [[thread_position_in_grid]]\n"
") {\n"
"    uint tok = id.x;\n"
"    uint d = id.y;\n"
"    if (tok >= total_tokens || d >= hidden_size) return;\n"
"    uint src_tok = indices[tok];\n"
"    out[tok * hidden_size + d] = in[src_tok * hidden_size + d];\n"
"}\n"
"\n"
"kernel void window_pool_kernel_fp16(\n"
"    device const half* in [[buffer(0)]],\n"
"    device half* out [[buffer(1)]],\n"
"    constant uint& num_windows [[buffer(2)]],\n"
"    constant uint& hidden_size [[buffer(3)]],\n"
"    constant uint& window_size [[buffer(4)]],\n"
"    uint2 id [[thread_position_in_grid]]\n"
") {\n"
"    uint w = id.x;\n"
"    uint d = id.y;\n"
"    if (w >= num_windows || d >= hidden_size) return;\n"
"    float sum = 0.0f;\n"
"    uint w_offset = w * window_size * hidden_size;\n"
"    for (uint p = 0; p < window_size; p++) {\n"
"        sum += float(in[w_offset + p * hidden_size + d]);\n"
"    }\n"
"    out[w * hidden_size + d] = half(sum / float(window_size));\n"
"}\n"
"\n"
"kernel void top_merger_regroup_kernel_fp16(\n"
"    device const half* in [[buffer(0)]],\n"
"    device half* out [[buffer(1)]],\n"
"    constant uint& in_tok_offset [[buffer(2)]],\n"
"    constant uint& out_tok_offset [[buffer(3)]],\n"
"    constant uint& h [[buffer(4)]],\n"
"    constant uint& w [[buffer(5)]],\n"
"    constant uint& hidden_size [[buffer(6)]],\n"
"    uint2 id [[thread_position_in_grid]]\n"
") {\n"
"    uint oh = id.x;\n"
"    uint ow = id.y;\n"
"    uint h_half = h / 2;\n"
"    uint w_half = w / 2;\n"
"    if (oh >= h_half || ow >= w_half) return;\n"
"    uint out_tok = out_tok_offset + oh * w_half + ow;\n"
"    uint flat_dim = 4 * hidden_size;\n"
"    for (uint r = 0; r < 2; r++) {\n"
"        for (uint c = 0; c < 2; c++) {\n"
"            uint in_y = oh * 2 + r;\n"
"            uint in_x = ow * 2 + c;\n"
"            uint in_tok = in_tok_offset + in_y * w + in_x;\n"
"            uint p_idx = r * 2 + c;\n"
"            for (uint d = 0; d < hidden_size; d++) {\n"
"                out[out_tok * flat_dim + p_idx * hidden_size + d] = in[in_tok * hidden_size + d];\n"
"            }\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"kernel void window_attention_kernel_qkv_fp16(\n"
"    device const half* qkv [[buffer(0)]],\n"
"    device half* out [[buffer(1)]],\n"
"    constant uint& num_windows [[buffer(2)]],\n"
"    constant uint& num_heads [[buffer(3)]],\n"
"    constant uint& head_dim [[buffer(4)]],\n"
"    uint2 tg_id [[threadgroup_position_in_grid]],\n"
"    uint tid [[thread_index_in_threadgroup]]\n"
") {\n"
"    uint h = tg_id.x;\n"
"    uint w = tg_id.y;\n"
"    if (h >= num_heads || w >= num_windows) return;\n"
"    uint hidden_size = num_heads * head_dim;\n"
"    uint qkv_stride = 3 * hidden_size;\n"
"    uint w_tok_base_qkv = w * 4 * qkv_stride + h * head_dim;\n"
"    uint w_tok_base_out = w * 4 * hidden_size + h * head_dim;\n"
"    threadgroup float S[4][4];\n"
"    threadgroup float P[4][4];\n"
"    float local_dot[4][4] = { {0} };\n"
"    for (uint d = tid; d < head_dim; d += 64) {\n"
"        float q_val[4];\n"
"        float k_val[4];\n"
"        for (uint i = 0; i < 4; i++) {\n"
"            q_val[i] = float(qkv[w_tok_base_qkv + i * qkv_stride + d]);\n"
"            k_val[i] = float(qkv[w_tok_base_qkv + i * qkv_stride + hidden_size + d]);\n"
"        }\n"
"        for (uint i = 0; i < 4; i++) {\n"
"            for (uint j = 0; j < 4; j++) {\n"
"                local_dot[i][j] += q_val[i] * k_val[j];\n"
"            }\n"
"        }\n"
"    }\n"
"    for (uint i = 0; i < 4; i++) {\n"
"        for (uint j = 0; j < 4; j++) {\n"
"            local_dot[i][j] = simd_sum(local_dot[i][j]);\n"
"        }\n"
"    }\n"
"    threadgroup float tg_sum[2][4][4];\n"
"    uint simd_id = tid / 32;\n"
"    uint simd_lane = tid % 32;\n"
"    if (simd_lane == 0) {\n"
"        for (uint i = 0; i < 4; i++) {\n"
"            for (uint j = 0; j < 4; j++) {\n"
"                tg_sum[simd_id][i][j] = local_dot[i][j];\n"
"            }\n"
"        }\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (tid == 0) {\n"
"        float alpha = 1.0f / sqrt((float)head_dim);\n"
"        for (uint i = 0; i < 4; i++) {\n"
"            float max_s = -1e30f;\n"
"            for (uint j = 0; j < 4; j++) {\n"
"                float s = (tg_sum[0][i][j] + tg_sum[1][i][j]) * alpha;\n"
"                S[i][j] = s;\n"
"                if (s > max_s) max_s = s;\n"
"            }\n"
"            float sum_exp = 0.0f;\n"
"            for (uint j = 0; j < 4; j++) {\n"
"                float e = metal::exp(S[i][j] - max_s);\n"
"                P[i][j] = e;\n"
"                sum_exp += e;\n"
"            }\n"
"            float inv_sum = 1.0f / sum_exp;\n"
"            for (uint j = 0; j < 4; j++) {\n"
"                P[i][j] *= inv_sum;\n"
"            }\n"
"        }\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    for (uint d = tid; d < head_dim; d += 64) {\n"
"        float v_val[4];\n"
"        for (uint j = 0; j < 4; j++) {\n"
"            v_val[j] = float(qkv[w_tok_base_qkv + j * qkv_stride + 2 * hidden_size + d]);\n"
"        }\n"
"        for (uint i = 0; i < 4; i++) {\n"
"            float o = 0.0f;\n"
"            for (uint j = 0; j < 4; j++) {\n"
"                o += P[i][j] * v_val[j];\n"
"            }\n"
"            out[w_tok_base_out + i * hidden_size + d] = half(o);\n"
"        }\n"
"    }\n"
"}\n"
"\n"
"kernel void batch_softmax_fp16_kernel(\n"
"    device half* s [[buffer(0)]],\n"
"    constant uint& seg_len [[buffer(1)]],\n"
"    constant uint& num_heads [[buffer(2)]],\n"
"    uint3 id [[thread_position_in_grid]],\n"
"    uint tid [[thread_index_in_threadgroup]]\n"
") {\n"
"    uint row = id.y;\n"
"    uint head = id.z;\n"
"    if (row >= seg_len || head >= num_heads) return;\n"
"    uint lane = tid;\n"
"    uint row_offset = (head * seg_len + row) * seg_len;\n"
"    device half* row_ptr = s + row_offset;\n"
"\n"
"    float local_max = -1e30f;\n"
"    for (uint c = lane; c < seg_len; c += 32) {\n"
"        float val = float(row_ptr[c]);\n"
"        if (val > local_max) local_max = val;\n"
"    }\n"
"    float row_max = simd_max(local_max);\n"
"\n"
"    float local_sum = 0.0f;\n"
"    for (uint c = lane; c < seg_len; c += 32) {\n"
"        float val = float(row_ptr[c]);\n"
"        local_sum += metal::exp(val - row_max);\n"
"    }\n"
"    float row_sum = simd_sum(local_sum);\n"
"    float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;\n"
"\n"
"    for (uint c = lane; c < seg_len; c += 32) {\n"
"        float val = float(row_ptr[c]);\n"
"        row_ptr[c] = half(metal::exp(val - row_max) * inv_sum);\n"
"    }\n"
"}\n"
"\n"
"// ==================== LFM3-VL VISION KERNELS ====================\n"
"\n"
"// RMSNorm over one row: x * weight / sqrt(mean(x^2) + eps).\n"
"kernel void rms_norm_kernel_fp16(\n"
"    device const half* in [[buffer(0)]],\n"
"    device const float* weight [[buffer(1)]],\n"
"    device half* out [[buffer(2)]],\n"
"    constant uint& dim [[buffer(3)]],\n"
"    constant float& eps [[buffer(4)]],\n"
"    uint tg [[threadgroup_position_in_grid]],\n"
"    uint tid [[thread_index_in_threadgroup]],\n"
"    uint tpg [[threads_per_threadgroup]]\n"
") {\n"
"    threadgroup float reduce_shared[8];\n"
"    device const half* row_in = in + (ulong)tg * dim;\n"
"    device half* row_out = out + (ulong)tg * dim;\n"
"    float local = 0.0f;\n"
"    for (uint i = tid; i < dim; i += tpg) {\n"
"        float v = float(row_in[i]);\n"
"        local += v * v;\n"
"    }\n"
"    float total = simd_sum(local);\n"
"    uint lane = tid % 32;\n"
"    uint sid = tid / 32;\n"
"    if (lane == 0) reduce_shared[sid] = total;\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    if (tid == 0) {\n"
"        float s = 0.0f;\n"
"        uint nsimd = (tpg + 31) / 32;\n"
"        for (uint i = 0; i < nsimd; i++) s += reduce_shared[i];\n"
"        reduce_shared[0] = metal::rsqrt(s / float(dim) + eps);\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"    float inv = reduce_shared[0];\n"
"    for (uint i = tid; i < dim; i += tpg) {\n"
"        row_out[i] = half(float(row_in[i]) * inv * weight[i]);\n"
"    }\n"
"}\n"
"\n"
"// Per-head RMSNorm + 2D axial RoPE (NeoX half rotation). head_dim must be 64:\n"
"// one simdgroup per (token, head), lane owns dims lane and lane+32.\n"
"kernel void qk_norm_rope_2d_kernel_fp16(\n"
"    device const half* q [[buffer(0)]],\n"
"    device const half* k [[buffer(1)]],\n"
"    device const float* q_norm [[buffer(2)]],\n"
"    device const float* k_norm [[buffer(3)]],\n"
"    device const float* cos_tab [[buffer(4)]],\n"
"    device const float* sin_tab [[buffer(5)]],\n"
"    device half* q_out [[buffer(6)]],\n"
"    device half* k_out [[buffer(7)]],\n"
"    constant uint& num_heads [[buffer(8)]],\n"
"    constant uint& num_kv_heads [[buffer(9)]],\n"
"    constant uint& head_dim [[buffer(10)]],\n"
"    constant float& eps [[buffer(11)]],\n"
"    uint2 tg [[threadgroup_position_in_grid]],\n"
"    uint lane [[thread_index_in_threadgroup]]\n"
") {\n"
"    uint which = tg.x;\n"
"    bool is_q = which < num_heads;\n"
"    uint head = is_q ? which : (which - num_heads);\n"
"    uint row = tg.y;\n"
"    uint dim = num_heads * head_dim;\n"
"    uint kv_dim = num_kv_heads * head_dim;\n"
"    uint stride = is_q ? dim : kv_dim;\n"
"    uint half_dim = head_dim / 2;\n"
"    device const half* src = is_q ? q : k;\n"
"    device half* dst = is_q ? q_out : k_out;\n"
"    device const float* w = is_q ? q_norm : k_norm;\n"
"    ulong base = (ulong)row * stride + (ulong)head * head_dim;\n"
"    uint d0 = lane;\n"
"    uint d1 = lane + half_dim;\n"
"    float x0 = float(src[base + d0]);\n"
"    float x1 = float(src[base + d1]);\n"
"    float ss = simd_sum(x0 * x0 + x1 * x1);\n"
"    float inv = metal::rsqrt(ss / float(head_dim) + eps);\n"
"    float y0 = x0 * inv * w[d0];\n"
"    float y1 = x1 * inv * w[d1];\n"
"    ulong tab = (ulong)row * head_dim;\n"
"    float c0 = cos_tab[tab + d0];\n"
"    float s0 = sin_tab[tab + d0];\n"
"    float c1 = cos_tab[tab + d1];\n"
"    float s1 = sin_tab[tab + d1];\n"
"    dst[base + d0] = half(y0 * c0 - y1 * s0);\n"
"    dst[base + d1] = half(y1 * c1 + y0 * s1);\n"
"}\n"
"\n"
"// Conv gate: grid[(y*gw+x)*dim + d] = proj[t, d] * proj[t, 2*dim + d].\n"
"// Token -> (y, x) uses block-major order with merge size m.\n"
"kernel void lfm3_conv_gate_scatter_kernel_fp16(\n"
"    device const half* proj [[buffer(0)]],\n"
"    device half* grid [[buffer(1)]],\n"
"    constant uint& gh [[buffer(2)]],\n"
"    constant uint& gw [[buffer(3)]],\n"
"    constant uint& m [[buffer(4)]],\n"
"    constant uint& dim [[buffer(5)]],\n"
"    uint2 id [[thread_position_in_grid]]\n"
") {\n"
"    uint d = id.x;\n"
"    uint t = id.y;\n"
"    if (d >= dim) return;\n"
"    uint gh_m = gh / m;\n"
"    uint gw_m = gw / m;\n"
"    uint per_block = m * m;\n"
"    uint block = t / per_block;\n"
"    uint within = t - block * per_block;\n"
"    uint ir = within / m;\n"
"    uint ic = within - ir * m;\n"
"    uint hb = block / gw_m;\n"
"    uint wb = block - hb * gw_m;\n"
"    uint y = hb * m + ir;\n"
"    uint x = wb * m + ic;\n"
"    ulong stride = (ulong)3 * dim;\n"
"    float b = float(proj[(ulong)t * stride + d]);\n"
"    float v = float(proj[(ulong)t * stride + 2 * dim + d]);\n"
"    grid[((ulong)y * gw + x) * dim + d] = half(b * v);\n"
"}\n"
"\n"
"// Depthwise 3x3 convolution, padding 1, channel-last [gh, gw, dim].\n"
"kernel void lfm3_depthwise_conv3x3_kernel_fp16(\n"
"    device const half* grid [[buffer(0)]],\n"
"    device const float* kernel_w [[buffer(1)]],\n"
"    device half* out [[buffer(2)]],\n"
"    constant uint& gh [[buffer(3)]],\n"
"    constant uint& gw [[buffer(4)]],\n"
"    constant uint& dim [[buffer(5)]],\n"
"    uint2 id [[thread_position_in_grid]]\n"
") {\n"
"    uint d = id.x;\n"
"    uint p = id.y;\n"
"    if (d >= dim || p >= gh * gw) return;\n"
"    int y = int(p / gw);\n"
"    int x = int(p - uint(y) * gw);\n"
"    float acc = 0.0f;\n"
"    for (int ky = -1; ky <= 1; ky++) {\n"
"        int sy = y + ky;\n"
"        if (sy < 0 || sy >= int(gh)) continue;\n"
"        for (int kx = -1; kx <= 1; kx++) {\n"
"            int sx = x + kx;\n"
"            if (sx < 0 || sx >= int(gw)) continue;\n"
"            acc += float(grid[((ulong)sy * gw + uint(sx)) * dim + d])\n"
"                 * kernel_w[d * 9 + (ky + 1) * 3 + (kx + 1)];\n"
"        }\n"
"    }\n"
"    out[(ulong)p * dim + d] = half(acc);\n"
"}\n"
"\n"
"// Conv gather gate: gated[t, d] = proj[t, dim + d] * conv[y(t), x(t), d].\n"
"kernel void lfm3_conv_gather_gate_kernel_fp16(\n"
"    device const half* conv_out [[buffer(0)]],\n"
"    device const half* proj [[buffer(1)]],\n"
"    device half* gated [[buffer(2)]],\n"
"    constant uint& gh [[buffer(3)]],\n"
"    constant uint& gw [[buffer(4)]],\n"
"    constant uint& m [[buffer(5)]],\n"
"    constant uint& dim [[buffer(6)]],\n"
"    uint2 id [[thread_position_in_grid]]\n"
") {\n"
"    uint d = id.x;\n"
"    uint t = id.y;\n"
"    if (d >= dim) return;\n"
"    uint gh_m = gh / m;\n"
"    uint gw_m = gw / m;\n"
"    uint per_block = m * m;\n"
"    uint block = t / per_block;\n"
"    uint within = t - block * per_block;\n"
"    uint ir = within / m;\n"
"    uint ic = within - ir * m;\n"
"    uint hb = block / gw_m;\n"
"    uint wb = block - hb * gw_m;\n"
"    uint y = hb * m + ir;\n"
"    uint x = wb * m + ic;\n"
"    ulong stride = (ulong)3 * dim;\n"
"    float c = float(proj[(ulong)t * stride + dim + d]);\n"
"    float o = float(conv_out[((ulong)y * gw + x) * dim + d]);\n"
"    gated[(ulong)t * dim + d] = half(c * o);\n"
"}\n"
"\n"
"// SwiGLU: fused row is [up (w3) | gate (w1)], SiLU on the second half.\n"
"kernel void lfm3_swiglu_kernel_fp16(\n"
"    device const half* fused [[buffer(0)]],\n"
"    device half* gated [[buffer(1)]],\n"
"    constant uint& hidden [[buffer(2)]],\n"
"    uint2 id [[thread_position_in_grid]]\n"
") {\n"
"    uint d = id.x;\n"
"    uint t = id.y;\n"
"    if (d >= hidden) return;\n"
"    device const half* row = fused + (ulong)t * (2 * hidden);\n"
"    float up = float(row[d]);\n"
"    float gate = float(row[hidden + d]);\n"
"    float silu = gate / (1.0f + metal::exp(-gate));\n"
"    gated[(ulong)t * hidden + d] = half(up * silu);\n"
"}\n"
"\n"
"// Projector merge: [n, dim] block-major tokens -> [n/m2, m2*dim] packs.\n"
"kernel void lfm3_merge_pack_kernel_fp16(\n"
"    device const half* in [[buffer(0)]],\n"
"    device half* out [[buffer(1)]],\n"
"    constant uint& dim [[buffer(2)]],\n"
"    constant uint& merge [[buffer(3)]],\n"
"    uint2 id [[thread_position_in_grid]]\n"
") {\n"
"    uint d = id.x;\n"
"    uint o = id.y;\n"
"    if (d >= dim) return;\n"
"    uint m2 = merge * merge;\n"
"    ulong packed_dim = (ulong)m2 * dim;\n"
"    ulong dst = ((ulong)o / m2) * packed_dim + ((ulong)o % m2) * dim + d;\n"
"    out[dst] = in[(ulong)o * dim + d];\n"
"}\n"
"\n"
"// Token-major [n, heads*dim] -> head-major [heads, n, dim].\n"
"// Why: MPS/AMX attention reads Q/K/V per head. In token-major layout every\n"
"// head's 128 bytes are separated by (heads-1)*128 bytes of other heads, so a\n"
"// dense GEMM touches whole cache lines for 1/heads of the data. Head-major\n"
"// makes each per-head matrix contiguous. Same trick applies to any per-head\n"
"// operator (per-head projection, RoPE applied before attention, ...).\n"
"kernel void lfm3_heads_to_major_kernel_fp16(\n"
"    device const half* in [[buffer(0)]],\n"
"    device half* out [[buffer(1)]],\n"
"    constant uint& n [[buffer(2)]],\n"
"    constant uint& heads [[buffer(3)]],\n"
"    constant uint& head_dim [[buffer(4)]],\n"
"    uint2 id [[thread_position_in_grid]]\n"
") {\n"
"    uint d = id.x;\n"
"    uint token_head = id.y;\n"
"    if (d >= head_dim) return;\n"
"    uint token = token_head / heads;\n"
"    uint head = token_head - token * heads;\n"
"    out[((ulong)head * n + token) * head_dim + d] = in[(ulong)token * heads * head_dim + (ulong)head * head_dim + d];\n"
"}\n"
"\n"
"// ==================== Fused flash attention (ported from MLX Steel attention) ====================\n"
"//\n"
"// Source: mlx/backend/metal/kernels/steel/attn/{attn.h,mma.h,loader.h}\n"
"//         and steel/attn/kernels/steel_attention.h (Apple MLX, MIT license).\n"
"//\n"
"// We ported the D=64 f16 instance of MLX's Steel attention kernel:\n"
"//   BQ=32 (query block), BK=32 (key block), BD=64 (head dim), WM=4 warps,\n"
"//   WN=1. Four simdgroups per threadgroup, each owning an 8-row query tile;\n"
"//   online softmax in registers (exp2 domain, M_LOG2E folded into the scale);\n"
"//   same fragment layout and row-reduce shuffles as MLX's BaseMMAFrag.\n"
"//\n"
"// Deliberate simplifications vs the full Steel kernel (this is inference for\n"
"// the vision tower only):\n"
"//   - bidirectional, no causal/window/mask/sinks/sliding-window variants;\n"
"//   - no split-KV kernel (single pass per query block);\n"
"//   - no save_lse / backward pass;\n"
"//   - tail handling is a runtime check instead of the align_Q/align_K\n"
"//     function-constant specializations;\n"
"//   - f16 MMA inputs with f32 accumulation (lfm3_mma_h); MLX uses f32 MMA\n"
"//     for its WN=1 D=64 variant. Both were validated: f16-in/f32-acc is\n"
"//     slightly faster and within the tower's parity budget.\n"
"//\n"
"// For future agents:\n"
"//   * The same porting method applies to other MLX Steel kernels (GEMM,\n"
"//     gather/scatter, other head dims): copy the tile geometry + fragment\n"
"//     helpers from mma.h/loader.h, then specialize constants for the shape\n"
"//     you need. Keep the fragment layout and row-reduce shuffles identical;\n"
"//     they encode the simdgroup_matrix register map.\n"
"//   * To support head_dim 128/192/256, instantiate WN=2 (split the head dim\n"
"//     across two warps) and the corresponding s_xchg score exchange from\n"
"//     steel_attention.h. That is the main reason MLX has WN=2.\n"
"//   * For very long sequences (n > ~32k) a split-KV variant pays off; see\n"
"//     `attention_split` in steel_attention.h (partial O + max/sum combine).\n"
"//   * If a full port is ever wanted (causal LM prefill, sliding window,\n"
"//     masks, sinks), the remaining pieces are mechanical: the mask branches\n"
"//     and the split kernel. Prefer a full upstream-style port over further\n"
"//     hand-tuning of this vision-only instance.\n"
"//\n"
"// Q/K/V are head-major contiguous (see lfm3_heads_to_major_kernel_fp16);\n"
"// O is token-major [n, H*D].\n"
"\n"
"inline metal::float2 lfm3_attn_coord(uint lane) {\n"
"    short qid = short(lane) / 4;\n"
"    short fm = (qid & 4) + ((short(lane) / 2) % 4);\n"
"    short fn = (qid & 2) * 2 + (short(lane) % 2) * 2;\n"
"    return metal::float2(float(fn), float(fm));\n"
"}\n"
"\n"
"inline metal::float2 lfm3_frag_load(const threadgroup half* src, int stride) {\n"
"    return metal::float2(float(src[0]), float(src[stride]));\n"
"}\n"
"\n"
"inline void lfm3_mma(thread float2& d, float2 a, float2 b, float2 c) {\n"
"    simdgroup_matrix<float, 8, 8> amat;\n"
"    simdgroup_matrix<float, 8, 8> bmat;\n"
"    simdgroup_matrix<float, 8, 8> cmat;\n"
"    simdgroup_matrix<float, 8, 8> dmat;\n"
"    amat.thread_elements()[0] = a.x;\n"
"    amat.thread_elements()[1] = a.y;\n"
"    bmat.thread_elements()[0] = b.x;\n"
"    bmat.thread_elements()[1] = b.y;\n"
"    cmat.thread_elements()[0] = c.x;\n"
"    cmat.thread_elements()[1] = c.y;\n"
"    simdgroup_multiply_accumulate(dmat, amat, bmat, cmat);\n"
"    d = metal::float2(dmat.thread_elements()[0], dmat.thread_elements()[1]);\n"
"}\n"
"\n"
"inline void lfm3_mma_h(thread float2& d, float2 a, float2 b, float2 c) {\n"
"    simdgroup_matrix<half, 8, 8> amat;\n"
"    simdgroup_matrix<half, 8, 8> bmat;\n"
"    simdgroup_matrix<float, 8, 8> cmat;\n"
"    simdgroup_matrix<float, 8, 8> dmat;\n"
"    amat.thread_elements()[0] = half(a.x);\n"
"    amat.thread_elements()[1] = half(a.y);\n"
"    bmat.thread_elements()[0] = half(b.x);\n"
"    bmat.thread_elements()[1] = half(b.y);\n"
"    cmat.thread_elements()[0] = c.x;\n"
"    cmat.thread_elements()[1] = c.y;\n"
"    simdgroup_multiply_accumulate(dmat, amat, bmat, cmat);\n"
"    d = metal::float2(dmat.thread_elements()[0], dmat.thread_elements()[1]);\n"
"}\n"
"\n"
"inline float lfm3_row_max(float2 v) {\n"
"    float t = metal::max(v.x, v.y);\n"
"    t = metal::max(t, simd_shuffle_xor(t, ushort(1)));\n"
"    t = metal::max(t, simd_shuffle_xor(t, ushort(8)));\n"
"    return t;\n"
"}\n"
"\n"
"inline float lfm3_row_sum(float2 v) {\n"
"    float t = v.x + v.y;\n"
"    t += simd_shuffle_xor(t, ushort(1));\n"
"    t += simd_shuffle_xor(t, ushort(8));\n"
"    return t;\n"
"}\n"
"\n"
"kernel void lfm3_flash_attention_fp16(\n"
"    device const half* Q [[buffer(0)]],\n"
"    device const half* K [[buffer(1)]],\n"
"    device const half* V [[buffer(2)]],\n"
"    device half* O [[buffer(3)]],\n"
"    constant uint& n [[buffer(4)]],\n"
"    constant uint& num_heads [[buffer(5)]],\n"
"    constant uint& num_kv_heads [[buffer(6)]],\n"
"    constant float& scale_log2e [[buffer(7)]],\n"
"    uint3 tg [[threadgroup_position_in_grid]],\n"
"    uint3 lid3 [[thread_position_in_threadgroup]]\n"
") {\n"
"    const uint lid = lid3.x;\n"
"    const uint lane = lid % 32;\n"
"    const uint warp = lid / 32;\n"
"    constexpr int BQ = 32;\n"
"    constexpr int BK = 32;\n"
"    constexpr int BD = 64;\n"
"    constexpr int LDQ = BD + 8;\n"
"    constexpr int LDK = BK + 8;\n"
"    constexpr int LDV = BD + 8;\n"
"    constexpr int TGP = 128;\n"
"\n"
"    const uint q0 = tg.x * BQ;\n"
"    const uint head = tg.y;\n"
"    const uint groups = num_heads / num_kv_heads;\n"
"    const uint kvh = head / groups;\n"
"    const uint NK = (n + BK - 1) / BK;\n"
"    const uint nq = num_heads * BD;\n"
"\n"
"    threadgroup half Qs[BQ * LDQ];\n"
"    threadgroup half Ks[BD * LDK];\n"
"    threadgroup half Vs[BK * LDV];\n"
"\n"
"    // Cooperative loads (Steel BlockLoaderT mapping: 16 halves per thread).\n"
"    const uint bi = lid / 4;\n"
"    const uint bj = (lid % 4) * 16;\n"
"    const bool q_tail = q0 + BQ > n;\n"
"    const bool k_tail = (n % BK) != 0;\n"
"    device const half* q_base = Q + (ulong)head * n * BD;\n"
"    device const half* k_base = K + (ulong)kvh * n * BD;\n"
"    device const half* v_base = V + (ulong)kvh * n * BD;\n"
"\n"
"    // Q block (16 halves = four half4 vectors per thread).\n"
"    if (q_tail && q0 + bi >= n) {\n"
"        threadgroup half4* dst = (threadgroup half4*)(&Qs[bi * LDQ + bj]);\n"
"        for (int j = 0; j < 4; j++) dst[j] = half4(half(0.0f));\n"
"    } else {\n"
"        const device half4* src = (const device half4*)(q_base + (ulong)(q0 + bi) * BD + bj);\n"
"        threadgroup half4* dst = (threadgroup half4*)(&Qs[bi * LDQ + bj]);\n"
"        for (int j = 0; j < 4; j++) dst[j] = src[j];\n"
"    }\n"
"    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"\n"
"    const float2 coord = lfm3_attn_coord(lane);\n"
"    const int sm = int(coord.y);\n"
"    const int sn = int(coord.x);\n"
"    const int tm = 8 * int(warp);\n"
"    const int Qs_offset = (tm + sm) * LDQ + sn;\n"
"    const int Ks_offset = sm * LDK + sn;\n"
"    const int Vs_offset = sm * LDV + sn;\n"
"\n"
"    // Per-lane Q fragments for the eight head-dim subtiles.\n"
"    float2 qfrag[8];\n"
"    for (int dd = 0; dd < 8; dd++) qfrag[dd] = lfm3_frag_load(&Qs[Qs_offset + dd * 8], 1);\n"
"\n"
"    float2 S[4];\n"
"    float2 Oacc[8];\n"
"    for (int i = 0; i < 4; i++) S[i] = metal::float2(0.0f, 0.0f);\n"
"    for (int i = 0; i < 8; i++) Oacc[i] = metal::float2(0.0f, 0.0f);\n"
"    float max_score = -INFINITY;\n"
"    float sum_score = 0.0f;\n"
"\n"
"    for (uint kb = 0; kb < NK; kb++) {\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        const uint k_tok = kb * BK + bi;\n"
"        if (k_tail && kb == NK - 1 && k_tok >= n) {\n"
"            for (int j = 0; j < 16; j++) {\n"
"                Ks[(bj + j) * LDK + bi] = half(0.0f);\n"
"                Vs[bi * LDV + bj + j] = half(0.0f);\n"
"            }\n"
"        } else {\n"
"            device const half* ksrc = k_base + (ulong)k_tok * BD + bj;\n"
"            device const half* vsrc = v_base + (ulong)k_tok * BD + bj;\n"
"            const device half4* vsrc4 = (const device half4*)vsrc;\n"
"            threadgroup half4* vdst4 = (threadgroup half4*)(&Vs[bi * LDV + bj]);\n"
"            for (int j = 0; j < 16; j++) Ks[(bj + j) * LDK + bi] = ksrc[j];\n"
"            for (int j = 0; j < 4; j++) vdst4[j] = vsrc4[j];\n"
"        }\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"\n"
"        for (int i = 0; i < 4; i++) S[i] = metal::float2(0.0f, 0.0f);\n"
"        for (int dd = 0; dd < 8; dd++) {\n"
"            for (int tk = 0; tk < 4; tk++) {\n"
"                float2 kf = lfm3_frag_load(&Ks[Ks_offset + dd * 8 * LDK + tk * 8], 1);\n"
"                lfm3_mma_h(S[tk], qfrag[dd], kf, S[tk]);\n"
"            }\n"
"        }\n"
"\n"
"        for (int tk = 0; tk < 4; tk++) {\n"
"            S[tk].x *= scale_log2e;\n"
"            S[tk].y *= scale_log2e;\n"
"        }\n"
"        if (k_tail && kb == NK - 1) {\n"
"            for (int tk = 0; tk < 4; tk++) {\n"
"                int col = int(kb * BK) + sn + tk * 8;\n"
"                if (col >= int(n)) S[tk].x = -INFINITY;\n"
"                if (col + 1 >= int(n)) S[tk].y = -INFINITY;\n"
"            }\n"
"        }\n"
"\n"
"        float new_max = max_score;\n"
"        for (int tk = 0; tk < 4; tk++) new_max = metal::max(new_max, lfm3_row_max(S[tk]));\n"
"        for (int tk = 0; tk < 4; tk++) {\n"
"            S[tk].x = fast::exp2(S[tk].x - new_max);\n"
"            S[tk].y = fast::exp2(S[tk].y - new_max);\n"
"        }\n"
"        const float factor = fast::exp2(max_score - new_max);\n"
"        max_score = new_max;\n"
"        float block_sum = 0.0f;\n"
"        for (int tk = 0; tk < 4; tk++) block_sum += lfm3_row_sum(S[tk]);\n"
"        sum_score = sum_score * factor + block_sum;\n"
"        for (int i = 0; i < 8; i++) {\n"
"            Oacc[i].x *= factor;\n"
"            Oacc[i].y *= factor;\n"
"        }\n"
"\n"
"        for (int id = 0; id < 8; id++) {\n"
"            for (int ik = 0; ik < 4; ik++) {\n"
"                float2 vf = lfm3_frag_load(&Vs[Vs_offset + ik * 8 * LDV + id * 8], 1);\n"
"                lfm3_mma_h(Oacc[id], S[ik], vf, Oacc[id]);\n"
"            }\n"
"        }\n"
"    }\n"
"\n"
"    float inv_sum = 1.0f / sum_score;\n"
"    for (int id = 0; id < 8; id++) {\n"
"        Oacc[id].x *= inv_sum;\n"
"        Oacc[id].y *= inv_sum;\n"
"    }\n"
"    if (q0 + tm + sm < n) {\n"
"        for (int id = 0; id < 8; id++) {\n"
"            ulong dst = (ulong)(q0 + tm + sm) * nq + (ulong)head * BD + id * 8 + sn;\n"
"            O[dst] = half(Oacc[id].x);\n"
"            O[dst + 1] = half(Oacc[id].y);\n"
"        }\n"
"    }\n"
"}\n"
"\n"
;

typedef struct {
    int m, n, k;
    BOOL transposeLeft, transposeRight;
    float alpha, beta;
    void* kernel;
} MatmulCacheEntry;

typedef struct {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLCommandBuffer> current_cb;
    id<MTLComputePipelineState> psoAddBiasFp16;
    id<MTLComputePipelineState> psoGeluTanhFp16;
    id<MTLComputePipelineState> psoGeluErfFp16;
    id<MTLComputePipelineState> psoTransposeQkvHeadsFp16;
    id<MTLComputePipelineState> psoUntransposeHeadsFp16;
    id<MTLComputePipelineState> psoAddResidualFp16;
    id<MTLComputePipelineState> psoAddPosEmbedIndexedFp16;
    id<MTLComputePipelineState> psoLayerNormFp16;
    id<MTLComputePipelineState> psoWindowGatherFp16;
    id<MTLComputePipelineState> psoWindowPoolFp16;
    id<MTLComputePipelineState> psoTopMergerRegroupFp16;
    id<MTLComputePipelineState> psoWindowAttentionQkvFp16;
    id<MTLComputePipelineState> psoBatchSoftmaxFp16;
    id<MTLComputePipelineState> psoAddBiasGeluTanhFp16;
    id<MTLComputePipelineState> psoAddBiasResidualFp16;
    id<MTLComputePipelineState> psoRmsNormFp16;
    id<MTLComputePipelineState> psoQkNormRope2dFp16;
    id<MTLComputePipelineState> psoLfm3ConvGateScatterFp16;
    id<MTLComputePipelineState> psoLfm3DepthwiseConv3x3Fp16;
    id<MTLComputePipelineState> psoLfm3ConvGatherGateFp16;
    id<MTLComputePipelineState> psoLfm3SwiGLUFp16;
    id<MTLComputePipelineState> psoLfm3MergePackFp16;
    id<MTLComputePipelineState> psoLfm3HeadsToMajorFp16;
    id<MTLComputePipelineState> psoLfm3FlashAttentionFp16;
    MPSMatrixSoftMax* softmax;
    MatmulCacheEntry cache[64];
    int cache_count;
    int cache_next;

    // Cached persistent scratch buffers (eliminates 161.5 MB allocations per forward pass)
    id<MTLBuffer> scratch_s;
    size_t scratch_s_bytes;
    id<MTLBuffer> scratch_qt;
    size_t scratch_qt_bytes;
    id<MTLBuffer> scratch_kt;
    size_t scratch_kt_bytes;
    id<MTLBuffer> scratch_vt;
    size_t scratch_vt_bytes;
    id<MTLBuffer> scratch_attnt;
    size_t scratch_attnt_bytes;
    id<MTLBuffer> scratch_indices;
    size_t scratch_indices_bytes;

    id<MTLComputeCommandEncoder> current_enc;
} MetalVisionContext;

static inline id<MTLComputeCommandEncoder> get_compute_encoder(MetalVisionContext* ctx) {
    if (ctx && ctx->current_cb != nil) {
        if (ctx->current_enc == nil) {
            ctx->current_enc = [ctx->current_cb computeCommandEncoder];
        }
        return ctx->current_enc;
    }
    return nil;
}

static inline void end_compute_encoder(MetalVisionContext* ctx) {
    if (ctx && ctx->current_enc != nil) {
        [ctx->current_enc endEncoding];
        ctx->current_enc = nil;
    }
}

MetalVisionContext* metal_vision_create_context(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return NULL;
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) return NULL;

        NSError *err = nil;
        NSString *src = [NSString stringWithUTF8String:METAL_SHADERS];
        id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:&err];
        if (!lib) {
            NSLog(@"MetalVision: Failed to compile shaders: %@", err);
            return NULL;
        }

        MetalVisionContext* ctx = (MetalVisionContext*)calloc(1, sizeof(MetalVisionContext));
        ctx->device = device;
        ctx->queue = queue;
        ctx->current_cb = nil;

        ctx->psoAddBiasFp16 = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"add_bias_kernel_fp16"] error:&err];
        ctx->psoGeluTanhFp16 = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gelu_tanh_kernel_fp16"] error:&err];
        ctx->psoGeluErfFp16 = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"gelu_erf_kernel_fp16"] error:&err];
        ctx->psoTransposeQkvHeadsFp16 = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"transpose_qkv_heads_kernel_fp16"] error:&err];
        ctx->psoUntransposeHeadsFp16 = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"untranspose_heads_kernel_fp16"] error:&err];
        ctx->psoAddResidualFp16 = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"add_residual_kernel_fp16"] error:&err];
        ctx->psoAddPosEmbedIndexedFp16 = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"add_pos_embed_indexed_kernel_fp16"] error:&err];
        ctx->psoLayerNormFp16 = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"layer_norm_kernel_fp16"] error:&err];
        ctx->psoWindowGatherFp16 = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"window_gather_kernel_fp16"] error:&err];
        ctx->psoWindowPoolFp16 = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"window_pool_kernel_fp16"] error:&err];
        ctx->psoTopMergerRegroupFp16 = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"top_merger_regroup_kernel_fp16"] error:&err];
        ctx->psoWindowAttentionQkvFp16 = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"window_attention_kernel_qkv_fp16"] error:&err];
        ctx->psoBatchSoftmaxFp16 = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"batch_softmax_fp16_kernel"] error:&err];
        ctx->psoAddBiasGeluTanhFp16 = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"add_bias_gelu_tanh_kernel_fp16"] error:&err];
        ctx->psoAddBiasResidualFp16 = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"add_bias_residual_kernel_fp16"] error:&err];
        ctx->psoRmsNormFp16 = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"rms_norm_kernel_fp16"] error:&err];
        ctx->psoQkNormRope2dFp16 = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"qk_norm_rope_2d_kernel_fp16"] error:&err];
        ctx->psoLfm3ConvGateScatterFp16 = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"lfm3_conv_gate_scatter_kernel_fp16"] error:&err];
        ctx->psoLfm3DepthwiseConv3x3Fp16 = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"lfm3_depthwise_conv3x3_kernel_fp16"] error:&err];
        ctx->psoLfm3ConvGatherGateFp16 = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"lfm3_conv_gather_gate_kernel_fp16"] error:&err];
        ctx->psoLfm3SwiGLUFp16 = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"lfm3_swiglu_kernel_fp16"] error:&err];
        ctx->psoLfm3MergePackFp16 = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"lfm3_merge_pack_kernel_fp16"] error:&err];
        ctx->psoLfm3HeadsToMajorFp16 = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"lfm3_heads_to_major_kernel_fp16"] error:&err];
        ctx->psoLfm3FlashAttentionFp16 = [device newComputePipelineStateWithFunction:[lib newFunctionWithName:@"lfm3_flash_attention_fp16"] error:&err];
        if (err) {
            NSLog(@"MetalVision: Failed to compile shaders: %@", err);
        }

        ctx->softmax = [[MPSMatrixSoftMax alloc] initWithDevice:device];

        return ctx;
    }
}

void metal_vision_destroy_context(MetalVisionContext* ctx) {
    if (!ctx) return;
    for (int i = 0; i < ctx->cache_count; i++) {
        if (ctx->cache[i].kernel) {
            CFRelease(ctx->cache[i].kernel);
            ctx->cache[i].kernel = NULL;
        }
    }
    ctx->scratch_s = nil;
    ctx->scratch_qt = nil;
    ctx->scratch_kt = nil;
    ctx->scratch_vt = nil;
    ctx->scratch_attnt = nil;
    ctx->scratch_indices = nil;
    ctx->softmax = nil;
    ctx->psoAddBiasFp16 = nil;
    ctx->psoGeluTanhFp16 = nil;
    ctx->psoGeluErfFp16 = nil;
    ctx->psoTransposeQkvHeadsFp16 = nil;
    ctx->psoUntransposeHeadsFp16 = nil;
    ctx->psoAddResidualFp16 = nil;
    ctx->psoAddPosEmbedIndexedFp16 = nil;
    ctx->psoLayerNormFp16 = nil;
    ctx->psoWindowGatherFp16 = nil;
    ctx->psoWindowPoolFp16 = nil;
    ctx->psoTopMergerRegroupFp16 = nil;
    ctx->psoWindowAttentionQkvFp16 = nil;
    ctx->psoBatchSoftmaxFp16 = nil;
    ctx->psoAddBiasGeluTanhFp16 = nil;
    ctx->psoAddBiasResidualFp16 = nil;
    ctx->psoRmsNormFp16 = nil;
    ctx->psoQkNormRope2dFp16 = nil;
    ctx->psoLfm3ConvGateScatterFp16 = nil;
    ctx->psoLfm3DepthwiseConv3x3Fp16 = nil;
    ctx->psoLfm3ConvGatherGateFp16 = nil;
    ctx->psoLfm3SwiGLUFp16 = nil;
    ctx->psoLfm3MergePackFp16 = nil;
    ctx->psoLfm3HeadsToMajorFp16 = nil;
    ctx->psoLfm3FlashAttentionFp16 = nil;

    ctx->current_enc = nil;
    ctx->current_cb = nil;
    ctx->queue = nil;
    ctx->device = nil;
    free(ctx);
}

void metal_vision_begin_pass(MetalVisionContext* ctx) {
    if (ctx && ctx->current_cb == nil) {
        ctx->current_cb = [ctx->queue commandBuffer];
        ctx->current_enc = nil;
    }
}

void metal_vision_end_pass(MetalVisionContext* ctx) {
    if (ctx) {
        end_compute_encoder(ctx);
        if (ctx->current_cb != nil) {
            [ctx->current_cb commit];
            [ctx->current_cb waitUntilCompleted];
            ctx->current_cb = nil;
        }
    }
}

void* metal_vision_alloc_buffer(MetalVisionContext* ctx, size_t bytes, const void* initial_data) {
    @autoreleasepool {
        id<MTLBuffer> buf = nil;
        if (initial_data) {
            buf = [ctx->device newBufferWithBytes:initial_data length:bytes options:MTLResourceStorageModeShared];
        } else {
            buf = [ctx->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        }
        return (__bridge_retained void*)buf;
    }
}

void metal_vision_free_buffer(void* buf_ptr) {
    if (!buf_ptr) return;
    @autoreleasepool {
        id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)buf_ptr;
        buf = nil;
    }
}

void metal_vision_read_buffer(void* buf_ptr, void* dst, size_t bytes) {
    if (!buf_ptr || !dst) return;
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)buf_ptr;
    memcpy(dst, [buf contents], bytes);
}

void metal_vision_write_buffer(void* buf_ptr, const void* src, size_t bytes) {
    if (!buf_ptr || !src) return;
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)buf_ptr;
    memcpy([buf contents], src, bytes);
}

static MPSMatrixMultiplication* get_or_create_matmul(
    MetalVisionContext* ctx,
    int m, int n, int k,
    BOOL transposeLeft, BOOL transposeRight,
    float alpha, float beta
) {
    for (int i = 0; i < ctx->cache_count; i++) {
        if (ctx->cache[i].m == m &&
            ctx->cache[i].n == n &&
            ctx->cache[i].k == k &&
            ctx->cache[i].transposeLeft == transposeLeft &&
            ctx->cache[i].transposeRight == transposeRight &&
            fabsf(ctx->cache[i].alpha - alpha) < 1e-6f &&
            fabsf(ctx->cache[i].beta - beta) < 1e-6f) {
            return (__bridge MPSMatrixMultiplication*)ctx->cache[i].kernel;
        }
    }
    MPSMatrixMultiplication *kernel = [[MPSMatrixMultiplication alloc]
        initWithDevice:ctx->device
        transposeLeft:transposeLeft
        transposeRight:transposeRight
        resultRows:m
        resultColumns:n
        interiorColumns:k
        alpha:alpha
        beta:beta
    ];
    int idx;
    if (ctx->cache_count < 64) {
        idx = ctx->cache_count++;
    } else {
        idx = ctx->cache_next % 64;
        ctx->cache_next++;
        if (ctx->cache[idx].kernel) {
            CFRelease(ctx->cache[idx].kernel);
            ctx->cache[idx].kernel = NULL;
        }
    }
    ctx->cache[idx].m = m;
    ctx->cache[idx].n = n;
    ctx->cache[idx].k = k;
    ctx->cache[idx].transposeLeft = transposeLeft;
    ctx->cache[idx].transposeRight = transposeRight;
    ctx->cache[idx].alpha = alpha;
    ctx->cache[idx].beta = beta;
    ctx->cache[idx].kernel = (__bridge_retained void*)kernel;
    return kernel;
}

// ==================== FP16 C API ====================

void metal_vision_gemm_fp16(
    MetalVisionContext* ctx,
    void* a_buf, int m, int k,
    void* b_buf, int n,
    void* bias_buf,
    void* c_buf
) {
    @autoreleasepool {
        id<MTLBuffer> bufA = (__bridge id<MTLBuffer>)a_buf;
        id<MTLBuffer> bufB = (__bridge id<MTLBuffer>)b_buf;
        id<MTLBuffer> bufC = (__bridge id<MTLBuffer>)c_buf;
        id<MTLBuffer> bufBias = bias_buf ? (__bridge id<MTLBuffer>)bias_buf : nil;

        MPSMatrixDescriptor *descA = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:k rowBytes:k * sizeof(uint16_t) dataType:MPSDataTypeFloat16];
        MPSMatrixDescriptor *descB = [MPSMatrixDescriptor matrixDescriptorWithRows:n columns:k rowBytes:k * sizeof(uint16_t) dataType:MPSDataTypeFloat16];
        MPSMatrixDescriptor *descC = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:n rowBytes:n * sizeof(uint16_t) dataType:MPSDataTypeFloat16];

        MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descA];
        MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:descB];
        MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:descC];

        MPSMatrixMultiplication *kernel = get_or_create_matmul(ctx, m, n, k, NO, YES, 1.0f, 0.0f);
        kernel.batchSize = 1;
        kernel.batchStart = 0;

        id<MTLCommandBuffer> cb = ctx->current_cb;
        BOOL manage_cb = NO;
        if (cb == nil) {
            cb = [ctx->queue commandBuffer];
            manage_cb = YES;
        }

        end_compute_encoder(ctx);
        [kernel encodeToCommandBuffer:cb leftMatrix:matA rightMatrix:matB resultMatrix:matC];

        if (bufBias) {
            id<MTLComputeCommandEncoder> enc = get_compute_encoder(ctx);
            BOOL local_enc = NO;
            if (!enc) {
                enc = [cb computeCommandEncoder];
                local_enc = YES;
            }
            [enc setComputePipelineState:ctx->psoAddBiasFp16];
            [enc setBuffer:bufC offset:0 atIndex:0];
            [enc setBuffer:bufBias offset:0 atIndex:1];
            uint m_val = (uint)m;
            uint n_val = (uint)n;
            [enc setBytes:&m_val length:sizeof(uint) atIndex:2];
            [enc setBytes:&n_val length:sizeof(uint) atIndex:3];
            [enc dispatchThreads:MTLSizeMake(n, m, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            if (local_enc) {
                [enc endEncoding];
            }
        }

        if (manage_cb) {
            [cb commit];
            [cb waitUntilCompleted];
        }
    }
}

void metal_vision_fc1_fp16(
    MetalVisionContext* ctx,
    void* in_buf, int m, int k,
    void* weight_buf, int n,
    void* bias_buf,
    void* out_buf
) {
    @autoreleasepool {
        id<MTLBuffer> bufA = (__bridge id<MTLBuffer>)in_buf;
        id<MTLBuffer> bufB = (__bridge id<MTLBuffer>)weight_buf;
        id<MTLBuffer> bufC = (__bridge id<MTLBuffer>)out_buf;
        id<MTLBuffer> bufBias = bias_buf ? (__bridge id<MTLBuffer>)bias_buf : nil;

        MPSMatrixDescriptor *descA = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:k rowBytes:k * sizeof(uint16_t) dataType:MPSDataTypeFloat16];
        MPSMatrixDescriptor *descB = [MPSMatrixDescriptor matrixDescriptorWithRows:n columns:k rowBytes:k * sizeof(uint16_t) dataType:MPSDataTypeFloat16];
        MPSMatrixDescriptor *descC = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:n rowBytes:n * sizeof(uint16_t) dataType:MPSDataTypeFloat16];

        MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descA];
        MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:descB];
        MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:descC];

        MPSMatrixMultiplication *kernel = get_or_create_matmul(ctx, m, n, k, NO, YES, 1.0f, 0.0f);
        kernel.batchSize = 1;
        kernel.batchStart = 0;

        id<MTLCommandBuffer> cb = ctx->current_cb;
        BOOL manage_cb = NO;
        if (cb == nil) {
            cb = [ctx->queue commandBuffer];
            manage_cb = YES;
        }

        end_compute_encoder(ctx);
        [kernel encodeToCommandBuffer:cb leftMatrix:matA rightMatrix:matB resultMatrix:matC];

        id<MTLComputeCommandEncoder> enc = get_compute_encoder(ctx);
        BOOL local_enc = NO;
        if (!enc) {
            enc = [cb computeCommandEncoder];
            local_enc = YES;
        }
        [enc setComputePipelineState:ctx->psoAddBiasGeluTanhFp16];
        [enc setBuffer:bufC offset:0 atIndex:0];
        if (bufBias) {
            [enc setBuffer:bufBias offset:0 atIndex:1];
        } else {
            [enc setBuffer:bufC offset:0 atIndex:1];
        }
        uint m_val = (uint)m;
        uint n_val = (uint)n;
        uint has_bias = bufBias ? 1 : 0;
        [enc setBytes:&m_val length:sizeof(uint) atIndex:2];
        [enc setBytes:&n_val length:sizeof(uint) atIndex:3];
        [enc setBytes:&has_bias length:sizeof(uint) atIndex:4];
        [enc dispatchThreads:MTLSizeMake(n, m, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        if (local_enc) {
            [enc endEncoding];
        }

        if (manage_cb) {
            [cb commit];
            [cb waitUntilCompleted];
        }
    }
}

void metal_vision_gemm_add_residual_fp16(
    MetalVisionContext* ctx,
    void* in_buf, int m, int k,
    void* weight_buf, int n,
    void* bias_buf,
    void* c_buf,
    void* res_buf
) {
    @autoreleasepool {
        id<MTLBuffer> bufA = (__bridge id<MTLBuffer>)in_buf;
        id<MTLBuffer> bufB = (__bridge id<MTLBuffer>)weight_buf;
        id<MTLBuffer> bufC = (__bridge id<MTLBuffer>)c_buf;
        id<MTLBuffer> bufRes = (__bridge id<MTLBuffer>)res_buf;
        id<MTLBuffer> bufBias = bias_buf ? (__bridge id<MTLBuffer>)bias_buf : nil;

        MPSMatrixDescriptor *descA = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:k rowBytes:k * sizeof(uint16_t) dataType:MPSDataTypeFloat16];
        MPSMatrixDescriptor *descB = [MPSMatrixDescriptor matrixDescriptorWithRows:n columns:k rowBytes:k * sizeof(uint16_t) dataType:MPSDataTypeFloat16];
        MPSMatrixDescriptor *descC = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:n rowBytes:n * sizeof(uint16_t) dataType:MPSDataTypeFloat16];

        MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descA];
        MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:descB];
        MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:descC];

        MPSMatrixMultiplication *kernel = get_or_create_matmul(ctx, m, n, k, NO, YES, 1.0f, 0.0f);
        kernel.batchSize = 1;
        kernel.batchStart = 0;

        id<MTLCommandBuffer> cb = ctx->current_cb;
        BOOL manage_cb = NO;
        if (cb == nil) {
            cb = [ctx->queue commandBuffer];
            manage_cb = YES;
        }

        end_compute_encoder(ctx);
        [kernel encodeToCommandBuffer:cb leftMatrix:matA rightMatrix:matB resultMatrix:matC];

        id<MTLComputeCommandEncoder> enc = get_compute_encoder(ctx);
        BOOL local_enc = NO;
        if (!enc) {
            enc = [cb computeCommandEncoder];
            local_enc = YES;
        }
        [enc setComputePipelineState:ctx->psoAddBiasResidualFp16];
        [enc setBuffer:bufRes offset:0 atIndex:0];
        [enc setBuffer:bufC offset:0 atIndex:1];
        if (bufBias) {
            [enc setBuffer:bufBias offset:0 atIndex:2];
        } else {
            [enc setBuffer:bufC offset:0 atIndex:2];
        }
        uint m_val = (uint)m;
        uint n_val = (uint)n;
        uint has_bias = bufBias ? 1 : 0;
        [enc setBytes:&m_val length:sizeof(uint) atIndex:3];
        [enc setBytes:&n_val length:sizeof(uint) atIndex:4];
        [enc setBytes:&has_bias length:sizeof(uint) atIndex:5];
        [enc dispatchThreads:MTLSizeMake(n, m, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        if (local_enc) {
            [enc endEncoding];
        }

        if (manage_cb) {
            [cb commit];
            [cb waitUntilCompleted];
        }
    }
}

void metal_vision_layer_norm_fp16(
    MetalVisionContext* ctx,
    void* in_buf,
    void* gamma_buf,
    void* beta_buf,
    void* out_buf,
    int num_tokens,
    int hidden_size,
    float eps
) {
    @autoreleasepool {
        id<MTLBuffer> bIn = (__bridge id<MTLBuffer>)in_buf;
        id<MTLBuffer> bGamma = (__bridge id<MTLBuffer>)gamma_buf;
        id<MTLBuffer> bBeta = beta_buf ? (__bridge id<MTLBuffer>)beta_buf : bGamma;
        id<MTLBuffer> bOut = (__bridge id<MTLBuffer>)out_buf;

        id<MTLCommandBuffer> cb = ctx->current_cb;
        BOOL manage_cb = NO;
        if (cb == nil) {
            cb = [ctx->queue commandBuffer];
            manage_cb = YES;
        }

        id<MTLComputeCommandEncoder> enc = get_compute_encoder(ctx);
        BOOL local_enc = NO;
        if (!enc) {
            enc = [cb computeCommandEncoder];
            local_enc = YES;
        }
        [enc setComputePipelineState:ctx->psoLayerNormFp16];
        [enc setBuffer:bIn offset:0 atIndex:0];
        [enc setBuffer:bGamma offset:0 atIndex:1];
        [enc setBuffer:bBeta offset:0 atIndex:2];
        [enc setBuffer:bOut offset:0 atIndex:3];
        uint h_val = (uint)hidden_size;
        uint has_beta = beta_buf ? 1 : 0;
        [enc setBytes:&h_val length:sizeof(uint) atIndex:4];
        [enc setBytes:&eps length:sizeof(float) atIndex:5];
        [enc setBytes:&has_beta length:sizeof(uint) atIndex:6];

        NSUInteger threads_per_group = 128;
        if (hidden_size > 2048) threads_per_group = 256;
        [enc dispatchThreadgroups:MTLSizeMake(num_tokens, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads_per_group, 1, 1)];
        if (local_enc) {
            [enc endEncoding];
        }

        if (manage_cb) {
            [cb commit];
            [cb waitUntilCompleted];
        }
    }
}

void metal_vision_gelu_tanh_fp16(MetalVisionContext* ctx, void* x_buf, int num_elements) {
    @autoreleasepool {
        id<MTLBuffer> bX = (__bridge id<MTLBuffer>)x_buf;
        id<MTLCommandBuffer> cb = ctx->current_cb;
        BOOL manage_cb = NO;
        if (cb == nil) {
            cb = [ctx->queue commandBuffer];
            manage_cb = YES;
        }

        id<MTLComputeCommandEncoder> enc = get_compute_encoder(ctx);
        BOOL local_enc = NO;
        if (!enc) {
            enc = [cb computeCommandEncoder];
            local_enc = YES;
        }
        [enc setComputePipelineState:ctx->psoGeluTanhFp16];
        [enc setBuffer:bX offset:0 atIndex:0];
        uint count = (uint)num_elements;
        [enc setBytes:&count length:sizeof(uint) atIndex:1];
        [enc dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        if (local_enc) {
            [enc endEncoding];
        }

        if (manage_cb) {
            [cb commit];
            [cb waitUntilCompleted];
        }
    }
}

void metal_vision_gelu_erf_fp16(MetalVisionContext* ctx, void* x_buf, int num_elements) {
    @autoreleasepool {
        id<MTLBuffer> bX = (__bridge id<MTLBuffer>)x_buf;
        id<MTLCommandBuffer> cb = ctx->current_cb;
        BOOL manage_cb = NO;
        if (cb == nil) {
            cb = [ctx->queue commandBuffer];
            manage_cb = YES;
        }

        id<MTLComputeCommandEncoder> enc = get_compute_encoder(ctx);
        BOOL local_enc = NO;
        if (!enc) {
            enc = [cb computeCommandEncoder];
            local_enc = YES;
        }
        [enc setComputePipelineState:ctx->psoGeluErfFp16];
        [enc setBuffer:bX offset:0 atIndex:0];
        uint count = (uint)num_elements;
        [enc setBytes:&count length:sizeof(uint) atIndex:1];
        [enc dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        if (local_enc) {
            [enc endEncoding];
        }

        if (manage_cb) {
            [cb commit];
            [cb waitUntilCompleted];
        }
    }
}

void metal_vision_add_residual_fp16(MetalVisionContext* ctx, void* x_buf, void* res_buf, int count) {
    @autoreleasepool {
        id<MTLBuffer> bX = (__bridge id<MTLBuffer>)x_buf;
        id<MTLBuffer> bRes = (__bridge id<MTLBuffer>)res_buf;
        id<MTLCommandBuffer> cb = ctx->current_cb;
        BOOL manage_cb = NO;
        if (cb == nil) {
            cb = [ctx->queue commandBuffer];
            manage_cb = YES;
        }

        id<MTLComputeCommandEncoder> enc = get_compute_encoder(ctx);
        BOOL local_enc = NO;
        if (!enc) {
            enc = [cb computeCommandEncoder];
            local_enc = YES;
        }
        [enc setComputePipelineState:ctx->psoAddResidualFp16];
        [enc setBuffer:bX offset:0 atIndex:0];
        [enc setBuffer:bRes offset:0 atIndex:1];
        uint c = (uint)count;
        [enc setBytes:&c length:sizeof(uint) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(c, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        if (local_enc) {
            [enc endEncoding];
        }

        if (manage_cb) {
            [cb commit];
            [cb waitUntilCompleted];
        }
    }
}

void metal_vision_add_pos_embed_indexed_fp16(
    MetalVisionContext* ctx,
    void* h_buf,
    void* pos_table_buf,
    void* pos_indices_buf,
    int num_patches,
    int hidden_size
) {
    @autoreleasepool {
        id<MTLBuffer> bH = (__bridge id<MTLBuffer>)h_buf;
        id<MTLBuffer> bTable = (__bridge id<MTLBuffer>)pos_table_buf;
        id<MTLBuffer> bIndices = (__bridge id<MTLBuffer>)pos_indices_buf;

        id<MTLCommandBuffer> cb = ctx->current_cb;
        BOOL manage_cb = NO;
        if (cb == nil) {
            cb = [ctx->queue commandBuffer];
            manage_cb = YES;
        }

        id<MTLComputeCommandEncoder> enc = get_compute_encoder(ctx);
        BOOL local_enc = NO;
        if (!enc) {
            enc = [cb computeCommandEncoder];
            local_enc = YES;
        }
        [enc setComputePipelineState:ctx->psoAddPosEmbedIndexedFp16];
        [enc setBuffer:bH offset:0 atIndex:0];
        [enc setBuffer:bTable offset:0 atIndex:1];
        [enc setBuffer:bIndices offset:0 atIndex:2];
        uint hs = (uint)hidden_size;
        uint np = (uint)num_patches;
        [enc setBytes:&hs length:sizeof(uint) atIndex:3];
        [enc setBytes:&np length:sizeof(uint) atIndex:4];

        NSUInteger w = hidden_size;
        NSUInteger h = num_patches;
        NSUInteger tw = 256;
        if (tw > w) tw = w;
        [enc dispatchThreads:MTLSizeMake(w, h, 1) threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
        if (local_enc) {
            [enc endEncoding];
        }

        if (manage_cb) {
            [cb commit];
            [cb waitUntilCompleted];
        }
    }
}

// =========================================================================================
// CRITICAL ARCHITECTURAL CONTRACT FOR FUTURE DEVELOPERS & AGENTS:
// DO NOT RASHLY "OPTIMIZE" OR REPLACE THIS FUNCTION WITH A PURE GPU COMPUTE SHADER FLASHATTENTION!
//
// [Apple M3 Empirical Benchmark (8-core GPU, 24 GB Unified Memory)]:
//   - Hardware AMX + Parallel GPU Softmax (THIS IMPLEMENTATION):
//       vision_tower = 2.070s | prepare_inputs = 2.209s | TTFT = 5.822s (Peak Decode: 25.8 t/s)
//   - Pure GPU Shader Loop FlashAttention (fused_segmented_flash_attention_kernel_qkv_fp16):
//       vision_tower = 3.650s | prepare_inputs = 3.883s | TTFT = 8.192s (43% SLOWER, regression +1.6s!)
//
// [Physical Hardware Root Cause]:
// 1. In MiniCPM-V-4.7, Vision Tower attention consists of dense bidirectional self-attention
//    on 1024x1024 matrices per segment across 16 heads (head_dim=72). Across 27 ViT layers and
//    9-10 dynamic image slices, this represents ~36.4 GFLOPS of dense FP16 matrix compute.
// 2. On Apple Silicon (M-series), MPSMatrixMultiplication dispatches directly to the dedicated
//    hardware systolic coprocessor (Apple Matrix Coprocessor, AMX), running at tens of TFLOPS
//    asynchronously alongside the GPU.
// 3. In contrast, on the 8 GPU cores of an M3 chip, a compute shader looping 1024 iterations
//    with simd_sum reductions completely saturates the GPU shader ALUs and becomes severely
//    compute-bound.
// 4. Therefore, the optimal architecture on Apple Silicon is:
//    - Pass 1: Single-pass GPU kernel (transpose_qkv_heads_kernel_fp16) to reorder QKV heads
//    - Pass 2: Hardware AMX for Q_s @ K_s^T (1024x1024 FP16 GEMM)
//    - Pass 3: Parallel GPU compute kernel (batch_softmax_fp16_kernel) across all 16 heads
//    - Pass 4: Hardware AMX for S @ V_s (1024x72 FP16 GEMM)
//    - Pass 5: Single-pass GPU kernel (untranspose_heads_kernel_fp16) to pack back to [N, H*D].
// =========================================================================================
void metal_vision_attention_qkv_fp16(
    MetalVisionContext* ctx,
    void* qkv_buf,
    const int* seg_lens,
    int num_segs,
    int total_tokens,
    int num_heads,
    int head_dim,
    void* out_buf
) {
    @autoreleasepool {
        if (!ctx || num_segs <= 0 || total_tokens <= 0) return;
        int max_seg_len = 0;
        for (int i = 0; i < num_segs; i++) {
            if (seg_lens[i] > max_seg_len) {
                max_seg_len = seg_lens[i];
            }
        }
        if (max_seg_len <= 0) {
            max_seg_len = total_tokens;
        }

        id<MTLBuffer> bQKV = (__bridge id<MTLBuffer>)qkv_buf;
        id<MTLBuffer> bOut = (__bridge id<MTLBuffer>)out_buf;

        uint N = (uint)total_tokens;
        uint H = (uint)num_heads;
        uint D = (uint)head_dim;

        size_t req_s = (size_t)H * (size_t)max_seg_len * (size_t)max_seg_len * sizeof(uint16_t);
        size_t req_tok = (size_t)H * (size_t)N * (size_t)D * sizeof(uint16_t);
        if (!ctx->scratch_s || ctx->scratch_s_bytes < req_s) {
            ctx->scratch_s = [ctx->device newBufferWithLength:req_s options:MTLResourceStorageModeShared];
            ctx->scratch_s_bytes = req_s;
        }
        if (!ctx->scratch_qt || ctx->scratch_qt_bytes < req_tok) {
            ctx->scratch_qt = [ctx->device newBufferWithLength:req_tok options:MTLResourceStorageModeShared];
            ctx->scratch_qt_bytes = req_tok;
        }
        if (!ctx->scratch_kt || ctx->scratch_kt_bytes < req_tok) {
            ctx->scratch_kt = [ctx->device newBufferWithLength:req_tok options:MTLResourceStorageModeShared];
            ctx->scratch_kt_bytes = req_tok;
        }
        if (!ctx->scratch_vt || ctx->scratch_vt_bytes < req_tok) {
            ctx->scratch_vt = [ctx->device newBufferWithLength:req_tok options:MTLResourceStorageModeShared];
            ctx->scratch_vt_bytes = req_tok;
        }
        if (!ctx->scratch_attnt || ctx->scratch_attnt_bytes < req_tok) {
            ctx->scratch_attnt = [ctx->device newBufferWithLength:req_tok options:MTLResourceStorageModeShared];
            ctx->scratch_attnt_bytes = req_tok;
        }

        id<MTLBuffer> bS = ctx->scratch_s;
        id<MTLBuffer> bQt = ctx->scratch_qt;
        id<MTLBuffer> bKt = ctx->scratch_kt;
        id<MTLBuffer> bVt = ctx->scratch_vt;
        id<MTLBuffer> bAttnT = ctx->scratch_attnt;

        id<MTLCommandBuffer> cb = ctx->current_cb;
        BOOL manage_cb = NO;
        if (cb == nil) {
            cb = [ctx->queue commandBuffer];
            manage_cb = YES;
        }

        // 1. Single kernel dispatch: Transpose Q, K, V simultaneously from [N, 3 * H * D] to [H, N, D]
        uint total_elements = N * H * D;
        id<MTLComputeCommandEncoder> enc = get_compute_encoder(ctx);
        BOOL local_enc = NO;
        if (!enc) {
            enc = [cb computeCommandEncoder];
            local_enc = YES;
        }
        [enc setComputePipelineState:ctx->psoTransposeQkvHeadsFp16];
        [enc setBuffer:bQKV offset:0 atIndex:0];
        [enc setBuffer:bQt offset:0 atIndex:1];
        [enc setBuffer:bKt offset:0 atIndex:2];
        [enc setBuffer:bVt offset:0 atIndex:3];
        [enc setBytes:&N length:sizeof(uint) atIndex:4];
        [enc setBytes:&H length:sizeof(uint) atIndex:5];
        [enc setBytes:&D length:sizeof(uint) atIndex:6];
        [enc dispatchThreads:MTLSizeMake(total_elements, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        if (local_enc) {
            [enc endEncoding];
        } else {
            end_compute_encoder(ctx);
        }

        // 2. Loop over segments: Batched across ALL H heads on Apple AMX hardware coprocessor
        float alpha = 1.0f / sqrtf((float)D);
        int current_start = 0;
        int last_seg_len = -1;
        MPSMatrixMultiplication *kernel_qk = nil;
        MPSMatrixMultiplication *kernel_sv = nil;
        MPSMatrixDescriptor *descS = nil;
        MPSMatrixDescriptor *descND = nil;

        for (int s = 0; s < num_segs; s++) {
            int seg_len = seg_lens[s];
            if (seg_len <= 0) continue;

            if (seg_len != last_seg_len) {
                last_seg_len = seg_len;
                kernel_qk = get_or_create_matmul(ctx, seg_len, seg_len, D, NO, YES, alpha, 0.0f);
                kernel_sv = get_or_create_matmul(ctx, seg_len, D, seg_len, NO, NO, 1.0f, 0.0f);
                kernel_qk.batchSize = 1;
                kernel_qk.batchStart = 0;
                kernel_sv.batchSize = 1;
                kernel_sv.batchStart = 0;

                descS = [MPSMatrixDescriptor
                    matrixDescriptorWithRows:seg_len
                    columns:seg_len
                    rowBytes:seg_len * sizeof(uint16_t)
                    dataType:MPSDataTypeFloat16
                ];

                descND = [MPSMatrixDescriptor
                    matrixDescriptorWithRows:seg_len
                    columns:D
                    rowBytes:D * sizeof(uint16_t)
                    dataType:MPSDataTypeFloat16
                ];
            }

            // Phase A: All H heads QK back-to-back on AMX
            for (uint h = 0; h < H; h++) {
                NSUInteger seg_s_offset = (NSUInteger)h * (NSUInteger)seg_len * (NSUInteger)seg_len * sizeof(uint16_t);
                NSUInteger tok_offset = ((NSUInteger)h * (NSUInteger)N + (NSUInteger)current_start) * (NSUInteger)D * sizeof(uint16_t);

                MPSMatrix *matS = [[MPSMatrix alloc] initWithBuffer:bS offset:seg_s_offset descriptor:descS];
                MPSMatrix *matQ_s = [[MPSMatrix alloc] initWithBuffer:bQt offset:tok_offset descriptor:descND];
                MPSMatrix *matK_s = [[MPSMatrix alloc] initWithBuffer:bKt offset:tok_offset descriptor:descND];

                [kernel_qk encodeToCommandBuffer:cb leftMatrix:matQ_s rightMatrix:matK_s resultMatrix:matS];
            }

            // Phase B: Single compute dispatch normalizes all H heads simultaneously on GPU
            id<MTLComputeCommandEncoder> sm_enc = [cb computeCommandEncoder];
            [sm_enc setComputePipelineState:ctx->psoBatchSoftmaxFp16];
            [sm_enc setBuffer:bS offset:0 atIndex:0];
            [sm_enc setBytes:&seg_len length:sizeof(uint) atIndex:1];
            [sm_enc setBytes:&H length:sizeof(uint) atIndex:2];
            [sm_enc dispatchThreads:MTLSizeMake(32, seg_len, H) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [sm_enc endEncoding];

            // Phase C: All H heads SV back-to-back on AMX
            for (uint h = 0; h < H; h++) {
                NSUInteger seg_s_offset = (NSUInteger)h * (NSUInteger)seg_len * (NSUInteger)seg_len * sizeof(uint16_t);
                NSUInteger tok_offset = ((NSUInteger)h * (NSUInteger)N + (NSUInteger)current_start) * (NSUInteger)D * sizeof(uint16_t);

                MPSMatrix *matS = [[MPSMatrix alloc] initWithBuffer:bS offset:seg_s_offset descriptor:descS];
                MPSMatrix *matV_s = [[MPSMatrix alloc] initWithBuffer:bVt offset:tok_offset descriptor:descND];
                MPSMatrix *matAttn_s = [[MPSMatrix alloc] initWithBuffer:bAttnT offset:tok_offset descriptor:descND];

                [kernel_sv encodeToCommandBuffer:cb leftMatrix:matS rightMatrix:matV_s resultMatrix:matAttn_s];
            }

            current_start += seg_len;
        }

        // 3. Untranspose attended result from [H, N, D] back to [N, H, D]
        enc = get_compute_encoder(ctx);
        local_enc = NO;
        if (!enc) {
            enc = [cb computeCommandEncoder];
            local_enc = YES;
        }
        [enc setComputePipelineState:ctx->psoUntransposeHeadsFp16];
        [enc setBuffer:bAttnT offset:0 atIndex:0]; [enc setBuffer:bOut offset:0 atIndex:1];
        [enc setBytes:&N length:sizeof(uint) atIndex:2]; [enc setBytes:&H length:sizeof(uint) atIndex:3]; [enc setBytes:&D length:sizeof(uint) atIndex:4];
        [enc dispatchThreads:MTLSizeMake(total_elements, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        if (local_enc) {
            [enc endEncoding];
        }

        if (manage_cb) {
            [cb commit];
            [cb waitUntilCompleted];
        }
    }
}

void metal_vision_window_attention_qkv_fp16(
    MetalVisionContext* ctx,
    void* qkv_buf,
    void* out_buf,
    int num_windows,
    int num_heads,
    int head_dim
) {
    @autoreleasepool {
        id<MTLBuffer> bQKV = (__bridge id<MTLBuffer>)qkv_buf;
        id<MTLBuffer> bOut = (__bridge id<MTLBuffer>)out_buf;

        id<MTLCommandBuffer> cb = ctx->current_cb;
        BOOL manage_cb = NO;
        if (cb == nil) {
            cb = [ctx->queue commandBuffer];
            manage_cb = YES;
        }

        id<MTLComputeCommandEncoder> enc = get_compute_encoder(ctx);
        BOOL local_enc = NO;
        if (!enc) {
            enc = [cb computeCommandEncoder];
            local_enc = YES;
        }
        [enc setComputePipelineState:ctx->psoWindowAttentionQkvFp16];
        [enc setBuffer:bQKV offset:0 atIndex:0];
        [enc setBuffer:bOut offset:0 atIndex:1];
        uint nw = (uint)num_windows;
        uint nh = (uint)num_heads;
        uint hd = (uint)head_dim;
        [enc setBytes:&nw length:sizeof(uint) atIndex:2];
        [enc setBytes:&nh length:sizeof(uint) atIndex:3];
        [enc setBytes:&hd length:sizeof(uint) atIndex:4];

        [enc dispatchThreadgroups:MTLSizeMake(nh, nw, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        if (local_enc) {
            [enc endEncoding];
        }

        if (manage_cb) {
            [cb commit];
            [cb waitUntilCompleted];
        }
    }
}

void metal_vision_window_gather_fp16(
    MetalVisionContext* ctx,
    void* in_buf,
    const uint32_t* indices,
    void* out_buf,
    int total_tokens,
    int hidden_size
) {
    @autoreleasepool {
        id<MTLBuffer> bIn = (__bridge id<MTLBuffer>)in_buf;
        id<MTLBuffer> bOut = (__bridge id<MTLBuffer>)out_buf;

        size_t req_bytes = total_tokens * sizeof(uint32_t);
        if (!ctx->scratch_indices || ctx->scratch_indices_bytes < req_bytes) {
            ctx->scratch_indices = [ctx->device newBufferWithLength:req_bytes options:MTLResourceStorageModeShared];
            ctx->scratch_indices_bytes = req_bytes;
        }
        memcpy([ctx->scratch_indices contents], indices, req_bytes);
        id<MTLBuffer> bIndices = ctx->scratch_indices;

        id<MTLCommandBuffer> cb = ctx->current_cb;
        BOOL manage_cb = NO;
        if (cb == nil) {
            cb = [ctx->queue commandBuffer];
            manage_cb = YES;
        }

        id<MTLComputeCommandEncoder> enc = get_compute_encoder(ctx);
        BOOL local_enc = NO;
        if (!enc) {
            enc = [cb computeCommandEncoder];
            local_enc = YES;
        }
        [enc setComputePipelineState:ctx->psoWindowGatherFp16];
        [enc setBuffer:bIn offset:0 atIndex:0];
        [enc setBuffer:bIndices offset:0 atIndex:1];
        [enc setBuffer:bOut offset:0 atIndex:2];
        uint n = (uint)total_tokens;
        uint h = (uint)hidden_size;
        [enc setBytes:&n length:sizeof(uint) atIndex:3];
        [enc setBytes:&h length:sizeof(uint) atIndex:4];
        [enc dispatchThreads:MTLSizeMake(n, h, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        if (local_enc) {
            [enc endEncoding];
        }

        if (manage_cb) {
            [cb commit];
            [cb waitUntilCompleted];
        }
    }
}

void metal_vision_window_pool_fp16(
    MetalVisionContext* ctx,
    void* in_buf,
    void* out_buf,
    int num_windows,
    int hidden_size,
    int window_size
) {
    @autoreleasepool {
        id<MTLBuffer> bIn = (__bridge id<MTLBuffer>)in_buf;
        id<MTLBuffer> bOut = (__bridge id<MTLBuffer>)out_buf;

        id<MTLCommandBuffer> cb = ctx->current_cb;
        BOOL manage_cb = NO;
        if (cb == nil) {
            cb = [ctx->queue commandBuffer];
            manage_cb = YES;
        }

        id<MTLComputeCommandEncoder> enc = get_compute_encoder(ctx);
        BOOL local_enc = NO;
        if (!enc) {
            enc = [cb computeCommandEncoder];
            local_enc = YES;
        }
        [enc setComputePipelineState:ctx->psoWindowPoolFp16];
        [enc setBuffer:bIn offset:0 atIndex:0];
        [enc setBuffer:bOut offset:0 atIndex:1];
        uint nw = (uint)num_windows;
        uint h = (uint)hidden_size;
        uint ws = (uint)window_size;
        [enc setBytes:&nw length:sizeof(uint) atIndex:2];
        [enc setBytes:&h length:sizeof(uint) atIndex:3];
        [enc setBytes:&ws length:sizeof(uint) atIndex:4];
        [enc dispatchThreads:MTLSizeMake(nw, h, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        if (local_enc) {
            [enc endEncoding];
        }

        if (manage_cb) {
            [cb commit];
            [cb waitUntilCompleted];
        }
    }
}

void metal_vision_top_merger_regroup_fp16(
    MetalVisionContext* ctx,
    void* in_buf,
    void* out_buf,
    int in_tok_offset,
    int out_tok_offset,
    int h,
    int w,
    int hidden_size
) {
    @autoreleasepool {
        id<MTLBuffer> bIn = (__bridge id<MTLBuffer>)in_buf;
        id<MTLBuffer> bOut = (__bridge id<MTLBuffer>)out_buf;

        id<MTLCommandBuffer> cb = ctx->current_cb;
        BOOL manage_cb = NO;
        if (cb == nil) {
            cb = [ctx->queue commandBuffer];
            manage_cb = YES;
        }

        id<MTLComputeCommandEncoder> enc = get_compute_encoder(ctx);
        BOOL local_enc = NO;
        if (!enc) {
            enc = [cb computeCommandEncoder];
            local_enc = YES;
        }
        [enc setComputePipelineState:ctx->psoTopMergerRegroupFp16];
        [enc setBuffer:bIn offset:0 atIndex:0];
        [enc setBuffer:bOut offset:0 atIndex:1];
        uint in_off = (uint)in_tok_offset;
        uint out_off = (uint)out_tok_offset;
        uint h_val = (uint)h;
        uint w_val = (uint)w;
        uint hid = (uint)hidden_size;
        [enc setBytes:&in_off length:sizeof(uint) atIndex:2];
        [enc setBytes:&out_off length:sizeof(uint) atIndex:3];
        [enc setBytes:&h_val length:sizeof(uint) atIndex:4];
        [enc setBytes:&w_val length:sizeof(uint) atIndex:5];
        [enc setBytes:&hid length:sizeof(uint) atIndex:6];
        [enc dispatchThreads:MTLSizeMake(h / 2, w / 2, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
        if (local_enc) {
            [enc endEncoding];
        }

        if (manage_cb) {
            [cb commit];
            [cb waitUntilCompleted];
        }
    }
}

// ==================== LFM3-VL C API ====================

void metal_vision_rms_norm_fp16(
    MetalVisionContext* ctx,
    void* in_buf,
    void* weight_buf,
    void* out_buf,
    int num_tokens,
    int dim,
    float eps
) {
    @autoreleasepool {
        id<MTLBuffer> bIn = (__bridge id<MTLBuffer>)in_buf;
        id<MTLBuffer> bW = (__bridge id<MTLBuffer>)weight_buf;
        id<MTLBuffer> bOut = (__bridge id<MTLBuffer>)out_buf;

        id<MTLCommandBuffer> cb = ctx->current_cb;
        BOOL manage_cb = NO;
        if (cb == nil) {
            cb = [ctx->queue commandBuffer];
            manage_cb = YES;
        }

        id<MTLComputeCommandEncoder> enc = get_compute_encoder(ctx);
        BOOL local_enc = NO;
        if (!enc) {
            enc = [cb computeCommandEncoder];
            local_enc = YES;
        }
        [enc setComputePipelineState:ctx->psoRmsNormFp16];
        [enc setBuffer:bIn offset:0 atIndex:0];
        [enc setBuffer:bW offset:0 atIndex:1];
        [enc setBuffer:bOut offset:0 atIndex:2];
        uint d_val = (uint)dim;
        [enc setBytes:&d_val length:sizeof(uint) atIndex:3];
        [enc setBytes:&eps length:sizeof(float) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(num_tokens, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        if (local_enc) {
            [enc endEncoding];
        }

        if (manage_cb) {
            [cb commit];
            [cb waitUntilCompleted];
        }
    }
}

void metal_vision_lfm3_qk_norm_rope_fp16(
    MetalVisionContext* ctx,
    void* q_buf,
    void* k_buf,
    void* q_norm_buf,
    void* k_norm_buf,
    void* cos_buf,
    void* sin_buf,
    int num_tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float eps
) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = ctx->current_cb;
        BOOL manage_cb = NO;
        if (cb == nil) {
            cb = [ctx->queue commandBuffer];
            manage_cb = YES;
        }

        id<MTLComputeCommandEncoder> enc = get_compute_encoder(ctx);
        BOOL local_enc = NO;
        if (!enc) {
            enc = [cb computeCommandEncoder];
            local_enc = YES;
        }
        [enc setComputePipelineState:ctx->psoQkNormRope2dFp16];
        [enc setBuffer:(__bridge id<MTLBuffer>)q_buf offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)k_buf offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)q_norm_buf offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)k_norm_buf offset:0 atIndex:3];
        [enc setBuffer:(__bridge id<MTLBuffer>)cos_buf offset:0 atIndex:4];
        [enc setBuffer:(__bridge id<MTLBuffer>)sin_buf offset:0 atIndex:5];
        [enc setBuffer:(__bridge id<MTLBuffer>)q_buf offset:0 atIndex:6];
        [enc setBuffer:(__bridge id<MTLBuffer>)k_buf offset:0 atIndex:7];
        uint nh = (uint)num_heads;
        uint nkv = (uint)num_kv_heads;
        uint hd = (uint)head_dim;
        [enc setBytes:&nh length:sizeof(uint) atIndex:8];
        [enc setBytes:&nkv length:sizeof(uint) atIndex:9];
        [enc setBytes:&hd length:sizeof(uint) atIndex:10];
        [enc setBytes:&eps length:sizeof(float) atIndex:11];
        [enc dispatchThreadgroups:MTLSizeMake(num_heads + num_kv_heads, num_tokens, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        if (local_enc) {
            [enc endEncoding];
        }

        if (manage_cb) {
            [cb commit];
            [cb waitUntilCompleted];
        }
    }
}

void metal_vision_lfm3_conv_gate_scatter_fp16(
    MetalVisionContext* ctx,
    void* proj_buf,
    void* grid_buf,
    int num_tokens,
    int grid_h,
    int grid_w,
    int merge,
    int dim
) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = ctx->current_cb;
        BOOL manage_cb = NO;
        if (cb == nil) {
            cb = [ctx->queue commandBuffer];
            manage_cb = YES;
        }
        id<MTLComputeCommandEncoder> enc = get_compute_encoder(ctx);
        BOOL local_enc = NO;
        if (!enc) {
            enc = [cb computeCommandEncoder];
            local_enc = YES;
        }
        [enc setComputePipelineState:ctx->psoLfm3ConvGateScatterFp16];
        [enc setBuffer:(__bridge id<MTLBuffer>)proj_buf offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)grid_buf offset:0 atIndex:1];
        uint gh = (uint)grid_h;
        uint gw = (uint)grid_w;
        uint m = (uint)merge;
        uint d_val = (uint)dim;
        [enc setBytes:&gh length:sizeof(uint) atIndex:2];
        [enc setBytes:&gw length:sizeof(uint) atIndex:3];
        [enc setBytes:&m length:sizeof(uint) atIndex:4];
        [enc setBytes:&d_val length:sizeof(uint) atIndex:5];
        [enc dispatchThreads:MTLSizeMake(dim, num_tokens, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        if (local_enc) {
            [enc endEncoding];
        }
        if (manage_cb) {
            [cb commit];
            [cb waitUntilCompleted];
        }
    }
}

void metal_vision_lfm3_depthwise_conv3x3_fp16(
    MetalVisionContext* ctx,
    void* grid_buf,
    void* kernel_buf,
    void* out_buf,
    int grid_h,
    int grid_w,
    int dim
) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = ctx->current_cb;
        BOOL manage_cb = NO;
        if (cb == nil) {
            cb = [ctx->queue commandBuffer];
            manage_cb = YES;
        }
        id<MTLComputeCommandEncoder> enc = get_compute_encoder(ctx);
        BOOL local_enc = NO;
        if (!enc) {
            enc = [cb computeCommandEncoder];
            local_enc = YES;
        }
        [enc setComputePipelineState:ctx->psoLfm3DepthwiseConv3x3Fp16];
        [enc setBuffer:(__bridge id<MTLBuffer>)grid_buf offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)kernel_buf offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)out_buf offset:0 atIndex:2];
        uint gh = (uint)grid_h;
        uint gw = (uint)grid_w;
        uint d_val = (uint)dim;
        [enc setBytes:&gh length:sizeof(uint) atIndex:3];
        [enc setBytes:&gw length:sizeof(uint) atIndex:4];
        [enc setBytes:&d_val length:sizeof(uint) atIndex:5];
        [enc dispatchThreads:MTLSizeMake(dim, grid_h * grid_w, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        if (local_enc) {
            [enc endEncoding];
        }
        if (manage_cb) {
            [cb commit];
            [cb waitUntilCompleted];
        }
    }
}

void metal_vision_lfm3_conv_gather_gate_fp16(
    MetalVisionContext* ctx,
    void* conv_buf,
    void* proj_buf,
    void* gated_buf,
    int num_tokens,
    int grid_h,
    int grid_w,
    int merge,
    int dim
) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = ctx->current_cb;
        BOOL manage_cb = NO;
        if (cb == nil) {
            cb = [ctx->queue commandBuffer];
            manage_cb = YES;
        }
        id<MTLComputeCommandEncoder> enc = get_compute_encoder(ctx);
        BOOL local_enc = NO;
        if (!enc) {
            enc = [cb computeCommandEncoder];
            local_enc = YES;
        }
        [enc setComputePipelineState:ctx->psoLfm3ConvGatherGateFp16];
        [enc setBuffer:(__bridge id<MTLBuffer>)conv_buf offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)proj_buf offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)gated_buf offset:0 atIndex:2];
        uint gh = (uint)grid_h;
        uint gw = (uint)grid_w;
        uint m = (uint)merge;
        uint d_val = (uint)dim;
        [enc setBytes:&gh length:sizeof(uint) atIndex:3];
        [enc setBytes:&gw length:sizeof(uint) atIndex:4];
        [enc setBytes:&m length:sizeof(uint) atIndex:5];
        [enc setBytes:&d_val length:sizeof(uint) atIndex:6];
        [enc dispatchThreads:MTLSizeMake(dim, num_tokens, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        if (local_enc) {
            [enc endEncoding];
        }
        if (manage_cb) {
            [cb commit];
            [cb waitUntilCompleted];
        }
    }
}

void metal_vision_lfm3_swiglu_fp16(
    MetalVisionContext* ctx,
    void* fused_buf,
    void* gated_buf,
    int num_tokens,
    int hidden
) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = ctx->current_cb;
        BOOL manage_cb = NO;
        if (cb == nil) {
            cb = [ctx->queue commandBuffer];
            manage_cb = YES;
        }
        id<MTLComputeCommandEncoder> enc = get_compute_encoder(ctx);
        BOOL local_enc = NO;
        if (!enc) {
            enc = [cb computeCommandEncoder];
            local_enc = YES;
        }
        [enc setComputePipelineState:ctx->psoLfm3SwiGLUFp16];
        [enc setBuffer:(__bridge id<MTLBuffer>)fused_buf offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)gated_buf offset:0 atIndex:1];
        uint h_val = (uint)hidden;
        [enc setBytes:&h_val length:sizeof(uint) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(hidden, num_tokens, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        if (local_enc) {
            [enc endEncoding];
        }
        if (manage_cb) {
            [cb commit];
            [cb waitUntilCompleted];
        }
    }
}

void metal_vision_lfm3_merge_pack_fp16(
    MetalVisionContext* ctx,
    void* in_buf,
    void* out_buf,
    int num_tokens,
    int dim,
    int merge
) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = ctx->current_cb;
        BOOL manage_cb = NO;
        if (cb == nil) {
            cb = [ctx->queue commandBuffer];
            manage_cb = YES;
        }
        id<MTLComputeCommandEncoder> enc = get_compute_encoder(ctx);
        BOOL local_enc = NO;
        if (!enc) {
            enc = [cb computeCommandEncoder];
            local_enc = YES;
        }
        [enc setComputePipelineState:ctx->psoLfm3MergePackFp16];
        [enc setBuffer:(__bridge id<MTLBuffer>)in_buf offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)out_buf offset:0 atIndex:1];
        uint d_val = (uint)dim;
        uint m_val = (uint)merge;
        [enc setBytes:&d_val length:sizeof(uint) atIndex:2];
        [enc setBytes:&m_val length:sizeof(uint) atIndex:3];
        [enc dispatchThreads:MTLSizeMake(dim, num_tokens, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        if (local_enc) {
            [enc endEncoding];
        }
        if (manage_cb) {
            [cb commit];
            [cb waitUntilCompleted];
        }
    }
}

void metal_vision_lfm3_heads_to_major_fp16(
    MetalVisionContext* ctx,
    void* in_buf,
    void* out_buf,
    int num_tokens,
    int heads,
    int head_dim
) {
    @autoreleasepool {
        id<MTLCommandBuffer> cb = ctx->current_cb;
        BOOL manage_cb = NO;
        if (cb == nil) {
            cb = [ctx->queue commandBuffer];
            manage_cb = YES;
        }
        id<MTLComputeCommandEncoder> enc = get_compute_encoder(ctx);
        BOOL local_enc = NO;
        if (!enc) {
            enc = [cb computeCommandEncoder];
            local_enc = YES;
        }
        [enc setComputePipelineState:ctx->psoLfm3HeadsToMajorFp16];
        [enc setBuffer:(__bridge id<MTLBuffer>)in_buf offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)out_buf offset:0 atIndex:1];
        uint n = (uint)num_tokens;
        uint h = (uint)heads;
        uint d = (uint)head_dim;
        [enc setBytes:&n length:sizeof(uint) atIndex:2];
        [enc setBytes:&h length:sizeof(uint) atIndex:3];
        [enc setBytes:&d length:sizeof(uint) atIndex:4];
        [enc dispatchThreads:MTLSizeMake(head_dim, num_tokens * heads, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        if (local_enc) {
            [enc endEncoding];
        }
        if (manage_cb) {
            [cb commit];
            [cb waitUntilCompleted];
        }
    }
}

void metal_vision_lfm3_flash_attention_fp16(
    MetalVisionContext* ctx,
    void* q_buf,
    void* k_buf,
    void* v_buf,
    void* out_buf,
    int num_tokens,
    int num_heads,
    int num_kv_heads
) {
    @autoreleasepool {
        if (!ctx || num_tokens <= 0 || num_heads <= 0 || num_kv_heads <= 0) return;
        id<MTLCommandBuffer> cb = ctx->current_cb;
        BOOL manage_cb = NO;
        if (cb == nil) {
            cb = [ctx->queue commandBuffer];
            manage_cb = YES;
        }
        id<MTLComputeCommandEncoder> enc = get_compute_encoder(ctx);
        BOOL local_enc = NO;
        if (!enc) {
            enc = [cb computeCommandEncoder];
            local_enc = YES;
        }
        [enc setComputePipelineState:ctx->psoLfm3FlashAttentionFp16];
        [enc setBuffer:(__bridge id<MTLBuffer>)q_buf offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)k_buf offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)v_buf offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)out_buf offset:0 atIndex:3];
        uint n = (uint)num_tokens;
        uint nh = (uint)num_heads;
        uint nkv = (uint)num_kv_heads;
        float scale_log2e = (1.0f / sqrtf(64.0f)) * 1.4426950408889634f;
        [enc setBytes:&n length:sizeof(uint) atIndex:4];
        [enc setBytes:&nh length:sizeof(uint) atIndex:5];
        [enc setBytes:&nkv length:sizeof(uint) atIndex:6];
        [enc setBytes:&scale_log2e length:sizeof(float) atIndex:7];
        NSUInteger q_blocks = (NSUInteger)((num_tokens + 31) / 32);
        [enc dispatchThreadgroups:MTLSizeMake(q_blocks, (NSUInteger)num_heads, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        if (local_enc) {
            [enc endEncoding];
        }
        if (manage_cb) {
            [cb commit];
            [cb waitUntilCompleted];
        }
    }
}
