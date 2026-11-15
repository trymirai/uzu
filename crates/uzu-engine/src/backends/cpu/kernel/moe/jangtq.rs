use half::{bf16, f16};
use num_traits::Float;
use uzu_engine_macros::kernel;

use crate::data_type::DataTypeRepr;

// CPU references for the JANGTQ routed-expert decode kernels (metal/kernel/moe/jangtq.metal).
// These mirror the MLX/Swift reference op-for-op; see the Metal file for the parity contract.
// Deviations from the reference reduction (recorded): simd_sum device reductions are
// sequential lane-order sums here, and exp uses libm instead of Metal fast::exp.

/// Odd-cubic codebook level: level = u * (alpha + beta * u^2), u = code - (2^bits-1)/2,
/// with the LSB-first bitstream extraction of JANGHDecodeQDot.source.
#[inline]
fn jangtq_level(
    row_words: &[u32],
    column: usize,
    bits: u32,
    alpha: f32,
    beta: f32,
) -> f32 {
    let code = if bits == 3 {
        // The 48-bit chunk window is anchored at the 16-coefficient chunk base (as in the
        // Metal kernel), keeping the second word access in bounds for the final chunk.
        let chunk_base = column & !15;
        let qbase = (chunk_base * 3) >> 5;
        let shift = (chunk_base * 3) & 31;
        let qw64 = ((row_words[qbase] as u64) | ((row_words[qbase + 1] as u64) << 32)) >> shift;
        ((qw64 >> ((column - chunk_base) * 3)) & 0x7) as u32
    } else {
        let bit = column * bits as usize;
        let word = bit >> 5;
        let shift = bit & 31;
        let mut unpacked = row_words[word] >> shift;
        if shift + bits as usize > 32 {
            unpacked |= row_words[word + 1] << (32 - shift);
        }
        unpacked & ((1u32 << bits) - 1)
    };
    let half_range = (((1u32 << bits) - 1) as f32) * 0.5;
    let u = (code as f32) - half_range;
    u * beta.mul_add(u * u, alpha)
}

/// F32 hadamard32 per 32-vector, replicating the staged simd butterfly exactly
/// (each stage is a simultaneous partner exchange, hence the stage snapshot).
#[inline]
fn hadamard32_f32(values: &mut [f32; 32]) {
    for stage in [1u32, 2, 4, 8, 16] {
        let snapshot = *values;
        for (lane, value) in values.iter_mut().enumerate() {
            let other = snapshot[lane ^ stage as usize];
            *value = if lane as u32 & stage != 0 {
                other - *value
            } else {
                other + *value
            };
        }
    }
    for value in values.iter_mut() {
        *value *= 0.176_776_69_f32;
    }
}

#[kernel(JangtqHadamard32)]
pub fn jangtq_hadamard32(
    x: *const bf16,
    out: *mut f32,
    row_length: u32,
    rows: u32,
) {
    let row_length_u = row_length as usize;
    debug_assert!(row_length_u % 32 == 0);
    for row_idx in 0..rows as usize {
        for block_idx in 0..(row_length_u / 32) {
            let mut values = [0.0f32; 32];
            for (lane, value) in values.iter_mut().enumerate() {
                // SAFETY: row_idx < rows and block_idx * 32 + lane < row_length by construction.
                *value = unsafe { *x.add(row_idx * row_length_u + block_idx * 32 + lane) }.to_f32();
            }
            hadamard32_f32(&mut values);
            for (lane, value) in values.iter().enumerate() {
                // SAFETY: same bounds as the loads above.
                unsafe { *out.add(row_idx * row_length_u + block_idx * 32 + lane) = *value };
            }
        }
    }
}

#[kernel(JangtqSortIndices)]
pub fn jangtq_sort_indices(
    expert_ids: *const u32,
    sorted_indices: *mut u32,
    expert_offsets: *mut u32,
    expert_counts: *mut u32,
    num_dispatches: u32,
) {
    let n = num_dispatches as usize;
    let mut expert_dispatches: [Vec<u32>; 256] = std::array::from_fn(|_| Vec::new());
    for i in 0..n {
        let exp = unsafe { *expert_ids.add(i) } as usize;
        if exp < 256 {
            expert_dispatches[exp].push(i as u32);
        }
    }
    let mut dest = 0;
    for (exp, dispatches) in expert_dispatches.iter().enumerate() {
        unsafe {
            *expert_offsets.add(exp) = dest as u32;
            *expert_counts.add(exp) = dispatches.len() as u32;
        }
        for &orig_dispatch in dispatches {
            unsafe {
                *sorted_indices.add(dest) = ((orig_dispatch + 1) << 8) | (exp as u32);
            }
            dest += 1;
        }
    }
}

#[kernel(JangtqFusedGateUp)]
#[variants(BITS_G, 2)]
#[variants(BITS_U, 2)]
pub fn jangtq_fused_gate_up<const BITS_G: u32, const BITS_U: u32>(
    x: *const f32,
    packed_g: *const u32,
    scales_g: *const f16,
    packed_u: *const u32,
    scales_u: *const f16,
    indices: *const u32,
    alpha_g: f32,
    beta_g: f32,
    alpha_u: f32,
    beta_u: f32,
    k: u32,
    n: u32,
    words_g: u32,
    words_u: u32,
    routes_per_token: u32,
    dispatches: u32,
    limit: f32,
    out: *mut f32,
) {
    let k_u = k as usize;
    let n_u = n as usize;
    let words_g_u = words_g as usize;
    let words_u_u = words_u as usize;
    let top_k = routes_per_token as usize;
    debug_assert!(k_u % 512 == 0 && n_u % 32 == 0);
    for dispatch in 0..dispatches as usize {
        // SAFETY: dispatch < batch_size * routes_per_token; all buffer accesses below are
        // bounded by the expert-plane and geometry contract (packed: [E, n, words]).
        let raw_idx = unsafe { *indices.add(dispatch) };
        let (orig_dispatch, expert) = if raw_idx >> 8 != 0 {
            (((raw_idx >> 8) - 1) as usize, (raw_idx & 0xFF) as usize)
        } else {
            (dispatch, raw_idx as usize)
        };
        let token = orig_dispatch / top_k;
        let x_row = unsafe { std::slice::from_raw_parts(x.add(token * k_u), k_u) };
        let mut hidden = [0.0f32; 32];
        for tile in 0..(n_u / 32) {
            for tile_row in 0..32usize {
                let row = tile * 32 + tile_row;
                let g_words =
                    unsafe { std::slice::from_raw_parts(packed_g.add((expert * n_u + row) * words_g_u), words_g_u) };
                let u_words =
                    unsafe { std::slice::from_raw_parts(packed_u.add((expert * n_u + row) * words_u_u), words_u_u) };
                // Reference accumulation: per-lane partials persist across 512-blocks,
                // coefficients in order, partials reduced in lane order.
                let mut lane_g = [0.0f32; 32];
                let mut lane_u = [0.0f32; 32];
                for block in 0..(k_u / 512) {
                    for lane in 0..32usize {
                        for i in 0..16usize {
                            let column = block * 512 + lane * 16 + i;
                            let value = x_row[column];
                            lane_g[lane] =
                                value.mul_add(jangtq_level(g_words, column, BITS_G, alpha_g, beta_g), lane_g[lane]);
                            lane_u[lane] =
                                value.mul_add(jangtq_level(u_words, column, BITS_U, alpha_u, beta_u), lane_u[lane]);
                        }
                    }
                }
                let g = lane_g.iter().sum::<f32>() * unsafe { *scales_g.add(expert * n_u + row) }.to_f32();
                let u = lane_u.iter().sum::<f32>() * unsafe { *scales_u.add(expert * n_u + row) }.to_f32();
                let (g, u) = if limit > 0.0 {
                    (g.min(limit), u.clamp(-limit, limit))
                } else {
                    (g, u)
                };
                hidden[tile_row] = g / (1.0f32 + (-g).exp()) * u;
            }
            hadamard32_f32(&mut hidden);
            for (lane, value) in hidden.iter().enumerate() {
                unsafe { *out.add(orig_dispatch * n_u + tile * 32 + lane) = *value };
            }
        }
    }
}

// CPU reference for in-place float32 hadamard32 rotation per 32-vector for intermediate activations.
// Decoupled from JangtqFusedGateUpTile to mirror the 16-row tile architecture of Metal.
#[kernel(JangtqHadamard32F32)]
pub fn jangtq_hadamard32_f32(
    x: *mut f32,
    row_length: u32,
    rows: u32,
) {
    let row_length_u = row_length as usize;
    debug_assert!(row_length_u % 32 == 0);
    for row_idx in 0..rows as usize {
        for block_idx in 0..(row_length_u / 32) {
            let mut values = [0.0f32; 32];
            for (lane, value) in values.iter_mut().enumerate() {
                *value = unsafe { *x.add(row_idx * row_length_u + block_idx * 32 + lane) };
            }
            hadamard32_f32(&mut values);
            for (lane, value) in values.iter().enumerate() {
                unsafe { *x.add(row_idx * row_length_u + block_idx * 32 + lane) = *value };
            }
        }
    }
}

// CPU reference for tile-based fused gate-up (mirrors Metal JangtqFusedGateUpTile):
// Computes 16-row tiles grouped by expert, with outputs stored directly into out without inlined Hadamard.
#[kernel(JangtqFusedGateUpTile)]
#[variants(BITS_G, 2)]
#[variants(BITS_U, 2)]
// The CPU reference mirrors the Metal kernel ABI: every variant binds the same
// buffer list, so some params are reserved for the sibling variants.
#[allow(unused_variables)]
pub fn jangtq_fused_gate_up_tile<const BITS_G: u32, const BITS_U: u32>(
    x: *const f32,
    packed_g: *const u32,
    scales_g: *const f16,
    packed_u: *const u32,
    scales_u: *const f16,
    sorted_indices: *const u32,
    expert_offsets: *const u32,
    expert_counts: *const u32,
    alpha_g: f32,
    beta_g: f32,
    alpha_u: f32,
    beta_u: f32,
    k: u32,
    n: u32,
    words_g: u32,
    words_u: u32,
    routes_per_token: u32,
    dispatches: u32,
    limit: f32,
    out: *mut f32,
) {
    let k_u = k as usize;
    let n_u = n as usize;
    let words_g_u = words_g as usize;
    let words_u_u = words_u as usize;
    let top_k = routes_per_token as usize;
    debug_assert!(k_u % 512 == 0 && n_u % 32 == 0);
    for exp in 0..256usize {
        let count = unsafe { *expert_counts.add(exp) } as usize;
        if count == 0 {
            continue;
        }
        let start_offset = unsafe { *expert_offsets.add(exp) } as usize;
        for d in 0..count {
            let raw = unsafe { *sorted_indices.add(start_offset + d) };
            let orig_dispatch = ((raw >> 8) - 1) as usize;
            let token = orig_dispatch / top_k;
            let x_row = unsafe { std::slice::from_raw_parts(x.add(token * k_u), k_u) };
            for tile in 0..(n_u / 32) {
                let mut hidden = [0.0f32; 32];
                for tile_row in 0..32usize {
                    let row = tile * 32 + tile_row;
                    let g_words =
                        unsafe { std::slice::from_raw_parts(packed_g.add((exp * n_u + row) * words_g_u), words_g_u) };
                    let u_words =
                        unsafe { std::slice::from_raw_parts(packed_u.add((exp * n_u + row) * words_u_u), words_u_u) };
                    let mut lane_g = [0.0f32; 32];
                    let mut lane_u = [0.0f32; 32];
                    for block in 0..(k_u / 512) {
                        for lane in 0..32usize {
                            for i in 0..16usize {
                                let column = block * 512 + lane * 16 + i;
                                let value = x_row[column];
                                lane_g[lane] =
                                    value.mul_add(jangtq_level(g_words, column, BITS_G, alpha_g, beta_g), lane_g[lane]);
                                lane_u[lane] =
                                    value.mul_add(jangtq_level(u_words, column, BITS_U, alpha_u, beta_u), lane_u[lane]);
                            }
                        }
                    }
                    let mut g_scaled = lane_g.iter().sum::<f32>() * unsafe { *scales_g.add(exp * n_u + row) }.to_f32();
                    let mut u_scaled = lane_u.iter().sum::<f32>() * unsafe { *scales_u.add(exp * n_u + row) }.to_f32();
                    if limit > 0.0 {
                        g_scaled = g_scaled.min(limit);
                        u_scaled = u_scaled.clamp(-limit, limit);
                    }
                    hidden[tile_row] = (g_scaled / (1.0f32 + (-g_scaled).exp())) * u_scaled;
                }
                for (lane, value) in hidden.iter().enumerate() {
                    unsafe { *out.add(orig_dispatch * n_u + tile * 32 + lane) = *value };
                }
            }
        }
    }
}

#[kernel(JangtqWeightedDown)]
#[variants(T, bf16)]
#[variants(BITS, 2, 3)]
pub fn jangtq_weighted_down<T: DataTypeRepr + Float, const BITS: u32>(
    hidden: *const f32,
    packed: *const u32,
    scales: *const f16,
    indices: *const u32,
    scores: *const f32,
    out: *mut T,
    alpha: f32,
    beta: f32,
    h: u32,
    n: u32,
    words: u32,
    routes_per_token: u32,
    batch_size: u32,
) {
    let h_u = h as usize;
    let n_u = n as usize;
    let words_u = words as usize;
    let top_k = routes_per_token as usize;
    for token in 0..batch_size as usize {
        for row in 0..n_u {
            let mut total = 0.0f32;
            for route in 0..top_k {
                let dispatch = token * top_k + route;
                // SAFETY: dispatch < batch_size * routes_per_token; packed plane bounds hold
                // by the [E, n, words] geometry contract.
                let expert = unsafe { *indices.add(dispatch) } as usize;
                let score = unsafe { *scores.add(dispatch) };
                let row_words =
                    unsafe { std::slice::from_raw_parts(packed.add((expert * n_u + row) * words_u), words_u) };
                let mut lane_acc = [0.0f32; 32];
                for block in 0..h_u.div_ceil(512) {
                    for lane in 0..32usize {
                        let column_base = block * 512 + lane * 16;
                        if column_base >= h_u {
                            continue;
                        }
                        for i in 0..16usize {
                            let column = column_base + i;
                            let value = unsafe { *hidden.add(dispatch * h_u + column) };
                            lane_acc[lane] =
                                value.mul_add(jangtq_level(row_words, column, BITS, alpha, beta), lane_acc[lane]);
                        }
                    }
                }
                let dot = lane_acc.iter().sum::<f32>();
                total += score * dot * unsafe { *scales.add(expert * n_u + row) }.to_f32();
            }
            unsafe { *out.add(token * n_u + row) = T::from(total).unwrap() };
        }
    }
}

#[kernel(JangtqWeightedDownCombine)]
#[variants(T, bf16)]
#[variants(BITS, 2, 3)]
pub fn jangtq_weighted_down_combine<T: DataTypeRepr + Float, const BITS: u32>(
    hidden: *const f32,
    packed: *const u32,
    scales: *const f16,
    indices: *const u32,
    scores: *const f32,
    shared_output: *const T,
    shared_gate: *const T,
    out: *mut T,
    alpha: f32,
    beta: f32,
    h: u32,
    n: u32,
    words: u32,
    routes_per_token: u32,
    batch_size: u32,
) {
    let h_u = h as usize;
    let n_u = n as usize;
    let words_u = words as usize;
    let top_k = routes_per_token as usize;
    for token in 0..batch_size as usize {
        let gate_f = unsafe { *shared_gate.add(token) }.to_f32().unwrap();
        let gate = T::from(1.0f32 / (1.0f32 + (-gate_f).exp())).unwrap().to_f32().unwrap();
        for row in 0..n_u {
            let mut total = 0.0f32;
            for route in 0..top_k {
                let dispatch = token * top_k + route;
                let expert = unsafe { *indices.add(dispatch) } as usize;
                let score = unsafe { *scores.add(dispatch) };
                let row_words =
                    unsafe { std::slice::from_raw_parts(packed.add((expert * n_u + row) * words_u), words_u) };
                let mut lane_acc = [0.0f32; 32];
                for block in 0..h_u.div_ceil(512) {
                    for lane in 0..32usize {
                        let column_base = block * 512 + lane * 16;
                        if column_base >= h_u {
                            continue;
                        }
                        for i in 0..16usize {
                            let column = column_base + i;
                            let value = unsafe { *hidden.add(dispatch * h_u + column) };
                            lane_acc[lane] =
                                value.mul_add(jangtq_level(row_words, column, BITS, alpha, beta), lane_acc[lane]);
                        }
                    }
                }
                let dot = lane_acc.iter().sum::<f32>();
                total += score * dot * unsafe { *scales.add(expert * n_u + row) }.to_f32();
            }
            let routed_value = T::from(total).unwrap().to_f32().unwrap();
            let shared_value = unsafe { *shared_output.add(token * n_u + row) }.to_f32().unwrap();
            let product = T::from(gate * shared_value).unwrap().to_f32().unwrap();
            let acc = routed_value + product;
            unsafe { *out.add(token * n_u + row) = T::from(acc).unwrap() };
        }
    }
}

#[kernel(JangtqDownProject)]
#[variants(BITS, 2, 3)]
// Mirror of the Metal kernel ABI: `routes_per_token` is bound on every variant
// even where the reference loop indexes dispatches directly.
#[allow(unused_variables)]
pub fn jangtq_down_project<const BITS: u32>(
    hidden: *const f32,
    packed: *const u32,
    scales: *const f16,
    indices: *const u32,
    scores: *const f32,
    out: *mut f32,
    alpha: f32,
    beta: f32,
    h: u32,
    n: u32,
    words: u32,
    routes_per_token: u32,
    dispatches: u32,
) {
    let h_u = h as usize;
    let n_u = n as usize;
    let words_u = words as usize;
    for dispatch in 0..dispatches as usize {
        let raw = unsafe { *indices.add(dispatch) };
        let (orig_dispatch, expert) = if (raw >> 8) != 0 {
            (((raw >> 8) - 1) as usize, (raw & 0xFF) as usize)
        } else {
            (dispatch, raw as usize)
        };
        let score = unsafe { *scores.add(orig_dispatch) };
        for row in 0..n_u {
            let row_words =
                unsafe { std::slice::from_raw_parts(packed.add((expert * n_u + row) * words_u), words_u) };
            let mut lane_acc = [0.0f32; 32];
            for block in 0..h_u.div_ceil(512) {
                for lane in 0..32usize {
                    let column_base = block * 512 + lane * 16;
                    if column_base >= h_u {
                        continue;
                    }
                    for i in 0..16usize {
                        let column = column_base + i;
                        let value = unsafe { *hidden.add(orig_dispatch * h_u + column) };
                        lane_acc[lane] =
                            value.mul_add(jangtq_level(row_words, column, BITS, alpha, beta), lane_acc[lane]);
                    }
                }
            }
            let dot = lane_acc.iter().sum::<f32>();
            let scale = unsafe { *scales.add(expert * n_u + row) }.to_f32();
            unsafe { *out.add(orig_dispatch * n_u + row) = score * dot * scale };
        }
    }
}

#[kernel(JangtqDownProjectTile)]
#[variants(BITS, 2, 3)]
// Mirror of the Metal kernel ABI: the tile variant walks sorted indices, so
// `routes_per_token`/`dispatches` stay bound for the non-tile sibling.
#[allow(unused_variables)]
pub fn jangtq_down_project_tile<const BITS: u32>(
    hidden: *const f32,
    packed: *const u32,
    scales: *const f16,
    sorted_indices: *const u32,
    expert_offsets: *const u32,
    expert_counts: *const u32,
    scores: *const f32,
    out: *mut f32,
    alpha: f32,
    beta: f32,
    h: u32,
    n: u32,
    words: u32,
    routes_per_token: u32,
    dispatches: u32,
) {
    let h_u = h as usize;
    let n_u = n as usize;
    let words_u = words as usize;
    for exp in 0..256usize {
        let count = unsafe { *expert_counts.add(exp) } as usize;
        if count == 0 {
            continue;
        }
        let start_offset = unsafe { *expert_offsets.add(exp) } as usize;
        for d in 0..count {
            let raw = unsafe { *sorted_indices.add(start_offset + d) };
            let orig_dispatch = ((raw >> 8) - 1) as usize;
            let score = unsafe { *scores.add(orig_dispatch) };
            for row in 0..n_u {
                let row_words =
                    unsafe { std::slice::from_raw_parts(packed.add((exp * n_u + row) * words_u), words_u) };
                let mut lane_acc = [0.0f32; 32];
                for block in 0..h_u.div_ceil(512) {
                    for lane in 0..32usize {
                        let column_base = block * 512 + lane * 16;
                        if column_base >= h_u {
                            continue;
                        }
                        for i in 0..16usize {
                            let column = column_base + i;
                            let value = unsafe { *hidden.add(orig_dispatch * h_u + column) };
                            lane_acc[lane] =
                                value.mul_add(jangtq_level(row_words, column, BITS, alpha, beta), lane_acc[lane]);
                        }
                    }
                }
                let dot = lane_acc.iter().sum::<f32>();
                let scale = unsafe { *scales.add(exp * n_u + row) }.to_f32();
                unsafe { *out.add(orig_dispatch * n_u + row) = score * dot * scale };
            }
        }
    }
}

#[kernel(JangtqDownReduce)]
#[variants(T, bf16)]
#[variants(BD, 256)]
pub fn jangtq_down_reduce<T: DataTypeRepr + Float, const BD: u32>(
    routed_dispatches: *const f32,
    routed_outputs: *mut T,
    model_dim: u32,
    batch_size: u32,
    routes_per_token: u32,
) {
    let n_u = model_dim as usize;
    let routes = routes_per_token as usize;
    for token in 0..batch_size as usize {
        for dim in 0..n_u {
            let mut sum = 0.0f32;
            for r in 0..routes {
                let dispatch = token * routes + r;
                sum += unsafe { *routed_dispatches.add(dispatch * n_u + dim) };
            }
            unsafe { *routed_outputs.add(token * n_u + dim) = T::from(sum).unwrap() };
        }
    }
}

#[kernel(JangtqCombine)]
#[variants(T, bf16)]
#[variants(BD, 256)]
pub fn jangtq_combine<T: DataTypeRepr + Float, const BD: u32>(
    routed: *const T,
    shared_output: *const T,
    shared_gate: *const T,
    output: *mut T,
    model_dim: u32,
    batch_size: u32,
) {
    let model_dim_u = model_dim as usize;
    for token_idx in 0..batch_size as usize {
        // Reference rounding: sigmoid -> bf16, multiply -> bf16, add -> bf16.
        let gate_f = unsafe { *shared_gate.add(token_idx) }.to_f32().unwrap();
        let gate = T::from(1.0f32 / (1.0f32 + (-gate_f).exp())).unwrap().to_f32().unwrap();
        for dim_idx in 0..model_dim_u {
            let routed_value = unsafe { *routed.add(token_idx * model_dim_u + dim_idx) }.to_f32().unwrap();
            let shared_value = unsafe { *shared_output.add(token_idx * model_dim_u + dim_idx) }.to_f32().unwrap();
            let product = T::from(gate * shared_value).unwrap().to_f32().unwrap();
            let acc = routed_value + product;
            unsafe { *output.add(token_idx * model_dim_u + dim_idx) = T::from(acc).unwrap() };
        }
    }
}

#[kernel(JangtqDownReduceCombine)]
#[variants(T, bf16)]
#[variants(BD, 256)]
pub fn jangtq_down_reduce_combine<T: DataTypeRepr + Float, const BD: u32>(
    routed_dispatches: *const f32,
    shared_output: *const T,
    shared_gate: *const T,
    output: *mut T,
    model_dim: u32,
    batch_size: u32,
    routes_per_token: u32,
) {
    let model_dim_u = model_dim as usize;
    let routes = routes_per_token as usize;
    for token_idx in 0..batch_size as usize {
        let gate_f = unsafe { *shared_gate.add(token_idx) }.to_f32().unwrap();
        let gate = T::from(1.0f32 / (1.0f32 + (-gate_f).exp())).unwrap().to_f32().unwrap();
        for dim_idx in 0..model_dim_u {
            let mut sum = 0.0f32;
            for r in 0..routes {
                let dispatch = token_idx * routes + r;
                sum += unsafe { *routed_dispatches.add(dispatch * model_dim_u + dim_idx) };
            }
            let routed_value = T::from(sum).unwrap().to_f32().unwrap();
            let shared_value = unsafe { *shared_output.add(token_idx * model_dim_u + dim_idx) }.to_f32().unwrap();
            let product = T::from(gate * shared_value).unwrap().to_f32().unwrap();
            let acc = routed_value + product;
            unsafe { *output.add(token_idx * model_dim_u + dim_idx) = T::from(acc).unwrap() };
        }
    }
}

#[kernel(JangtqDenseFusedGateUp)]
pub fn jangtq_dense_fused_gate_up(
    x: *const f32,
    packed_g: *const u32,
    scales_g: *const f16,
    packed_u: *const u32,
    scales_u: *const f16,
    out: *mut f32,
    alpha_g: f32,
    beta_g: f32,
    alpha_u: f32,
    beta_u: f32,
    k: u32,
    n: u32,
    words: u32,
    batch_size: u32,
) {
    let k_u = k as usize;
    let n_u = n as usize;
    let words_u = words as usize;
    debug_assert!(k_u % 512 == 0 && n_u % 32 == 0);
    for token in 0..batch_size as usize {
        let x_row = unsafe { std::slice::from_raw_parts(x.add(token * k_u), k_u) };
        let mut hidden = [0.0f32; 32];
        for tile in 0..(n_u / 32) {
            for tile_row in 0..32usize {
                let row = tile * 32 + tile_row;
                let g_words = unsafe { std::slice::from_raw_parts(packed_g.add(row * words_u), words_u) };
                let u_words = unsafe { std::slice::from_raw_parts(packed_u.add(row * words_u), words_u) };
                let mut lane_g = [0.0f32; 32];
                let mut lane_u = [0.0f32; 32];
                for block in 0..(k_u / 512) {
                    for lane in 0..32usize {
                        for i in 0..16usize {
                            let column = block * 512 + lane * 16 + i;
                            let value = x_row[column];
                            lane_g[lane] = value.mul_add(jangtq_level(g_words, column, 3, alpha_g, beta_g), lane_g[lane]);
                            lane_u[lane] = value.mul_add(jangtq_level(u_words, column, 3, alpha_u, beta_u), lane_u[lane]);
                        }
                    }
                }
                let g = lane_g.iter().sum::<f32>() * unsafe { *scales_g.add(row) }.to_f32();
                let u = lane_u.iter().sum::<f32>() * unsafe { *scales_u.add(row) }.to_f32();
                hidden[tile_row] = (g / (1.0f32 + (-g).exp())) * u;
            }
            hadamard32_f32(&mut hidden);
            for (lane, value) in hidden.iter().enumerate() {
                unsafe { *out.add(token * n_u + tile * 32 + lane) = *value };
            }
        }
    }
}

#[kernel(JangtqDenseFusedGateUpMma)]
pub fn jangtq_dense_fused_gate_up_mma(
    x: *const f32,
    packed_g: *const u32,
    scales_g: *const f16,
    packed_u: *const u32,
    scales_u: *const f16,
    out: *mut f32,
    alpha_g: f32,
    beta_g: f32,
    alpha_u: f32,
    beta_u: f32,
    k: u32,
    n: u32,
    words: u32,
    batch_size: u32,
) {
    jangtq_dense_fused_gate_up(
        x, packed_g, scales_g, packed_u, scales_u, out, alpha_g, beta_g, alpha_u, beta_u, k, n, words, batch_size,
    );
}

#[kernel(JangtqDenseDown)]
#[variants(T, bf16)]
pub fn jangtq_dense_down<T: DataTypeRepr + Float>(
    hidden: *const f32,
    packed: *const u32,
    scales: *const f16,
    out: *mut T,
    alpha: f32,
    beta: f32,
    h: u32,
    n: u32,
    words: u32,
    batch_size: u32,
) {
    let h_u = h as usize;
    let n_u = n as usize;
    let words_u = words as usize;
    for token in 0..batch_size as usize {
        for row in 0..n_u {
            let row_words = unsafe { std::slice::from_raw_parts(packed.add(row * words_u), words_u) };
            let mut lane_acc = [0.0f32; 32];
            for block in 0..(h_u / 512) {
                for lane in 0..32usize {
                    for i in 0..16usize {
                        let column = block * 512 + lane * 16 + i;
                        let value = unsafe { *hidden.add(token * h_u + column) };
                        lane_acc[lane] = value.mul_add(jangtq_level(row_words, column, 3, alpha, beta), lane_acc[lane]);
                    }
                }
            }
            let dot = lane_acc.iter().sum::<f32>() * unsafe { *scales.add(row) }.to_f32();
            unsafe { *out.add(token * n_u + row) = T::from(dot).unwrap() };
        }
    }
}

#[kernel(JangtqDenseDownMma)]
#[variants(T, bf16)]
pub fn jangtq_dense_down_mma<T: DataTypeRepr + Float>(
    hidden: *const f32,
    packed: *const u32,
    scales: *const f16,
    out: *mut T,
    alpha: f32,
    beta: f32,
    h: u32,
    n: u32,
    words: u32,
    batch_size: u32,
) {
    jangtq_dense_down(hidden, packed, scales, out, alpha, beta, h, n, words, batch_size);
}
