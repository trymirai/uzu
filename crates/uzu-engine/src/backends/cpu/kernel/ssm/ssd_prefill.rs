use half::{bf16, f16};
use num_traits::Float;
use uzu_engine_macros::kernel;

use crate::{array::ArrayElement, backends::common::gpu_types::ActivationType};

/// `ssd_prefill` for a state of exactly 64 elements: panics when there is work and `state_size` is not 64.
#[kernel(SSDPrefill64)]
#[variants(T, f32, f16, bf16)]
pub fn ssd_prefill64<T: ArrayElement + Float>(
    x: *const T,
    dt_raw: *const T,
    b: *const T,
    c: *const T,
    d: *const T,
    z: *const T,
    state: *mut T,
    y: *mut T,
    suffix_len: u32,
    group_size: u32,
    state_size: u32,
    x_strides: &[u32; 3],
    dt_strides: &[u32; 2],
    cb_strides: &[u32; 3],
    state_strides: &[u32; 3],
    num_heads: u32,
    head_dim: u32,
) {
    let work = suffix_len > 0 && num_heads > 0 && head_dim > 0;
    assert!(!work || state_size == 64, "SSDPrefill64: state_size {state_size} is not 64");
    ssd_prefill(
        x,
        dt_raw,
        b,
        c,
        d,
        z,
        state,
        y,
        suffix_len,
        group_size,
        state_size,
        x_strides,
        dt_strides,
        cb_strides,
        state_strides,
        num_heads,
        head_dim,
    );
}

/// Mamba2's SSD recurrence over `suffix_len` tokens Q for every head h and head element e: x, z and y are [Q, H, Dh]
/// at `x_strides`, dt_raw [Q, H] at `dt_strides`, B and C [Q, groups, N] at `cb_strides` with group h /
/// max(group_size, 1), d [H], and the state [H, Dh, N] at `state_strides`.
///
/// Each state row is widened to f32 once, kept in f32 across all tokens and stored as T once at the end. Per token,
/// in separately rounded f32 operations: decay = exp(-softplus(dt_raw)), s_i = s_i decay + B_i x for i ascending, the
/// dot from +0 over s_i C_i, then y = T((dot + d x) gate) with gate = SiLU rounded to T. Without work (Q, H or Dh 0)
/// nothing is read or written; with N 0, dt_raw, B, C and the state are never addressed.
///
/// The caller passes aligned pointers whose addressed elements lie in live buffers, y and the state aliasing no other
/// argument, and strides under which all written y and state elements have pairwise distinct addresses.
#[kernel(SSDPrefill)]
#[variants(T, f32, f16, bf16)]
pub fn ssd_prefill<T: ArrayElement + Float>(
    x: *const T,
    dt_raw: *const T,
    b: *const T,
    c: *const T,
    d: *const T,
    z: *const T,
    state: *mut T,
    y: *mut T,
    suffix_len: u32,
    group_size: u32,
    state_size: u32,
    x_strides: &[u32; 3],
    dt_strides: &[u32; 2],
    cb_strides: &[u32; 3],
    state_strides: &[u32; 3],
    num_heads: u32,
    head_dim: u32,
) {
    if suffix_len == 0 || num_heads == 0 || head_dim == 0 {
        return;
    }
    let [suffix_len, state_size, num_heads, head_dim] =
        [suffix_len, state_size, num_heads, head_dim].map(|value| value as usize);
    let x_strides = x_strides.map(|stride| stride as usize);
    let mut row = Vec::new();
    row.try_reserve_exact(state_size).expect("SSDPrefill: cannot allocate the f32 state row");
    row.resize(state_size, 0.0f32);

    unsafe {
        for h in 0..num_heads {
            let skip = (*d.add(h)).to_f32().unwrap();
            for e in 0..head_dim {
                let state_row = (state_size > 0).then(|| h * state_strides[0] as usize + e * state_strides[1] as usize);
                if let Some(state_row) = state_row {
                    for (i, s) in row.iter_mut().enumerate() {
                        *s = (*state.add(state_row + i * state_strides[2] as usize)).to_f32().unwrap();
                    }
                }
                for t in 0..suffix_len {
                    let x_index = t * x_strides[0] + h * x_strides[1] + e * x_strides[2];
                    let this_x = (*x.add(x_index)).to_f32().unwrap();
                    let gate = ActivationType::SILU.activate(*z.add(x_index)).to_f32().unwrap();
                    let mut dot = 0.0f32;
                    if state_size > 0 {
                        let dt_index = t * dt_strides[0] as usize + h * dt_strides[1] as usize;
                        let dt = ActivationType::SOFTPLUS.activate((*dt_raw.add(dt_index)).to_f32().unwrap());
                        let decay = (-dt).exp();
                        let cb_row =
                            t * cb_strides[0] as usize + h / group_size.max(1) as usize * cb_strides[1] as usize;
                        for (i, s) in row.iter_mut().enumerate() {
                            let cb_index = cb_row + i * cb_strides[2] as usize;
                            *s = *s * decay + (*b.add(cb_index)).to_f32().unwrap() * this_x;
                            dot += *s * (*c.add(cb_index)).to_f32().unwrap();
                        }
                    }
                    *y.add(x_index) = T::from((dot + skip * this_x) * gate).unwrap();
                }
                if let Some(state_row) = state_row {
                    for (i, &s) in row.iter().enumerate() {
                        *state.add(state_row + i * state_strides[2] as usize) = T::from(s).unwrap();
                    }
                }
            }
        }
    }
}
