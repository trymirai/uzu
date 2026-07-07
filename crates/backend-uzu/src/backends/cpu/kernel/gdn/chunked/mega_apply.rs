use half::{bf16, f16};
use proc_macros::kernel;

#[kernel(DeltaNetChunkedMegaApply)]
#[variants(T, f32, f16, bf16)]
#[variants(O, f32, bf16)]
#[variants(VT, 32)]
#[variants(USE_MXU, false, true)]
#[allow(unused_variables, clippy::too_many_arguments)]
pub fn delta_net_chunked_mega_apply<T, O, const VT: u32, const USE_MXU: bool>(
    q_norm: *const f32,
    k_norm: *const f32,
    in_proj: *const T,
    qk_scaled: *const f32,
    t_mat: *const bf16,
    g: *const f32,
    beta: *const f32,
    state: *mut f32,
    out: *mut O,
    num_v_heads: u32,
    num_k_heads: u32,
    head_v_dim: u32,
    key_dim: u32,
    value_dim: u32,
    suffix_len: u32,
) {
    panic!("DeltaNet chunked prefill is Metal-only");
}
