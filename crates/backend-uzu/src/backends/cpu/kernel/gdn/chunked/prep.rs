use half::{bf16, f16};
use proc_macros::kernel;

#[kernel(DeltaNetChunkedPrep)]
#[variants(T, f32, f16, bf16)]
#[variants(HEAD_K_DIM, 128)]
#[allow(unused_variables)]
pub fn delta_net_chunked_prep<T, const HEAD_K_DIM: u32>(
    in_proj: *const T,
    a_log: *const f32,
    dt_bias: *const f32,
    q_norm_out: *mut f32,
    k_norm_out: *mut f32,
    beta_out: *mut f32,
    log_decay_out: *mut f32,
    num_v_heads: u32,
    num_k_heads: u32,
    key_dim: u32,
    value_dim: u32,
    suffix_len: u32,
) {
    panic!("DeltaNet chunked prefill is Metal-only");
}

#[kernel(DeltaNetChunkedCumsum)]
#[allow(unused_variables)]
pub fn delta_net_chunked_cumsum(
    log_decay: *const f32,
    g_out: *mut f32,
    num_v_heads: u32,
    suffix_len: u32,
    chunk_size: u32,
) {
    panic!("DeltaNet chunked prefill is Metal-only");
}
