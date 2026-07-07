use proc_macros::kernel;

#[kernel(DeltaNetChunkedGram)]
#[variants(HEAD_K_DIM, 128)]
#[variants(CHUNK_SIZE, 16, 32, 64)]
#[allow(unused_variables)]
pub fn delta_net_chunked_gram<const HEAD_K_DIM: u32, const CHUNK_SIZE: u32>(
    q_norm: *const f32,
    k_norm: *const f32,
    g: *const f32,
    kk_out: *mut f32,
    qk_scaled_out: *mut f32,
    num_v_heads: u32,
    num_k_heads: u32,
    key_dim: u32,
    suffix_len: u32,
) {
    panic!("DeltaNet chunked prefill is Metal-only");
}
