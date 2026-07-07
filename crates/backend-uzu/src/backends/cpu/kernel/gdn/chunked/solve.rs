use proc_macros::kernel;

#[kernel(DeltaNetChunkedSolve)]
#[variants(CHUNK_SIZE, 16, 32, 64)]
#[variants(RECOMPUTE_G, false, true)]
#[allow(unused_variables)]
pub fn delta_net_chunked_solve<const CHUNK_SIZE: u32, const RECOMPUTE_G: bool>(
    kk: *const f32,
    beta: *const f32,
    g_or_log_decay: *const f32,
    a_packed: *mut f32,
    a_inv: *mut f32,
    num_v_heads: u32,
    num_k_heads: u32,
    suffix_len: u32,
) {
    panic!("DeltaNet chunked prefill is Metal-only");
}
