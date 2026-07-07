use half::bf16;
use proc_macros::kernel;

#[kernel(DeltaNetChunkedSolveT)]
#[variants(CHUNK_SIZE, 16, 32, 64)]
#[variants(BV, 16, 32)]
#[allow(unused_variables)]
pub fn delta_net_chunked_solve_t<const CHUNK_SIZE: u32, const BV: u32>(
    a_packed: *const f32,
    a_inv: *const f32,
    t_out: *mut bf16,
    num_v_heads: u32,
    suffix_len: u32,
) {
    panic!("DeltaNet chunked prefill is Metal-only");
}
