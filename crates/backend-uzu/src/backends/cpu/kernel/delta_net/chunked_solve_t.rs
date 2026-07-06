use half::bf16;
use num_traits::NumCast;
use proc_macros::kernel;

// CPU mirror of `DeltaNetChunkedSolveT`: the dense unit-lower-triangular inverse
// T = (I + A)^{-1} per (chunk, v-head) via one block forward substitution over
// the block inverses (a_packed strips + a_inv diagonal-block inverses) with an
// identity RHS -- BuildWU with RHS = I. T is BF16 and the substitution reads it
// back in BF16 (matching the Metal kernel and the old W/U precision contract).
#[kernel(DeltaNetChunkedSolveT)]
#[variants(CHUNK_SIZE, 16, 32, 64)]
#[variants(BV, 16, 32)]
#[variants(USE_MXU, false, true)]
pub fn delta_net_chunked_solve_t<const CHUNK_SIZE: u32, const BV: u32, const USE_MXU: bool>(
    a_packed: *const f32,
    a_inv: *const f32,
    t_out: *mut bf16,
    num_v_heads: u32,
    suffix_len: u32,
) {
    let _ = BV;
    let _ = USE_MXU;
    let chunk_size = CHUNK_SIZE as usize;
    let num_v_heads = num_v_heads as usize;
    let suffix_len = suffix_len as usize;
    let num_chunks = suffix_len.div_ceil(chunk_size);
    let block = 16usize;
    let num_blocks = chunk_size.div_ceil(block);
    let num_col_pairs = num_blocks.div_ceil(2);

    for chunk in 0..num_chunks {
        for hv in 0..num_v_heads {
            for block_idx in 0..num_blocks {
                let row_base = block_idx * block;

                // RHS = identity: acc[row][col] = (row_base+row == col) ? 1 : 0.
                let mut acc_t = vec![0.0f32; block * chunk_size];
                for row in 0..block {
                    let global_row = row_base + row;
                    if global_row < chunk_size {
                        acc_t[row * chunk_size + global_row] = 1.0;
                    }
                }

                for prev_block in 0..block_idx {
                    for row in 0..block {
                        for prev_row in 0..block {
                            let local_col = (prev_block % 2) * block + prev_row;
                            let a_idx = (((chunk * num_v_heads + hv) * num_blocks + block_idx) * num_col_pairs
                                + prev_block / 2)
                                * (block * 2 * block)
                                + row * (2 * block)
                                + local_col;
                            let a = unsafe { *a_packed.add(a_idx) };
                            let prev_token = prev_block * block + prev_row;
                            let prev_out_base = ((chunk * num_v_heads + hv) * chunk_size + prev_token) * chunk_size;
                            for d in 0..chunk_size {
                                let t_prev: f32 = NumCast::from(unsafe { *t_out.add(prev_out_base + d) }).unwrap();
                                acc_t[row * chunk_size + d] -= a * t_prev;
                            }
                        }
                    }
                }

                for row in 0..block {
                    let local_token = row_base + row;
                    let out_base = ((chunk * num_v_heads + hv) * chunk_size + local_token) * chunk_size;
                    for d in 0..chunk_size {
                        let mut value = 0.0f32;
                        for source_row in 0..block {
                            let inv_idx = ((chunk * num_v_heads + hv) * num_blocks + block_idx) * block * block
                                + row * block
                                + source_row;
                            let inv = unsafe { *a_inv.add(inv_idx) };
                            value += inv * acc_t[source_row * chunk_size + d];
                        }
                        unsafe { *t_out.add(out_base + d) = bf16::from_f32(value) };
                    }
                }
            }
        }
    }
}
