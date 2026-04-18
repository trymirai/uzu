use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(AudioFsqDecode)]
#[variants(T, f32, f16, bf16)]
pub fn audio_fsq_decode<T: ArrayElement + Float>(
    tokens: *const i32,
    out: *mut T,
    lengths: *const i32,
    num_groups: i32,
    seq_len: i32,
    codebook_dim: i32,
    num_levels: &[i32],
    dim_base_index: &[i32],
    batch_size: i32,
) {
    let _ = (tokens, out, lengths, num_groups, seq_len, codebook_dim, num_levels, dim_base_index, batch_size);
    todo!()
}
