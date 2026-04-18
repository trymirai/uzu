use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(AudioFsqEncode)]
#[variants(T, f32, f16, bf16)]
pub fn audio_fsq_encode<T: ArrayElement + Float>(
    input: *const T,
    tokens: *mut i32,
    lengths: *const i32,
    num_groups: i32,
    seq_len: i32,
    codebook_dim: i32,
    num_levels: &[i32],
    dim_base_index: &[i32],
    eps: f32,
    batch_size: i32,
) {
    let _ = (input, tokens, lengths, num_groups, seq_len, codebook_dim, num_levels, dim_base_index, eps, batch_size);
    todo!()
}
