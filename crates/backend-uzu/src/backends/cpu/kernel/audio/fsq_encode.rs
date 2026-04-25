use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(AudioFsqEncode)]
#[variants(T, f32, f16, bf16)]
pub fn audio_fsq_encode<T: ArrayElement + Float>(
    #[allow(unused)] input: *const T,
    #[allow(unused)] tokens: *mut i32,
    #[allow(unused)] lengths: *const i32,
    #[allow(unused)] num_groups: i32,
    #[allow(unused)] seq_len: i32,
    #[allow(unused)] codebook_dim: i32,
    #[allow(unused)] num_levels: &[i32],
    #[allow(unused)] dim_base_index: &[i32],
    #[allow(unused)] eps: f32,
    #[allow(unused)] batch_size: i32,
) {
    todo!()
}
