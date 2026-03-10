use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(PoolingCls)]
#[variants(T, f32, f16, bf16)]
pub fn pooling_cls<T: ArrayElement + Float>(
    #[allow(unused)] input: *const T,
    #[allow(unused)] output: *mut T,
    #[allow(unused)] seq_len: u32,
    #[allow(unused)] hidden_dim: u32,
    #[allow(unused)] batch_size: u32,
) {
    todo!()
}

#[kernel(PoolingMean)]
#[variants(T, f32, f16, bf16)]
pub fn pooling_mean<T: ArrayElement + Float>(
    #[allow(unused)] input: *const T,
    #[allow(unused)] output: *mut T,
    #[allow(unused)] seq_len: u32,
    #[allow(unused)] hidden_dim: u32,
    #[allow(unused)] batch_size: u32,
) {
    todo!()
}
