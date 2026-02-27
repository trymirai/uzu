use dsl::kernel;
use half::{bf16, f16};

use crate::ArrayElement;

#[kernel(FwhtBlock)]
#[variants(T, f16, f32, bf16)]
#[variants(BLOCK_SIZE, 32, 64, 128, 256)]
pub fn fwht_block<T: ArrayElement, const BLOCK_SIZE: i32>(
    #[allow(unused)] data: *mut T,
    #[allow(unused)] batch_size: u32,
    #[allow(unused)] n: u32,
    #[allow(unused)] scale: f32,
) {
    todo!()
}
