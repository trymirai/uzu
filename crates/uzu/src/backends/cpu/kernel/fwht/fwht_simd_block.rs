use dsl::kernel;
use half::{bf16, f16};

use crate::ArrayElement;

#[kernel(FwhtSimdBlock)]
#[variants(T, f16, f32, bf16, i8)]
#[variants(N, 128, 512, 1024, 2048)]
pub fn fwht_simd_block<T: ArrayElement, const N: i32>(
    #[allow(unused)] data: *mut T,
    #[allow(unused)] batch_size: u32,
    #[allow(unused)] scale: f32,
) {
    todo!()
}
