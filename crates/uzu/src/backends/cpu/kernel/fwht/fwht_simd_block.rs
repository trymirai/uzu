use dsl::kernel;
use half::{bf16, f16};

use crate::ArrayElement;

#[kernel(FwhtSimdBlock)]
#[variants(T, f16, f32, bf16)]
#[variants(N, 32, 64, 128)]
pub fn fwht_simd_block<T: ArrayElement, const N: i32>(
    #[allow(unused)] data: *mut T,
    #[allow(unused)] batch_size: u32,
    #[allow(unused)] scale: f32,
) {
    todo!()
}
