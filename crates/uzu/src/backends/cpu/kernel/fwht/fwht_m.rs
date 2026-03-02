use dsl::kernel;
use half::{bf16, f16};

use crate::ArrayElement;

#[kernel(FwhtM)]
#[variants(T, f16, f32, bf16)]
#[variants(M, 12, 20, 28)]
pub fn fwht_m<T: ArrayElement, const M: i32>(
    #[allow(unused)] data: *mut T,
    #[allow(unused)] batch_size: u32,
    #[allow(unused)] n: u32,
    #[allow(unused)] scale: f32,
) {
    todo!()
}
