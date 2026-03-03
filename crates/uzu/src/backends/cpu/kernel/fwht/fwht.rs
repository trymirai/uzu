use dsl::kernel;
use half::{bf16, f16};

use crate::ArrayElement;

#[kernel(Fwht)]
#[variants(T, f16, f32, bf16)]
#[variants(N, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)]
pub fn fwht<T: ArrayElement, const N: i32>(
    #[allow(unused)] data: *mut T,
    #[allow(unused)] batch_size: u32,
    #[allow(unused)] scale: f32,
) {
    todo!()
}
