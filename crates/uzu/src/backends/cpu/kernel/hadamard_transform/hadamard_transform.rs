use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(HadamardTransform)]
#[variants(T, f32, f16, bf16)]
pub fn hadamard_transform_mul<T: ArrayElement + Float>(
    data: *mut T,
    factors: *const i32,
    hidden_dim: u32,
    batch_size: u32,
) {
    let _ = (data, factors, hidden_dim, batch_size);
    todo!()
}
