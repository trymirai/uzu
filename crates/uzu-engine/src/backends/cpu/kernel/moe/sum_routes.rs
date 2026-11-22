use half::bf16;
use num_traits::Float;
use uzu_engine_macros::kernel;

use crate::data_type::DataTypeRepr;

#[kernel(MoeSumRoutes)]
#[variants(T, bf16)]
pub fn moe_sum_routes<T: DataTypeRepr + Float>(
    route_outputs: *const f32,
    output: *mut T,
    tokens: u32,
    model_dim: u32,
    top_k: u32,
) {
    let (tokens, model_dim, top_k) = (tokens as usize, model_dim as usize, top_k as usize);
    for token in 0..tokens {
        for dim in 0..model_dim {
            let mut sum = 0.0f32;
            for pick in 0..top_k {
                sum += unsafe { *route_outputs.add((token * top_k + pick) * model_dim + dim) };
            }
            unsafe {
                *output.add(token * model_dim + dim) = T::from(sum).unwrap();
            }
        }
    }
}
