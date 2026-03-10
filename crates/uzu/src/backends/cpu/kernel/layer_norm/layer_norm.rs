use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(LayerNorm)]
#[variants(IN, f32, f16, bf16)]
#[variants(SC, f32, f16, bf16)]
#[variants(OUT, f32, f16, bf16)]
#[variants(ACC, f32)]
pub fn layer_norm<
    IN: ArrayElement + Float,
    SC: ArrayElement + Float,
    OUT: ArrayElement + Float,
    ACC: ArrayElement + Float,
>(
    #[allow(unused)]
    #[optional(!in_place)]
    input: Option<*const IN>,
    #[allow(unused)] scales: *const SC,
    #[allow(unused)] output: *mut OUT,
    #[allow(unused)] batch_size: u32,
    #[allow(unused)] model_dim: u32,
    #[allow(unused)] epsilon: f32,
    #[allow(unused)] scale_offset: f32,
    #[allow(unused)] full_layer: u32,
    #[allow(unused)]
    #[specialize]
    in_place: bool,
) {
    todo!()
}
