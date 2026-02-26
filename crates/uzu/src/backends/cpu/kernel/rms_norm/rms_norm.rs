use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(RMSNorm)]
#[variants(InputT, f32, f16, bf16)]
#[variants(ScaleT, f32, f16, bf16)]
#[variants(OutputT, f32, f16, bf16)]
#[variants(AccumT, f32, f16)]
pub fn rms_norm<
    InputT: ArrayElement + Float,
    ScaleT: ArrayElement + Float,
    OutputT: ArrayElement + Float,
    AccumT: ArrayElement + Float,
>(
    #[allow(unused)]
    #[optional(!in_place)]
    input: Option<*const InputT>,
    #[allow(unused)] scales: *const ScaleT,
    #[allow(unused)] output: *mut OutputT,
    #[allow(unused)] batch_size: u32,
    #[allow(unused)] element_count: u32,
    #[allow(unused)] epsilon: f32,
    #[allow(unused)] scale_offset: f32,
    #[allow(unused)] full_layer: bool,
    #[allow(unused)]
    #[specialize]
    in_place: bool,
) {
    todo!()
}
