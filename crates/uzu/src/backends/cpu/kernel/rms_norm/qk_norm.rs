use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(QKNorm)]
#[variants(InputT, f32, f16, bf16)]
#[variants(ScaleT, f32, f16, bf16)]
#[variants(OutputT, f32, f16, bf16)]
#[variants(AccumT, f32, f16)]
pub fn qk_norm<
    InputT: ArrayElement + Float,
    ScaleT: ArrayElement + Float,
    OutputT: ArrayElement + Float,
    AccumT: ArrayElement + Float,
>(
    #[allow(unused)]
    #[optional(!in_place)]
    qkv_input: Option<*const InputT>,
    #[allow(unused)] scales: *const ScaleT,
    #[allow(unused)] qkv_output: *mut OutputT,
    #[allow(unused)] batch_size: u32,
    #[allow(unused)] num_q_heads: u32,
    #[allow(unused)] num_kv_heads: u32,
    #[allow(unused)] head_dim: u32,
    #[allow(unused)] epsilon: f32,
    #[allow(unused)] scale_offset: f32,
    #[allow(unused)] head_offset: u32,
    #[allow(unused)] head_count: u32,
    #[allow(unused)] full_layer: bool,
    #[allow(unused)]
    #[specialize]
    in_place: bool,
) {
    todo!()
}
