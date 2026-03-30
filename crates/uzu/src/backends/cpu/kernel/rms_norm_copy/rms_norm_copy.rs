use dsl::kernel;
use half::{bf16, f16};
use num_traits::Float;

use crate::ArrayElement;

#[kernel(RMSNormCopy)]
#[variants(ScaleT, f32, f16, bf16)]
#[variants(DataT, f32, f16, bf16)]
#[variants(AccumT, f32, f16)]
pub fn rms_norm_copy<ScaleT: ArrayElement + Float, DataT: ArrayElement + Float, AccumT: ArrayElement + Float>(
    #[allow(unused)] main_buffer: *mut DataT,
    #[allow(unused)] shortcut_buffer: *mut DataT,
    #[allow(unused)] scales: *const ScaleT,
    #[allow(unused)] batch_size: u32,
    #[allow(unused)] element_count: u32,
    #[allow(unused)] epsilon: f32,
    #[allow(unused)] scale_offset: f32,
    #[allow(unused)] full_layer: bool,
) {
    todo!()
}
