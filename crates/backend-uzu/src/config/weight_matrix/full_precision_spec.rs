use proc_macros::uzu_config;

use crate::config::weight_matrix::Layout;

#[uzu_config(super::WeightMatrixSpec)]
pub struct FullPrecisionSpec {
    pub layout: Layout,
}

impl FullPrecisionSpec {
    /// Full-precision fallback for expert banks that carry no stored spec.
    pub fn output_input() -> Self {
        Self {
            ty: monostate::MustBeStr,
            layout: Layout::OutputInput,
        }
    }
}
