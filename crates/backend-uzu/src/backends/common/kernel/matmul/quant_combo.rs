use crate::backends::common::gpu_types::{QuantizationMethod, QuantizationMode};

#[derive(Debug, Clone, Copy)]
pub struct MatmulQuantCombo {
    pub method: QuantizationMethod,
    pub mode: QuantizationMode,
    pub group_size: u32,
}
