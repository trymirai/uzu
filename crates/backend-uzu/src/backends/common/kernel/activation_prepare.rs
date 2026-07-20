use crate::backends::common::gpu_types::{ACTIVATION_QUANTIZATION_GROUP_SIZE, INT8_SYMMETRIC_QUANTIZATION_MAXIMUM};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ActivationPrepareConfig {
    pub enabled: bool,
}

impl ActivationPrepareConfig {
    pub const fn supports_group_size(
        self,
        group_size: usize,
    ) -> bool {
        self.enabled && group_size == ACTIVATION_QUANTIZATION_GROUP_SIZE as usize
    }
}

pub fn min_max_symmetric_divisor(values: &[f32]) -> f32 {
    let (min, max) =
        values.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), &value| (min.min(value), max.max(value)));
    let magnitude = min.abs().max(max.abs());
    if magnitude.is_finite() && magnitude > 0.0 {
        magnitude / INT8_SYMMETRIC_QUANTIZATION_MAXIMUM
    } else {
        1.0
    }
}

pub fn quantize_symmetric_i8(
    value: f32,
    divisor: f32,
) -> i8 {
    (value / divisor).round().clamp(-INT8_SYMMETRIC_QUANTIZATION_MAXIMUM, INT8_SYMMETRIC_QUANTIZATION_MAXIMUM) as i8
}

pub fn pack_signed_weight_codes(weights: &[u8]) -> Vec<u8> {
    weights.iter().map(|code| code ^ 0x80).collect()
}
