pub mod activation_transform;

use crate::backends::common::gpu_types::HADAMARD_TRANSFORM_BLOCK_SIZE;

pub const INT8_SYMMETRIC_QUANTIZATION_MAXIMUM: f32 = 127.0;

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

pub(crate) fn hadamard_transform(values: &mut [f32; HADAMARD_TRANSFORM_BLOCK_SIZE]) {
    let mut stride = 1;
    while stride < HADAMARD_TRANSFORM_BLOCK_SIZE {
        for lane in 0..HADAMARD_TRANSFORM_BLOCK_SIZE {
            if lane & stride == 0 {
                let a = values[lane];
                let b = values[lane | stride];
                values[lane] = a + b;
                values[lane | stride] = a - b;
            }
        }
        stride <<= 1;
    }
    let scale = 1.0 / (HADAMARD_TRANSFORM_BLOCK_SIZE as f32).sqrt();
    for v in values.iter_mut() {
        *v *= scale;
    }
}
