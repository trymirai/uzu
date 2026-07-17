use crate::backends::common::gpu_types::{
    ActivationQuantScheme, ActivationScaleStatistic, HADAMARD_TRANSFORM_BLOCK_SIZE,
    INT8_ASYMMETRIC_QUANTIZATION_MAXIMUM, INT8_ASYMMETRIC_QUANTIZATION_MINIMUM_MAGNITUDE,
    INT8_SYMMETRIC_QUANTIZATION_MAXIMUM,
};

const INT8_ASYMMETRIC_QUANTIZATION_MINIMUM: f32 = -INT8_ASYMMETRIC_QUANTIZATION_MINIMUM_MAGNITUDE;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivationPrepareConfig {
    pub enabled: bool,
    pub scheme: ActivationQuantScheme,
    pub statistic: ActivationScaleStatistic,
}

impl ActivationPrepareConfig {
    pub const fn supports_group_size(
        self,
        group_size: usize,
    ) -> bool {
        self.enabled && group_size != 0 && group_size.is_multiple_of(HADAMARD_TRANSFORM_BLOCK_SIZE)
    }
}

impl Default for ActivationPrepareConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            scheme: ActivationQuantScheme::Symmetric,
            statistic: ActivationScaleStatistic::AbsMax,
        }
    }
}

pub fn group_stat(
    values: &[f32],
    stat: ActivationScaleStatistic,
) -> f32 {
    match stat {
        ActivationScaleStatistic::AbsMax => values.iter().fold(0.0, |result, value| result.max(value.abs())),
        ActivationScaleStatistic::Rms => {
            let count = values.len().max(1) as f32;
            (values.iter().map(|value| value * value).sum::<f32>() / count).sqrt()
        },
    }
}

fn group_min_max(values: &[f32]) -> (f32, f32) {
    values.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), &value| (min.min(value), max.max(value)))
}

fn group_mean(values: &[f32]) -> f32 {
    let count = values.len().max(1) as f32;
    values.iter().sum::<f32>() / count
}

pub fn symmetric_divisor(stat: f32) -> f32 {
    if stat > 0.0 {
        stat / INT8_SYMMETRIC_QUANTIZATION_MAXIMUM
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

pub fn asymmetric_scale_zero_point(
    values: &[f32],
    statistic: ActivationScaleStatistic,
) -> (f32, i8) {
    match statistic {
        ActivationScaleStatistic::AbsMax => {
            let (min, max) = group_min_max(values);
            if !(min.is_finite() && max.is_finite()) || max <= min {
                return (1.0, 0);
            }
            let scale = (max - min) / (INT8_ASYMMETRIC_QUANTIZATION_MAXIMUM - INT8_ASYMMETRIC_QUANTIZATION_MINIMUM);
            let zp = (INT8_ASYMMETRIC_QUANTIZATION_MINIMUM - min / scale)
                .round()
                .clamp(INT8_ASYMMETRIC_QUANTIZATION_MINIMUM, INT8_ASYMMETRIC_QUANTIZATION_MAXIMUM)
                as i8;
            (scale.max(f32::EPSILON), zp)
        },
        ActivationScaleStatistic::Rms => {
            let mean = group_mean(values);
            let rms = group_stat(values, ActivationScaleStatistic::Rms);
            let scale = if rms > 0.0 {
                rms / INT8_SYMMETRIC_QUANTIZATION_MAXIMUM
            } else {
                1.0
            };
            let zp = (-mean / scale)
                .round()
                .clamp(INT8_ASYMMETRIC_QUANTIZATION_MINIMUM, INT8_ASYMMETRIC_QUANTIZATION_MAXIMUM)
                as i8;
            (scale, zp)
        },
    }
}

pub fn quantize_asymmetric_i8(
    value: f32,
    scale: f32,
    zero_point: i8,
) -> i8 {
    ((value / scale).round() as i32 + i32::from(zero_point))
        .clamp(INT8_ASYMMETRIC_QUANTIZATION_MINIMUM as i32, INT8_ASYMMETRIC_QUANTIZATION_MAXIMUM as i32) as i8
}

pub fn compute_b_col_sums(
    weights: &[u8],
    n: usize,
    k: usize,
    group_size: usize,
) -> Vec<i32> {
    let groups = k.div_ceil(group_size);
    let mut sums = vec![0i32; n * groups];
    for col in 0..n {
        for group in 0..groups {
            let start = group * group_size;
            let end = (start + group_size).min(k);
            let mut sum = 0i32;
            for inner in start..end {
                sum += i32::from((weights[col * k + inner] ^ 0x80) as i8);
            }
            sums[col * groups + group] = sum;
        }
    }
    sums
}
