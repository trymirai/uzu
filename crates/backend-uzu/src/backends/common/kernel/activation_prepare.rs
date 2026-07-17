use crate::backends::common::gpu_types::{ActivationScaleStatistic, HADAMARD_TRANSFORM_BLOCK_SIZE};

pub const INT8_SYMMETRIC_QMAX: f32 = 127.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActivationPrepareConfig {
    pub enabled: bool,
    pub stat: ActivationScaleStatistic,
}

impl ActivationPrepareConfig {
    pub fn from_env() -> Self {
        Self {
            enabled: std::env::var("UZU_A8W8").is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true")),
            stat: match std::env::var("UZU_A8W8_STAT").ok().as_deref() {
                Some("rms") => ActivationScaleStatistic::Rms,
                _ => ActivationScaleStatistic::AbsMax,
            },
        }
    }

    pub const fn supports_group_size(
        self,
        group_size: usize,
    ) -> bool {
        self.enabled && group_size != 0 && group_size.is_multiple_of(HADAMARD_TRANSFORM_BLOCK_SIZE)
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

pub fn symmetric_divisor(stat: f32) -> f32 {
    if stat > 0.0 {
        stat / INT8_SYMMETRIC_QMAX
    } else {
        1.0
    }
}

pub fn quantize_symmetric_i8(
    value: f32,
    divisor: f32,
) -> i8 {
    (value / divisor).round().clamp(-INT8_SYMMETRIC_QMAX, INT8_SYMMETRIC_QMAX) as i8
}
