use proc_macros::uzu_config;

use crate::backends::common::gpu_types::ActivationType;

#[derive(Copy)]
#[uzu_config]
#[serde(tag = "type")]
pub enum ActivationConfig {
    #[serde(rename = "SiLU")]
    SILU {
        alpha: f32,
    },
    #[serde(rename = "GELU")]
    GELU {
        approximate: bool,
    },
    #[serde(rename = "Identity")]
    IDENTITY,
}

impl ActivationConfig {
    pub fn act_type(&self) -> ActivationType {
        match self {
            ActivationConfig::SILU {
                alpha: _,
            } => ActivationType::SILU,
            ActivationConfig::GELU {
                approximate: true,
            } => ActivationType::GELUApprox,
            ActivationConfig::GELU {
                approximate: false,
            } => ActivationType::GELUExact,
            ActivationConfig::IDENTITY => ActivationType::IDENTITY,
        }
    }

    pub fn alpha(&self) -> f32 {
        match self {
            ActivationConfig::SILU {
                alpha,
            } => *alpha,
            ActivationConfig::GELU {
                approximate: _,
            } => 1.0,
            ActivationConfig::IDENTITY => 1.0,
        }
    }
}
