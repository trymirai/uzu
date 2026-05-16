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
                ..
            } => ActivationType::SILU,
            ActivationConfig::GELU {
                ..
            } => ActivationType::GELU,
            ActivationConfig::IDENTITY => ActivationType::IDENTITY,
        }
    }

    pub fn alpha(&self) -> f32 {
        match self {
            ActivationConfig::SILU {
                alpha,
            } => *alpha,
            ActivationConfig::GELU {
                ..
            } => 1.0,
            ActivationConfig::IDENTITY => 1.0,
        }
    }
}
