use serde::{Deserialize, Serialize};

use crate::backends::common::gpu_types::ActivationType;

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum ActivationConfig {
    #[serde(rename = "SiLU")]
    SILU {
        #[serde(default = "default_silu_alpha")]
        alpha: f32,
    },
    #[serde(rename = "GELU")]
    GELU,
    #[serde(rename = "Identity")]
    IDENTITY,
}

impl ActivationConfig {
    pub fn silu_default() -> ActivationConfig {
        ActivationConfig::SILU {
            alpha: 1.0,
        }
    }

    pub fn act_type(&self) -> ActivationType {
        match self {
            ActivationConfig::SILU {
                alpha,
            } => ActivationType::SILU {
                alpha: *alpha,
            },
            ActivationConfig::GELU => ActivationType::GELU,
            ActivationConfig::IDENTITY => ActivationType::IDENTITY,
        }
    }

    pub fn alpha(&self) -> f32 {
        match self {
            ActivationConfig::SILU {
                alpha,
            } => *alpha,
            ActivationConfig::GELU => 1.0,
            ActivationConfig::IDENTITY => 1.0,
        }
    }
}

fn default_silu_alpha() -> f32 {
    1.0
}
