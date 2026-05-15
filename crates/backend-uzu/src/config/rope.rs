use serde::{Deserialize, Serialize};

use crate::ConfigDataType;

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct RopeConfigCommon {
    pub precision: ConfigDataType,
    pub base: f32,
    pub max_sequence_length: usize,
    pub head_dim: usize,
    pub partial_rotary_dim: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
#[serde(tag = "type")]
pub enum RoPEConfig {
    #[serde(rename = "UnscaledRoPEConfig")]
    Unscaled(RopeConfigCommon),
    #[serde(rename = "LlamaRoPEConfig")]
    Llama {
        #[serde(flatten)]
        common: RopeConfigCommon,
        scaling_factor: f32,
        original_context_length: usize,
        low_frequency_factor: f32,
        high_frequency_factor: f32,
    },
    #[serde(rename = "YARNRoPEConfig")]
    YARN {
        #[serde(flatten)]
        common: RopeConfigCommon,
        scaling_factor: f32,
        original_context_length: usize,
        beta_fast: f32,
        beta_slow: f32,
        truncate: bool,
    },
    #[serde(rename = "LinearScalingRoPEConfig")]
    LinearScaling {
        #[serde(flatten)]
        common: RopeConfigCommon,
        scaling_factor: f32,
    },
}

impl RoPEConfig {
    pub fn common(&self) -> &RopeConfigCommon {
        match self {
            RoPEConfig::Unscaled(config) => config,
            RoPEConfig::Llama {
                common,
                ..
            } => common,
            RoPEConfig::YARN {
                common,
                ..
            } => common,
            RoPEConfig::LinearScaling {
                common,
                ..
            } => common,
        }
    }
}
