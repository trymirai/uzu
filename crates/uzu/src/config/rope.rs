use serde::{Deserialize, Serialize};

use super::common::ConfigDataType;

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct RopeConfigCommon {
    pub precision: ConfigDataType,
    pub base: f32,
    pub max_sequence_length: usize,
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
        low_frequency_factor: f32,
        high_frequency_factor: f32,
    },
    LinearScalingRoPEConfig {
        #[serde(flatten)]
        common: RopeConfigCommon,
        scaling_factor: f32,
    },
}

impl RoPEConfig {
    pub fn common(&self) -> &RopeConfigCommon {
        match self {
            RoPEConfig::Unscaled(config) => &config,
            RoPEConfig::Llama {
                common,
                ..
            } => common,
            RoPEConfig::YARN {
                common,
                ..
            } => common,
            RoPEConfig::LinearScalingRoPEConfig {
                common,
                ..
            } => common,
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::from_str;

    use super::*;

    #[test]
    fn test_rope_config() {
        let config_str = r#"
            {
                "type": "LlamaRoPEConfig",
                "precision": "bfloat16",
                "base": 500000.0,
                "max_sequence_length": 131072,
                "scaling_factor": 32.0,
                "original_context_length": 8192,
                "low_frequency_factor": 1.0,
                "high_frequency_factor": 4.0
            }
        "#;

        let ground_truth_config = RoPEConfig::Llama {
            common: RopeConfigCommon {
                precision: ConfigDataType::BFloat16,
                base: 500000.0,
                max_sequence_length: 131072,
            },
            scaling_factor: 32.0,
            original_context_length: 8192,
            low_frequency_factor: 1.0,
            high_frequency_factor: 4.0,
        };

        let deserialized_config: RoPEConfig = from_str(config_str).unwrap();
        assert_eq!(deserialized_config, ground_truth_config);
    }
}
