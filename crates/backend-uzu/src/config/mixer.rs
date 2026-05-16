use proc_macros::uzu_config;

use super::{AttentionConfig, ConfigDataType, DeltaNetAttentionConfig, Mamba2Config, ShortConvConfig};

#[uzu_config]
#[serde(tag = "type")]
pub enum MixerConfig {
    #[serde(rename = "AttentionConfig")]
    Attention(AttentionConfig),
    #[serde(rename = "Mamba2Config")]
    Mamba(Mamba2Config),
    #[serde(rename = "ShortConvConfig")]
    ShortConv(ShortConvConfig),
    #[serde(rename = "DeltaNetAttentionConfig")]
    DeltaNet(DeltaNetAttentionConfig),
}

impl MixerConfig {
    pub fn as_attention(&self) -> Option<&AttentionConfig> {
        match self {
            MixerConfig::Attention(config) => Some(config),
            _ => None,
        }
    }

    pub fn as_short_conv(&self) -> Option<&ShortConvConfig> {
        match self {
            MixerConfig::ShortConv(config) => Some(config),
            _ => None,
        }
    }

    pub fn as_delta_net(&self) -> Option<&DeltaNetAttentionConfig> {
        match self {
            MixerConfig::DeltaNet(config) => Some(config),
            _ => None,
        }
    }

    pub fn sliding_window_size(&self) -> Option<usize> {
        match self {
            MixerConfig::Attention(config) => config.sliding_window_size,
            MixerConfig::Mamba(_) | MixerConfig::ShortConv(_) | MixerConfig::DeltaNet(_) => None,
        }
    }

    pub fn activation_precision(&self) -> ConfigDataType {
        match self {
            MixerConfig::Attention(c) => c.qkv_projection_config.activation_precision(),
            MixerConfig::Mamba(c) => c.in_projection_config.activation_precision(),
            MixerConfig::ShortConv(c) => c.in_projection_config.activation_precision(),
            MixerConfig::DeltaNet(c) => c.in_proj_config.activation_precision(),
        }
    }
}
