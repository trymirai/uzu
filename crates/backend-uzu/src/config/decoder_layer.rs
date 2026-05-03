use serde::{Deserialize, Serialize};

use super::{
    RoPEConfig, attention::AttentionConfig, delta_net::DeltaNetAttentionConfig, mamba::Mamba2Config, mlp::MLPConfig,
    normalization::NormalizationConfig, short_conv::ShortConvConfig,
};

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
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

    pub fn as_mamba(&self) -> Option<&Mamba2Config> {
        match self {
            MixerConfig::Mamba(config) => Some(config),
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

    pub fn num_heads(&self) -> Option<usize> {
        match self {
            MixerConfig::Attention(config) => config.num_heads,
            MixerConfig::Mamba(config) => Some(config.num_heads),
            MixerConfig::ShortConv(_) => None,
            MixerConfig::DeltaNet(config) => Some(config.num_heads),
        }
    }

    pub fn num_groups(&self) -> Option<usize> {
        match self {
            MixerConfig::Attention(config) => config.num_groups,
            MixerConfig::Mamba(config) => Some(config.num_groups),
            MixerConfig::ShortConv(_) => None,
            MixerConfig::DeltaNet(config) => Some(config.num_groups),
        }
    }

    pub fn head_dim(&self) -> Option<usize> {
        match self {
            MixerConfig::Attention(config) => config.head_dim,
            MixerConfig::Mamba(config) => Some(config.head_dim),
            MixerConfig::ShortConv(_) => None,
            MixerConfig::DeltaNet(config) => Some(config.head_dim),
        }
    }

    pub fn sliding_window_size(&self) -> Option<usize> {
        match self {
            MixerConfig::Attention(config) => config.sliding_window_size,
            MixerConfig::Mamba(_) | MixerConfig::ShortConv(_) | MixerConfig::DeltaNet(_) => None,
        }
    }

    pub fn attention_scale(&self) -> Option<f32> {
        match self {
            MixerConfig::Attention(config) => config.scale,
            MixerConfig::Mamba(_) | MixerConfig::ShortConv(_) | MixerConfig::DeltaNet(_) => None,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
pub struct DecoderLayerConfig {
    #[serde(alias = "pre_mixer_norm_config")]
    pub pre_attention_norm_config: NormalizationConfig,
    #[serde(alias = "mixer_config")]
    pub mixer_config: MixerConfig,
    #[serde(alias = "post_mixer_norm_config")]
    pub post_attention_norm_config: Option<NormalizationConfig>,
    #[serde(alias = "pre_mlp_norm_config")]
    pub pre_mlp_norm_config: NormalizationConfig,
    pub mlp_config: MLPConfig,
    #[serde(alias = "post_mlp_norm_config")]
    pub post_mlp_norm_config: Option<NormalizationConfig>,
    #[serde(default)]
    pub rope_config: Option<RoPEConfig>,
}

impl DecoderLayerConfig {
    pub fn attention_config(&self) -> Option<&AttentionConfig> {
        self.mixer_config.as_attention()
    }
}

#[cfg(test)]
#[path = "../../tests/unit/config/decoder_layer_test.rs"]
mod tests;
