use uzu_engine_macros::uzu_config;

use crate::{
    backends::common::gpu_types::vision::CHANNELS,
    config::{
        linear::LinearConfig, mlp::ungated_mlp::UngatedMLPConfig, normalization::NormalizationConfig,
        token_mixer::attention::AttentionConfig,
    },
};

#[uzu_config]
pub struct InterpolatedPositionEmbeddingConfig {
    pub num_positions: u32,
}

#[uzu_config]
pub struct VisionRoPEConfig {
    #[serde(rename = "type")]
    ty: monostate::MustBe!("VisionRoPEConfig"),
    pub base: f32,
    pub head_dim: u32,
}

#[uzu_config]
pub struct VisionLayerConfig {
    pub pre_attention_norm_config: NormalizationConfig,
    pub attention_config: AttentionConfig,
    pub pre_mlp_norm_config: NormalizationConfig,
    pub mlp_config: UngatedMLPConfig,
}

#[uzu_config]
pub struct VisionEncoderConfig {
    pub patch_embedding_config: LinearConfig,
    pub position_embedding_config: InterpolatedPositionEmbeddingConfig,
    pub rope_config: VisionRoPEConfig,
    pub layer_configs: Box<[VisionLayerConfig]>,
    pub merger_norm_config: NormalizationConfig,
    pub merger_mlp_config: UngatedMLPConfig,
    pub in_channels: u32,
    pub temporal_patch_size: u32,
    pub patch_size: u32,
    pub spatial_merge_size: u32,
    pub model_dim: u32,
    pub hidden_dim: u32,
    pub output_dim: u32,
}

#[uzu_config]
pub struct ImagePreprocessingConfig {
    pub min_pixels: u32,
    pub max_pixels: u32,
    pub image_mean: [f32; CHANNELS as usize],
    pub image_std: [f32; CHANNELS as usize],
}

#[uzu_config]
pub struct VisionConfig {
    pub encoder_config: VisionEncoderConfig,
    pub preprocessing_config: ImagePreprocessingConfig,
}
