use uzu_engine_macros::uzu_config;

use crate::{
    config::{
        decoder::DecoderConfig, linear::LinearConfig, mlp::ungated::UngatedMLPConfig,
        normalization::NormalizationConfig,
    },
    engine::language_model::GenerationConfig,
};

#[uzu_config]
pub struct VisionPatchEmbeddingConfig {
    pub linear_config: LinearConfig,
    pub in_channels: u32,
    pub temporal_patch_size: u32,
    pub patch_size: u32,
}

#[uzu_config]
pub struct VisionPositionEmbeddingConfig {
    pub num_positions: u32,
}

#[uzu_config]
pub struct VisionRoPEConfig {
    pub base: f32,
    pub head_dim: u32,
}

#[uzu_config]
pub struct VisionAttentionConfig {
    pub qkv_projection_config: LinearConfig,
    pub out_projection_config: LinearConfig,
    pub num_heads: u32,
    pub head_dim: u32,
    pub has_biases: bool,
}

#[uzu_config]
pub struct VisionLayerConfig {
    pub pre_attention_norm_config: NormalizationConfig,
    pub attention_config: VisionAttentionConfig,
    pub pre_mlp_norm_config: NormalizationConfig,
    pub mlp_config: UngatedMLPConfig,
}

#[uzu_config]
pub struct VisionPatchMergerConfig {
    pub norm_config: NormalizationConfig,
    pub mlp_config: UngatedMLPConfig,
}

#[uzu_config]
pub struct VisionEncoderConfig {
    pub patch_embedding_config: VisionPatchEmbeddingConfig,
    pub position_embedding_config: VisionPositionEmbeddingConfig,
    pub rope_config: VisionRoPEConfig,
    pub layer_configs: Box<[VisionLayerConfig]>,
    pub merger_config: VisionPatchMergerConfig,
    pub spatial_merge_size: u32,
    pub model_dim: u32,
    pub hidden_dim: u32,
    pub output_dim: u32,
}

#[uzu_config]
pub struct VisionImagePreprocessingConfig {
    pub patch_size: u32,
    pub temporal_patch_size: u32,
    pub spatial_merge_size: u32,
    pub min_pixels: u32,
    pub max_pixels: u32,
    pub image_mean: Box<[f32]>,
    pub image_std: Box<[f32]>,
}

#[uzu_config(super::ModelConfig)]
pub struct VisionLanguageModelConfig {
    pub decoder_config: DecoderConfig,
    pub generation_config: GenerationConfig,
    pub vision_encoder_config: VisionEncoderConfig,
    pub image_preprocessing_config: VisionImagePreprocessingConfig,
    pub image_token_id: u64,
}
