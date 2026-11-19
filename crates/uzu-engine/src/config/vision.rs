use uzu_engine_macros::uzu_config;

use crate::config::{
    activation::AnyActivation, linear::LinearConfig, normalization::NormalizationConfig,
    token_mixer::attention::AttentionConfig,
};

#[uzu_config]
pub struct UngatedMLPConfig {
    pub linear_config: LinearConfig,
    pub activation: AnyActivation,
    pub has_up_biases: bool,
    pub has_down_biases: bool,
}

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
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

#[uzu_config]
pub struct VisionConfig {
    pub encoder_config: VisionEncoderConfig,
    pub preprocessing_config: ImagePreprocessingConfig,
}

impl VisionConfig {
    pub fn image_token_count(
        &self,
        grid_thw: [u32; 3],
    ) -> u32 {
        let [frames, height, width] = grid_thw;
        frames * height * width / self.encoder_config.spatial_merge_size.pow(2)
    }
}
