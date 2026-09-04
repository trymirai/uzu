use uzu_engine_macros::uzu_config;

use crate::config::{
    linear::LinearConfig, normalization::NormalizationConfig, rope::AnyRoPEConfig,
    transformer_layer::TransformerLayerConfig,
};

#[uzu_config]
pub struct DFlashDraftConfig {
    pub model_dim: u32,
    pub hidden_dim: u32,
    pub block_size: u32,
    pub mask_token_id: u64,
    pub target_layer_ids: Box<[u32]>,
    pub num_target_layers: u32,
    pub vocab_size: u32,
    pub context_projection_config: LinearConfig,
    pub context_norm_config: NormalizationConfig,
    pub rope_config: AnyRoPEConfig,
    pub layer_configs: Box<[TransformerLayerConfig]>,
    pub output_norm_config: NormalizationConfig,
}
