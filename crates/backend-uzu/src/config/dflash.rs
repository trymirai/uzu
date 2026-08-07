use proc_macros::uzu_config;

use crate::config::{
    linear::LinearConfig, normalization::NormalizationConfig, rope::AnyRoPEConfig,
    transformer_layer::TransformerLayerConfig,
};

#[uzu_config]
pub struct DFlashDraftConfig {
    pub model_dim: usize,
    pub hidden_dim: usize,
    pub block_size: usize,
    pub mask_token_id: u64,
    pub target_layer_ids: Box<[usize]>,
    pub num_target_layers: usize,
    pub vocab_size: usize,
    pub context_projection_config: LinearConfig,
    pub context_norm_config: NormalizationConfig,
    pub rope_config: AnyRoPEConfig,
    pub layer_configs: Box<[TransformerLayerConfig]>,
    pub output_norm_config: NormalizationConfig,
}
