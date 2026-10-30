use uzu_engine_macros::uzu_config;

#[uzu_config]
pub struct ConfidenceHeadTreeAttentionConfig {
    pub model_dim: u32,
    pub num_layers: u32,
    pub num_heads: u32,
    pub hidden_dim: u32,
}
