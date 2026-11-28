use uzu_engine_macros::uzu_config_abstract;

pub mod linear_scaling_rope;
pub mod llama_rope;
pub mod longrope;
pub mod multimodal_rope;
pub mod unscaled_rope;
pub mod yarn_rope;

#[uzu_config_abstract(
    unscaled_rope::UnscaledRoPEConfig,
    llama_rope::LlamaRoPEConfig,
    yarn_rope::YARNRoPEConfig,
    linear_scaling_rope::LinearScalingRoPEConfig,
    longrope::LongRoPEConfig,
    multimodal_rope::MultimodalRoPEConfig
)]
pub struct RoPEConfig {
    pub base: f32,
    pub max_sequence_length: u32,
    pub head_dim: u32,
}
