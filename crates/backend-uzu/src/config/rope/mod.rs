use proc_macros::uzu_config_abstract;

pub mod linear_scaling_rope;
pub mod llama_rope;
pub mod longrope;
pub mod unscaled_rope;
pub mod yarn_rope;

#[uzu_config_abstract(
    unscaled_rope::UnscaledRoPEConfig,
    llama_rope::LlamaRoPEConfig,
    yarn_rope::YARNRoPEConfig,
    linear_scaling_rope::LinearScalingRoPEConfig,
    longrope::LongRoPEConfig
)]
pub struct RoPEConfig {
    pub base: f32,
    pub max_sequence_length: usize,
    pub head_dim: usize,
}
