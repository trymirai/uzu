use proc_macros::uzu_config;

#[uzu_config(super::RoPEConfig)]
pub struct YARNRoPEConfig {
    pub scaling_factor: f32,
    pub original_context_length: usize,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub truncate: bool,
}
