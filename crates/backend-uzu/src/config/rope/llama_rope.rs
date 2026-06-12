use proc_macros::uzu_config;

#[uzu_config(super::RoPEConfig)]
pub struct LlamaRoPEConfig {
    pub scaling_factor: f32,
    pub original_context_length: usize,
    pub low_frequency_factor: f32,
    pub high_frequency_factor: f32,
}
