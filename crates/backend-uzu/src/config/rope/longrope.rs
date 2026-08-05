use proc_macros::uzu_config;

#[uzu_config(super::RoPEConfig)]
pub struct LongRoPEConfig {
    pub short_factor: Box<[f32]>,
    pub long_factor: Box<[f32]>,
    pub original_context_length: usize,
    pub scaling_factor: f32,
}
