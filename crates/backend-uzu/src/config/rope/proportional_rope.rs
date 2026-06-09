use proc_macros::uzu_config;

#[uzu_config(super::RoPEConfig)]
pub struct ProportionalRoPEConfig {
    pub partial_rotary_factor: f32,
}
