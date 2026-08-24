use backend_uzu_macros::uzu_config;

#[uzu_config(super::RoPEConfig)]
pub struct LinearScalingRoPEConfig {
    pub scaling_factor: f32,
}
