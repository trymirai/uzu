use proc_macros::uzu_config;

#[uzu_config(super::Activation)]
pub struct SiLU {
    pub alpha: f32,
}
