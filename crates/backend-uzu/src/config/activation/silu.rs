use proc_macros::uzu_config;

#[uzu_config(super::Activation)]
pub struct SiLU {
    pub alpha: f32,
}

impl SiLU {
    pub fn new(alpha: f32) -> Self {
        Self {
            ty: Default::default(),
            alpha,
        }
    }
}
