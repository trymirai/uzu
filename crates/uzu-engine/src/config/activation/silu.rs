use uzu_engine_macros::uzu_config;

#[uzu_config(super::Activation)]
pub struct SiLU {
    pub alpha: f32,
    pub soft_cap: Option<f32>,
}

impl Default for SiLU {
    fn default() -> Self {
        Self {
            ty: Default::default(),
            alpha: 1.0,
            soft_cap: None,
        }
    }
}
