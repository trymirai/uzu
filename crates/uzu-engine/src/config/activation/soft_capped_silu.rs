use uzu_engine_macros::uzu_config;

/// SiLU whose linear factor is soft-capped while the sigmoid sees the raw input:
/// `soft_cap * tanh(x / soft_cap) * sigmoid(x)`, evaluated in float32.
#[uzu_config(super::Activation)]
pub struct SoftCappedSiLU {
    pub soft_cap: f32,
}
