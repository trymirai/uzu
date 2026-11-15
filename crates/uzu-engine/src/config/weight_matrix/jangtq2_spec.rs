use uzu_engine_macros::uzu_config;

/// JANGTQ (jangtq2) routed-expert quantization: LSB-first bitstream-packed codebook
/// weights with one f16 scale per output row and a hadamard32 activation rotation.
/// `alpha`/`beta` are the odd-cubic codebook coefficients for this bit width
/// (levels = u * (alpha + beta * u^2)); they ride on the spec so the bundle's
/// codebook manifest is the single source of truth.
#[uzu_config(super::WeightMatrixSpec)]
pub struct Jangtq2Spec {
    pub bits: u32,
    pub alpha: f32,
    pub beta: f32,
}
